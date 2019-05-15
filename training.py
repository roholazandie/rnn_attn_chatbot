import torch
import random
import os
from cornell_movie_dialog_dataset import CornellMovieDialogDataset
from prepare_data import PrepareDataForModel
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from loss import compute_loss

def masked_NLL_loss(inp, target, mask):
    n_total = mask.sum()
    cross_entropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = cross_entropy.masked_select(mask).mean()
    return loss, n_total.item()

def batched_masked_nll_loss(logits, target, mask):
    pass

def train(config, vocab, input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding,
          encoder_optimizer, decoder_optimizer):

    # zeros gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # set device options
    input_variable = input_variable.to(config.device)
    lengths = lengths.to(config.device)
    target_variable = target_variable.to(config.device)
    mask = mask.to(config.device)

    # initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # forward pass through the encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    #create initial decoder inputs
    decoder_input = torch.LongTensor([vocab.bos_id for _ in range(config.batch_size)])
    decoder_input = decoder_input.unsqueeze(0)
    decoder_input = decoder_input.to(config.device)

    # set initial decoder hidden state to encoder final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # determine whether we using teacher forcing this iteration or not
    use_teacher_forcing = True if random.random() < config.teacher_forcing_ratio else False

    # forward batch of sequences through decoder one time iteraion
    outputs = Variable(torch.zeros(max_target_len, config.batch_size, len(vocab))).cuda()
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_outputs, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            # teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
            outputs[t] = decoder_outputs

            #masked_loss = F.cross_entropy(decoder_outputs, target_variable[t], ignore_index=vocab.pad_id)
            n_total = mask[t].sum()
            #masked_loss, n_total = masked_NLL_loss(decoder_outputs, target_variable[t], mask[t])
            #print("cross entropy ", masked_loss.item())
            #print("nll ", masked_loss2.item())

            #loss += masked_loss

            # if torch.abs(masked_loss - masked_loss2).item() > 1:
                #print("wrong")
            #outputs.permute(1,0,2).contiguous().view(-1,outputs.shape[-1])
            #target_variable.permute(1,0).contiguous().view(-1,1)

            #print_losses.append(masked_loss.item() * n_total)
            n_totals += n_total
    else:
        for t in range(max_target_len):
            decoder_outputs, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            # no teacher forcing: next input is decoder's current output
            _, topi = decoder_outputs.topk(1)
            decoder_input = torch.LongTensor([topi[i][0] for i in range(config.batch_size)]).to(config.device)
            outputs[t] = decoder_outputs


            masked_loss = F.cross_entropy(decoder_outputs, target_variable[t], ignore_index=vocab.pad_id)
            n_total = mask[t].sum()
            #masked_loss, n_total = masked_NLL_loss(decoder_outputs, target_variable[t], mask[t])
            loss += masked_loss

            # # calculate and accumulate loss

            print_losses.append(masked_loss.item() * n_total)
            n_totals += n_total

    # calculate loss
    #masked_loss, n_total = masked_NLL_loss(outputs, target_variable, mask)

    # loss = F.cross_entropy(outputs[1:].permute(1, 0, 2).contiguous().view(-1, len(vocab)),
    #                        target_variable[1:].permute(1, 0).contiguous().view(-1), ignore_index=vocab.pad_id)
    #

    #torch.gather(outputs.view(-1,outputs.shape[-1]), 1, target_variable.view(-1,1))
    #-torch.log(torch.gather(outputs.view(-1,outputs.shape[-1]), 1, target_variable.view(-1,1)).squeeze(1))
    cross_entropy = -torch.log(torch.gather(outputs.view(-1, outputs.shape[-1]), 1, target_variable.view(-1, 1)).squeeze(1)).reshape(-1, config.batch_size)
    loss = cross_entropy.masked_select(mask).mean()
    loss.backward()


    # clip gradients: gradients are clipped in place
    clip_grad_norm_(encoder.parameters(), config.clip)
    clip_grad_norm_(decoder.parameters(), config.clip)

    # adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    loss_value = loss.item()

    return loss_value#sum(print_losses)/n_totals




def train1(config, bos_id, input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding,
          encoder_optimizer, decoder_optimizer):

    # zeros gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()


    # initialize variables
    loss = 0
    print_losses = []
    n_totals = 0




    # perform backprob
    loss.backward()

    # clip gradients: gradients are clipped in place
    torch.nn.utils.clip_grad_norm_(encoder.parameters(), config.clip)
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), config.clip)

    # adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses)/n_totals


def training_iters(config, voc, pairs, encoder, decoder, encoder_optimizer,
                   decoder_optimizer, embedding, save_dir, load_filename):

    # load batches for each iteration
    data = PrepareDataForModel()

    training_batches = [data.batch_to_traindata(voc, [random.choice(pairs) for _ in range(config.batch_size)])
                        for _ in range(config.n_iterations)]

    #initializing
    print("Initializing ...")
    start_iteration = 1
    print_loss = 0
    if load_filename:
        checkpoint = torch.load(load_filename)
        start_iteration = checkpoint["iteration"] + 1

    # training loop
    for iteration in range(start_iteration, config.n_iterations+1):
        training_batch = training_batches[iteration-1]
        # extract fields from batch
        input_variable, lengths, target_variable, mask, max_target_length = training_batch

        #run a training iteration
        loss = train(config,
                     voc,
                     input_variable,
                     lengths,
                     target_variable,
                     mask, max_target_length,
                     encoder, decoder, embedding,
                     encoder_optimizer, decoder_optimizer)

        print_loss += loss

        # print progress
        if iteration % config.print_every == 0:
            print_loss_avg = print_loss / config.print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration,
                                                                                          iteration / config.n_iterations * 100,
                                                                                          print_loss_avg))
            print_loss = 0

        # save checkpoints
        if iteration % config.save_every == 0:
            directory = os.path.join(save_dir, config.model_name, config.corpus_name, '{}-{}_{}'.format(config.encoder_n_layers,
                                                                                          config.decoder_n_layers,
                                                                                          config.hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)

            torch.save({"iteration": iteration,
                        "en": encoder.state_dict(),
                        "de": decoder.state_dict(),
                        "en_opt": encoder_optimizer.state_dict(),
                        "de_opt": decoder_optimizer.state_dict(),
                        "loss": loss,
                        "voc_dict": voc.__dict__,
                        "embedding": embedding.state_dict()},
                       os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))



def training_iters1(data_loader, config, voc, encoder, decoder, encoder_optimizer,
                   decoder_optimizer, embedding, save_dir, load_filename, epoch):
    #initializing
    print("Initializing ...")
    start_iteration = 1
    print_loss = 0
    if load_filename:
        checkpoint = torch.load(load_filename)
        start_iteration = checkpoint["iteration"] + 1

    # training loop
    data_length = len(data_loader)
    for iteration, training_batch in enumerate(data_loader):
        # extract fields from batch
        input_variable, lengths, target_variable, mask, max_target_length = training_batch

        #run a training iteration
        loss = train(config,
                     voc,
                     input_variable,
                     lengths,
                     target_variable,
                     mask, max_target_length,
                     encoder, decoder, embedding,
                     encoder_optimizer, decoder_optimizer)

        print_loss += loss

        # print progress
        if iteration % config.print_every == 0:
            print_loss_avg = print_loss / config.print_every
            print("Epoch:{}; Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(epoch, iteration,
                                                                                          iteration / data_length * 100,
                                                                                          print_loss_avg))
            print_loss = 0

        # save checkpoints
        if iteration % config.save_every == 0:
            directory = os.path.join(save_dir, config.model_name, config.corpus_name, '{}-{}_{}'.format(config.encoder_n_layers,
                                                                                          config.decoder_n_layers,
                                                                                          config.hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)

            torch.save({"iteration": iteration,
                        "en": encoder.state_dict(),
                        "de": decoder.state_dict(),
                        "en_opt": encoder_optimizer.state_dict(),
                        "de_opt": decoder_optimizer.state_dict(),
                        "loss": loss,
                        "voc_dict": voc.__dict__,
                        "embedding": embedding.state_dict()},
                       os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))











