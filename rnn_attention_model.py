import torch
from torch import nn
import random
from encoder_rnn import EncoderRNN
from luong_attnention_decoder_rnn import LuongAttnDecoderRNN


class RNNAttentionModel(nn.Module):

    def __init__(self, config, embedding, vocab, loss_criterion):
        super().__init__()
        self.encoder = EncoderRNN(config.hidden_size, embedding, config.encoder_n_layers, config.dropout)
        self.decoder = LuongAttnDecoderRNN(config.attn_model, embedding, config.hidden_size,
                                  vocab.num_words, config.decoder_n_layers, config.dropout)

        self.config = config
        self.bos_id = vocab.bos_id
        self.loss_criterion = loss_criterion


    def forward(self, input_variable, lengths, target_variable, mask, max_target_len):
        # initialize variables
        loss = 0
        print_losses = []
        n_totals = 0

        # forward pass through the encoder
        encoder_outputs, encoder_hidden = self.encoder(input_variable, lengths)

        # create initial decoder inputs
        decoder_input = torch.LongTensor([self.bos_id for _ in range(self.config.batch_size)])
        decoder_input = decoder_input.unsqueeze(0)

        # set initial decoder hidden state to encoder final hidden state
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]

        # determine whether we using teacher forcing this iteration or not
        use_teacher_forcing = True if random.random() < self.config.teacher_forcing_ratio else False

        # forward batch of sequences through decoder one time iteraion

        if use_teacher_forcing:
            for t in range(max_target_len):
                decoder_outputs, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                # teacher forcing: next input is current target
                decoder_input = target_variable[t].view(1, -1)
                # calculate and accumulate loss
                masked_loss, n_total = self.loss_criterion(decoder_outputs, target_variable[t], mask[t])
                loss += masked_loss
                print_losses.append(masked_loss.item() * n_total)
                n_totals += n_total
        else:
            for t in range(max_target_len):
                decoder_outputs, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                # no teacher forcing: next input is decoder's current output
                _, topi = decoder_outputs.topk(1)
                decoder_input = torch.LongTensor([topi[i][0] for i in range(self.config.batch_size)])

                # calculate and accumulate loss
                masked_loss, n_total = self.loss_criterion(decoder_outputs, target_variable[t], mask[t])
                loss += masked_loss
                print_losses.append(masked_loss.item() * n_total)
                n_totals += n_total

        return loss