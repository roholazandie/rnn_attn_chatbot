import torch
from torch import nn, optim
import os

from config import Config
from encoder_rnn import EncoderRNN
from luong_attnention_decoder_rnn import LuongAttnDecoderRNN
from prepare_data import PrepareData
from training import training_iters
import fire


# configure models
# attn_model = 'general'
# attn_model = 'concat'


def run_training(corpus_dir, save_dir, datafile, config_file, load_filename=""):
    # read data
    # corpus_dir = "cornell_movie_dialogs_corpus"
    # save_dir = os.path.join(corpus_dir, "save")
    # datafile = os.path.join(corpus_dir, "formatted_movie_lines.txt")
    config = Config.from_json_file(config_file)
    prepare_data = PrepareData(min_count=config.MIN_COUNT, max_length=config.MAX_LENGTH)
    vocab, pairs = prepare_data.load_prepare_data(corpus_dir, datafile, save_dir)

    # set checkpoint to load from; set to None if starting from scratch
    # load_filename = os.path.join(save_dir, model_name, corpus_name, '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
    #                            '{}_checkpoint.tar'.format(checkpoint_iter))

    if load_filename:
        # if loading on the same machine the model trained on
        checkpoint = torch.load(load_filename)
        # if loading a model trained on gpu to cpu
        # checkpoint = torch.load(load_filename, map_location=torch.device('cpu'))
        encoder_sd = checkpoint["enc"]
        decoder_sd = checkpoint["dec"]
        encoder_optimizer_sd = checkpoint["en_opt"]
        decoder_optimizer_sd = checkpoint["de_opt"]
        embedding_sd = checkpoint["embedding"]
        vocab.__dict__ = checkpoint["vocab_dict"]

    print("Building encoder and decoder ...")
    # initialize word embeddings
    embedding = nn.Embedding(vocab.num_words, config.hidden_size)
    if load_filename:
        embedding.load_state_dict(embedding_sd)

    # initialize encoder and decoder models
    encoder = EncoderRNN(config.hidden_size, embedding, config.encoder_n_layers, config.dropout)
    decoder = LuongAttnDecoderRNN(config.attn_model, embedding, config.hidden_size,
                                  vocab.num_words, config.decoder_n_layers, config.dropout)

    if load_filename:
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)

    print("Models built and ready to go.")

    #####################################
    # ensure dropout layers are in train mode
    encoder.train()
    decoder.train()

    # initilize optimizers
    print("building optimizers")
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=config.learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=config.learning_rate * config.decoder_learning_ratio)

    if load_filename:
        encoder_optimizer.load_state_dict(encoder_optimizer_sd)
        decoder_optimizer.load_state_dict(decoder_optimizer_sd)

    # run training iterations
    training_iters(config, vocab, pairs, encoder, decoder, encoder_optimizer,
                   decoder_optimizer, embedding, save_dir)


if __name__ == "__main__":
    # run_training(corpus_dir, save_dir, datafile, config_file)
    fire.Fire()
