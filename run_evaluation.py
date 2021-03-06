from evaluate import evaluate_input, evaluate_input1
from greedy_search_decoder import GreedySearchDecoder
import torch
from torch import nn, optim
from config import Config
from encoder_rnn import EncoderRNN
from luong_attnention_decoder_rnn import LuongAttnDecoderRNN
from prepare_data import PrepareData
from text import BPEVocab
from training import training_iters
import os
import fire

from vocab import Vocabulary


def run_evaluation(corpus_dir, save_dir, datafile, config_file):
    config = Config.from_json_file(config_file)
    vocab = Vocabulary("words")

    # set checkpoint to load from; set to None if starting from scratch
    load_filename = os.path.join(save_dir, config.model_name, config.corpus_name, '{}-{}_{}'.format(config.encoder_n_layers,
                                                                                      config.decoder_n_layers,
                                                                                      config.hidden_size),
                               'last_checkpoint.tar')

    # if loading on the same machine the model trained on
    checkpoint = torch.load(load_filename)
    # if loading a model trained on gpu to cpu
    # checkpoint = torch.load(load_filename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint["en"]
    decoder_sd = checkpoint["de"]
    encoder_optimizer_sd = checkpoint["en_opt"]
    decoder_optimizer_sd = checkpoint["de_opt"]
    embedding_sd = checkpoint["embedding"]
    vocab.__dict__ = checkpoint["voc_dict"]

    print("Building encoder and decoder ...")
    # initialize word embeddings
    embedding = nn.Embedding(vocab.num_words, config.hidden_size)
    embedding.load_state_dict(embedding_sd)

    # initialize encoder and decoder models
    encoder = EncoderRNN(config.hidden_size, embedding, config.encoder_n_layers, config.dropout)
    decoder = LuongAttnDecoderRNN(config.attn_model, embedding, config.hidden_size,
                                  vocab.num_words, config.decoder_n_layers, config.dropout)

    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)

    # Set dropout layers to eval mode

    encoder.eval()
    decoder.eval()

    # Initialize search module
    searcher = GreedySearchDecoder(encoder, decoder)

    # Begin chatting (uncomment and run the following line to begin)
    evaluate_input(encoder, decoder, searcher, vocab)


def run_evaluation1(corpus_dir, save_dir, datafile, config_file):
    config = Config.from_json_file(config_file)
    save_dir = config.save_dir
    # set checkpoint to load from; set to None if starting from scratch
    load_filename = os.path.join(save_dir, config.model_name, config.corpus_name,
                                 '{}-{}_{}'.format(config.encoder_n_layers,
                                                   config.decoder_n_layers,
                                                   config.hidden_size),
                                 "last_checkpoint.tar")


    # if loading on the same machine the model trained on
    checkpoint = torch.load(load_filename, map_location=lambda storage, loc: storage)
    # if loading a model trained on gpu to cpu
    # checkpoint = torch.load(load_filename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint["en"]
    decoder_sd = checkpoint["de"]
    encoder_optimizer_sd = checkpoint["en_opt"]
    decoder_optimizer_sd = checkpoint["de_opt"]
    embedding_sd = checkpoint["embedding"]

    vocab = BPEVocab.from_files(config.bpe_vocab_path, config.bpe_codes_path)
    print("Building encoder and decoder ...")
    # initialize word embeddings
    embedding = nn.Embedding(len(vocab), config.hidden_size)
    embedding.load_state_dict(embedding_sd)

    # initialize encoder and decoder models
    encoder = EncoderRNN(config.hidden_size, embedding, config.encoder_n_layers, config.dropout)
    decoder = LuongAttnDecoderRNN(config.attn_model, embedding, config.hidden_size,
                                  len(vocab), config.decoder_n_layers, config.dropout)

    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)

    # Set dropout layers to eval mode

    encoder.eval()
    decoder.eval()

    # Initialize search module
    searcher = GreedySearchDecoder(encoder, decoder)

    # Begin chatting (uncomment and run the following line to begin)
    evaluate_input1(encoder, decoder, searcher, vocab)


if __name__ == "__main__":
    #fire.Fire()
    corpus_dir = "cornell_movie_dialogs_corpus"
    save_dir = "cornell_movie_dialogs_corpus/save"
    datafile = "cornell_movie_dialogs_corpus/formatted_movie_lines.txt"
    config_file = "config.json"
    run_evaluation1(corpus_dir, save_dir, datafile, config_file)
    #run_evaluation(corpus_dir, save_dir, datafile, config_file)