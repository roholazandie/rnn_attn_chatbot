from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import itertools

import re
import os
import unicodedata
from io import open
import torch
import random

from text import SpacySentenceTokenizer
from vocab import Vocabulary

MAX_LENGTH = 10  # maximum sentence length to consider
MIN_COUNT = 3

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token


class PrepareData():

    def __init__(self, min_count=3, max_length=10):
        self.min_count = min_count
        self.max_length = max_length

    # Turn a Unicode string to plain ASCII
    def unicode_to_ascii(self, s):
        return ''.join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

    def normalize_string(self, s):
        s = self.unicode_to_ascii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        s = re.sub(r"\s+", r" ", s).strip()
        return s

    # read query/response pairs and return a voc object
    def read_vocabs(self, datafile, corpus_name):
        lines = open(datafile, encoding="utf-8").read().strip().split('\n')

        pairs = [[self.normalize_string(s) for s in line.split('\t')] for line in lines]
        vocab = Vocabulary(corpus_name)

        return vocab, pairs

    def read_pairs(self, datafile):
        tokenizer = SpacySentenceTokenizer()
        lines = open(datafile, encoding="utf-8").read().strip().split('\n')
        #pairs = [[self.normalize_string(s) for s in line.split('\t')] for line in lines]
        pairs = []
        i = 0
        for line in lines:
            chats = line.split('\t')
            #todo nomalize text
            sentences0 = chats[0]#tokenizer.tokenize(chats[0])
            sentences1 = chats[1]#tokenizer.tokenize(chats[1])
            pairs.append((sentences0, sentences1))
            # i+=1
            # if i>10:
            #     break

        return pairs

    # Returns True iff both sentences in a pair 'p' are under the MAX_LENGTH threshold
    def filter_pair(self, p):
        return len(p[0].split(' ')) < self.max_length and len(p[1].split(' ')) < self.max_length

    def filter_pairs(self, pairs):
        return [pair for pair in pairs if self.filter_pair(pair)]

    def trim_rare_words(self, vocab, pairs):
        vocab.trim(self.min_count)

        keep_pairs = []
        for pair in pairs:
            input_sentence = pair[0]
            output_sentence = pair[1]

            keep_input = True
            keep_output = True

            for word in input_sentence.split(' '):
                if word not in vocab.word2index:
                    keep_input = False
                    break

            for word in output_sentence.split(' '):
                if word not in vocab.word2index:
                    keep_output = False
                    break

            if keep_input and keep_output:
                keep_pairs.append(pair)

        print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs),
                                                                    len(keep_pairs) / len(pairs)))

        return keep_pairs

    def load_prepare_data(self, corpus_name, datafile, save_dir):
        vocab, pairs = self.read_vocabs(datafile, corpus_name)
        pairs = self.filter_pairs(pairs)

        for pair in pairs:
            vocab.add_sentence(pair[0])
            vocab.add_sentence(pair[1])

        pairs = self.trim_rare_words(vocab, pairs)

        return vocab, pairs



class PrepareDataForModel():

    def __init__(self):
        pass

    def index_from_sentence(self, vocab, sentence):
        return [vocab.word2index[word] for word in sentence.split(' ')] + [EOS_token]

    def zero_padding(self, l, fillvalue=PAD_token):
        return list(itertools.zip_longest(*l, fillvalue=fillvalue))

    def binary_matrix(self, l, value_pad=PAD_token):
        matrix = []

        for i, seq in enumerate(l):
            matrix.append([])
            for token in seq:
                if token is value_pad:
                    matrix[i].append(0)
                else:
                    matrix[i].append(1)

        return matrix

    # Returns padded input sequence tensor and lengths
    def input_var(self, l, vocab):
        indexes_batch = [self.index_from_sentence(vocab, sentence) for sentence in l]
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        pad_list = self.zero_padding(indexes_batch)
        pad_var = torch.LongTensor(pad_list)
        return pad_var, lengths

    def output_var(self, l, vocab):
        indexes_batch = [self.index_from_sentence(vocab, sentence) for sentence in l]
        max_target_length = max(len(indexes) for indexes in indexes_batch)
        pad_list = self.zero_padding(indexes_batch)
        mask = self.binary_matrix(pad_list)
        mask = torch.ByteTensor(mask)
        pad_var = torch.LongTensor(pad_list)
        return pad_var, mask, max_target_length

    def batch_to_traindata(self, vocab, pair_batch):
        pair_batch.sort(key=lambda x: len(x[0].split(' ')), reverse=True)
        input_batch, output_batch = [], []

        for pair in pair_batch:
            input_batch.append(pair[0])
            output_batch.append(pair[1])

        input_, lengths = self.input_var(input_batch, vocab)
        output, mask, max_target_length = self.output_var(output_batch, vocab)

        return input_, lengths, output, mask, max_target_length


if __name__ == "__main__":
    corpus_dir = "cornell_movie_dialogs_corpus"
    save_dir = os.path.join(corpus_dir, "save")
    datafile = os.path.join(corpus_dir, "formatted_movie_lines.txt")
    prepare_data = PrepareData(min_count=MIN_COUNT, max_length=MAX_LENGTH)
    vocab, pairs = prepare_data.load_prepare_data(corpus_dir, datafile, save_dir)

    for pair in pairs[:10]:
        print(pair)

    small_batch_size = 5
    prepare_data_for_model = PrepareDataForModel()
    rand_pairs = [random.choice(pairs) for _ in range(small_batch_size)]
    input_variable, lengths, target_variable, mask, max_target_length = prepare_data_for_model.batch_to_traindata(vocab,rand_pairs )

    print("input_variable:", input_variable)
    print("lengths:", lengths)
    print("target_variable:", target_variable)
    print("mask:", mask)
    print("max_target_len:", max_target_length)