from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals



class Vocabulary():

    def __init__(self, name):
        self.name = name
        self.trimmed = False

        # Default word tokens
        self.pad_id = 0  # Used for padding short sentences
        self.bos_id = 1  # Start-of-sentence token
        self.eos_id = 2  # End-of-sentence token

        self.word2index = {}
        self.word2count = {}
        self.index2word = {self.pad_id: "PAD", self.bos_id: "SOS", self.eos_id: "EOS"}
        self.num_words = 3  # the 3 basic ones

    # @property
    # def bos_id(self):
    #     return self.bos_id
    #
    # @property
    # def eos_id(self):
    #     return self.eos_id
    #
    # @property
    # def pad_id(self):
    #     return self.pad_id

    def __len__(self):
        return self.num_words

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print("num keep words: ", len(keep_words))

        # reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {self.pad_id: "PAD", self.bos_id: "SOS", self.eos_id: "EOS"}
        self.num_words = 3

        for word in keep_words:
            self.add_word(word)
