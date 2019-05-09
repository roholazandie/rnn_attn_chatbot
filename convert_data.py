from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import codecs
import csv
import os
from io import open
import torch
import fire

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


def print_lines(file, n=10):
    with open(file, 'rb') as file_reader:
        lines = file_reader.readlines()
    for line in lines:
        print(line)


def load_lines(filename, fields):
    lines = {}
    with open(filename, 'r', encoding="iso-8859-1") as file_reader:
        for line in file_reader:
            values = line.split(" +++$+++ ")
            # extract fields
            line_obj = {}
            for i, field in enumerate(fields):
                line_obj[field] = values[i]

            lines[line_obj["lineID"]] = line_obj

    return lines


def load_conversations(filename, lines, fields):
    conversations = []
    with open(filename, 'r', encoding="iso-8859-1") as file_reader:
        for line in file_reader:
            values = line.split(" +++$+++ ")
            # extract fields
            conv_obj = {}
            for i, field in enumerate(fields):
                conv_obj[field] = values[i]

            line_ids = eval(conv_obj["utteranceIDs"])
            conv_obj["lines"] = []
            for line_id in line_ids:
                conv_obj["lines"].append(lines[line_id])

            conversations.append(conv_obj)

    return conversations

def extract_sentence_pairs(conversations):
    qa_pairs = []
    for conversation in conversations:
        for i in range(len(conversation["lines"])-1):# we ignore the last line, no answer for it
            input_line = conversation["lines"][i]["text"].strip()
            target_line = conversation["lines"][i+1]["text"].strip()
            # filter wrong samples
            if input_line and target_line: #shouldn't be empty
                qa_pairs.append([input_line, target_line])

    return qa_pairs


def prepare_data(corpus_dir, corpus_file, conversations_corpus_file):

    corpus_file = os.path.join(corpus_dir, corpus_file)
    conversations_corpus_file = os.path.join(corpus_dir, conversations_corpus_file)
    # Define path to new file
    datafile = os.path.join(corpus_dir, "formatted_movie_lines.txt")

    delimiter = '\t'
    # Unescape the delimiter
    delimiter = str(codecs.decode(delimiter, "unicode_escape"))

    # Initialize lines dict, conversations list, and field ids
    lines = {}
    conversations = []
    MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
    MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]

    # Load lines and process conversations
    print("\nProcessing corpus...")
    lines = load_lines(corpus_file, MOVIE_LINES_FIELDS)

    print("\nLoading conversations...")
    conversations = load_conversations(conversations_corpus_file,
                                       lines, MOVIE_CONVERSATIONS_FIELDS)

    # Write new csv file
    print("\nWriting newly formatted file...")
    with open(datafile, 'w', encoding='utf-8') as outputfile:
        writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
        for pair in extract_sentence_pairs(conversations):
            writer.writerow(pair)

    print_lines(datafile)


if __name__ == "__main__":
    fire.Fire()