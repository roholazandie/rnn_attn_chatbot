import torch

from prepare_data import PrepareDataForModel

MAX_LENGTH = 10
device = "cpu"

def evaluate(encoder, decoder, searcher, vocab, sentence, max_length=MAX_LENGTH):
    # word -> indexes
    data = PrepareDataForModel()
    indexes_batch = [data.index_from_sentence(vocab, sentence)]
    # create length tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # transpose dimensions of batch to match model's expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [vocab.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluate_input(encoder, decoder, searcher, vocab):
    input_sentence = ""
    while True:
        try:
            # get the input
            input_sentence = input(">")
            if input_sentence == 'q':
                break

            # normalize sentence
            output_words = evaluate(encoder, decoder, searcher, vocab, input_sentence)
            # format and print the response
            output_words[:] = [x for x in output_words if not (x=="EOS" or x=="SOS")]
            print(" ".join(output_words))

        except KeyError:
            raise ValueError("unknown word")

