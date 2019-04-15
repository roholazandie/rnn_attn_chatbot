import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class EncoderRNN(nn.Module):

    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.dropout = dropout

        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers==1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # convert word indexes to word embedding vectors
        embedded = self.embedding(input_seq)
        # pack padded batch of sequences for rnn module
        packed = pack_padded_sequence(embedded, input_lengths)
        # forward pass throught gru
        outputs, hidden = self.gru(packed, hidden)
        # unpack padding
        outputs, _ = pad_packed_sequence(outputs)
        # sum bidirectional gru outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]

        return outputs, hidden

        

