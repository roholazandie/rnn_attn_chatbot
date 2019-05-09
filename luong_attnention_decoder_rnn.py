import torch
from torch import nn
import torch.nn.functional as F

from atttention import Attention


class LuongAttnDecoderRNN(nn.Module):

    def __init__(self, attention_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super().__init__()

        self.attention_model = attention_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # define layers
        self.embedding = embedding
        self.embedding_droput = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attention = Attention(attention_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        embedded = self.embedding(input_step)
        embedded = self.embedding_droput(embedded)
        # forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # calculate the attention weights from the current GRU output
        attention_weights = self.attention(rnn_output, encoder_outputs)
        # multiply attention weight to encoder outputs to get new "weighted sum" context vector
        context = torch.bmm(attention_weights, encoder_outputs.transpose(0, 1))
        # concatenate weighted context vector and GRU output
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # predict the next word
        output = F.softmax(self.out(concat_output), dim=1)
        # return output and final hidden state
        return output, hidden
