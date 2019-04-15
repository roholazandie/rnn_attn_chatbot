import torch
from torch import nn
import torch.nn.functional as F


class Attention(nn.Module):

    def __init__(self, method, hidden_size):
        super().__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method not in ["dot", "general", "concat"]:
            raise ValueError("method" + str(self.method) + "is not defined")

        if self.method == "general":
            self.attention = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == "concat":
            self.attention = torch.nn.Linear(self.hidden_size*2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attention(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        a = encoder_output.size(0)
        energy = self.attention(torch.cat((hidden.expand(a, -1, -1), encoder_output),dim=2))
        return torch.sum(self.v * torch.tanh(energy), dim=2)

    def forward(self, hidden, encoder_outputs):
        #hidden (seq_len=1, batch, num_directions * hidden_size)
        #encoder_outputs (max_length, batch_size, hidden_size)
        if self.method == "general":
            attention_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == "concat":
            attention_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == "dot":
            attention_energies = self.dot_score(hidden, encoder_outputs)

        # transpose max_length and batch_size
        attention_energies = attention_energies.t()

        return F.softmax(attention_energies, dim=1).unsqueeze(1)

