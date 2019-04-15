import torch
from torch import nn

device = "cpu"

SOS_token = 1


class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        # forward inputs through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # prepare encoder's final hidden layer to be the first hidden input to the decoder
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        # initialize decoder input with SOS token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        # initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # iteratively decode one word token at a time
        for _ in range(max_length):
            # forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # obtain most likely word token and it's softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # prepare current token to be next decoder input
            decoder_input = torch.unsqueeze(decoder_input, 0)

        return all_tokens, all_scores
