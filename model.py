from typing import Dict

import torch
from torch.nn import Embedding
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=self.embed.weight.size()[1],
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True,
                            bidirectional=bidirectional)
        #self.drop = nn.Dropout(p=dropout)
        self.bidirectional = bidirectional
        self.fc = nn.Linear(self.encoder_output_size, num_class)
        self.softmax = nn.Softmax(dim=1)

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        #raise NotImplementedError
        return (int(self.bidirectional)+1) * self.hidden_size

    def forward(self, text, text_len) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        # ins = {"text":..., "intent":..., "id":..., "text_len":...}
        #raise NotImplementedError
        embeds = self.embed(text)
        output, _ = self.lstm(embeds)

        out_forward = output[range(len(output)), text_len - 1, :self.hidden_size]
        if self.bidirectional:
            out_reverse = output[:, 0, self.hidden_size:]
            out_forward = torch.cat((out_forward, out_reverse), 1)
        text_feat = out_forward

        out = self.fc(text_feat)
        out = self.softmax(out)
        return out


class SeqTagger(SeqClassifier):
    def __init__(self, embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,):

        super(SeqTagger, self).__init__(embeddings=embeddings, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, num_class=num_class)
        self.softmax = nn.Softmax(dim=2)
    def forward(self, text, text_len) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        # raise NotImplementedError
        embeds = self.embed(text)
        # B * maxlen * hidden size
        output, _ = self.lstm(embeds)
        # output = output[:, :, :self.hidden_size]
        # B * maxlen * num class
        output = self.fc(output)
        output = self.softmax(output)
        return output
