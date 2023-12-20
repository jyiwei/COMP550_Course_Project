import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch

from utils import CustomError

class LSTM_attention(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_dim,
                 num_layers=1,
                 drop_keep_prob=0.2,
                 n_class=2,
                 bidirectional=True,
                 pretrained_weight=None,
                 update_w2v=True,
                ):
        
        super(LSTM_attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.n_class = n_class

        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        if pretrained_weight is not None:
            self.embedding.weight = nn.Parameter(torch.tensor(pretrained_weight, dtype=torch.float32))
            self.embedding.weight.requires_grad = update_w2v

        self.encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=num_layers,
            bidirectional=self.bidirectional,
            dropout=drop_keep_prob,
            batch_first=True
        )

        self.attention_W = nn.Linear(2 * hidden_dim if bidirectional else hidden_dim, 2 * hidden_dim if bidirectional else hidden_dim, bias=False)
        self.attention_proj = nn.Linear(2 * hidden_dim if bidirectional else hidden_dim, 1, bias=False)

        if self.bidirectional:
            self.dense = nn.Linear(2 * hidden_dim, n_class)
        else:
            self.dense = nn.Linear(hidden_dim, n_class)

    def forward(self, inputs):

        lengths = torch.sum(inputs != 0, dim=1)
        # lengths = torch.clamp(lengths, min=1)

        embeddings = self.embedding(inputs.data)
        packed_embeddings = pack_padded_sequence(embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False)

        packed_output, (hidden, cell) = self.encoder(packed_embeddings)

        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        u = torch.tanh(self.attention_W(output))
        att = self.attention_proj(u)
        att_score = F.softmax(att, dim=1)
        scored_x = output * att_score
        encoding = torch.sum(scored_x, dim=1)

        outputs = self.dense(encoding)

        return outputs
    
class LSTM_Model(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_dim,
                 num_layers=1,
                 drop_keep_prob=0.2,
                 n_class=2,
                 bidirectional=True,
                 pretrained_weight=None,
                 update_w2v=True,
                ):
        super(LSTM_Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        if pretrained_weight is not None:
            self.embedding.weight = nn.Parameter(torch.tensor(pretrained_weight, dtype=torch.float32))
            self.embedding.weight.requires_grad = update_w2v

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, dropout=drop_keep_prob, bidirectional = bidirectional)
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.dense = nn.Linear(lstm_output_dim, n_class)
            

    def forward(self, inputs):

        lengths = torch.sum(inputs != 0, dim=1)
        # lengths = torch.clamp(lengths, min=1)

        embedded = self.embedding(inputs)
        
        packed_embedded = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # final_hidden = output[torch.arange(output.size(0)), lengths-1]

        if self.lstm.bidirectional:
            final_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            final_hidden = hidden[-1]

        outputs = self.dense(final_hidden)
        
        return outputs

