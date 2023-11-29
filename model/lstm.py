import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class LSTM_attention(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        pretrained_weight,
        update_w2v,
        hidden_dim,
        num_layers,
        drop_keep_prob,
        n_class,
        bidirectional,
        use_pretrained = False,
        **kwargs
    ):
        super(LSTM_attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.n_class = n_class

        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        if use_pretrained:
            self.embedding.weight = nn.Parameter(torch.tensor(pretrained_weight, dtype=torch.float32))
            self.embedding.weight.requires_grad = update_w2v

        else:
            nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
            self.embedding.weight.requires_grad = True

        self.encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=num_layers,
            bidirectional=self.bidirectional,
            dropout=drop_keep_prob,
        )

        self.weight_W = nn.Parameter(torch.Tensor(2 * hidden_dim, 2 * hidden_dim))
        self.weight_proj = nn.Parameter(torch.Tensor(2 * hidden_dim, 1))

        if self.bidirectional:
            self.decoder1 = nn.Linear(hidden_dim * 2, hidden_dim)
            self.decoder2 = nn.Linear(hidden_dim, n_class)
        else:
            self.decoder1 = nn.Linear(hidden_dim, hidden_dim)
            self.decoder2 = nn.Linear(hidden_dim, n_class)

        nn.init.uniform_(self.weight_W, -0.1, 0.1)
        nn.init.uniform_(self.weight_proj, -0.1, 0.1)

    def forward(self, inputs):

        embeddings = self.embedding(inputs.data)
        packed_states, hidden = self.encoder(embeddings)
        
        u = torch.tanh(torch.matmul(packed_states.data, self.weight_W))
        att = torch.matmul(u, self.weight_proj)
        att_score = F.softmax(att, dim=1)
        scored_x = packed_states.data * att_score
        packed_scored_x = torch.nn.utils.rnn.PackedSequence(scored_x, inputs.batch_sizes)
        unpacked_scored_x, _ = pad_packed_sequence(packed_scored_x, batch_first=True)
        encoding = torch.sum(unpacked_scored_x, dim=1)
        outputs = self.decoder1(encoding)
        outputs = self.decoder2(outputs)
        return outputs