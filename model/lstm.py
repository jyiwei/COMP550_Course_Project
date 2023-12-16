import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch

from utils import CustomError

class LSTM_attention(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_layers,
        drop_keep_prob,
        n_class,
        bidirectional,
        pretrained_weight = None,
        update_w2v = None,
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

        lengths = torch.sum(inputs != 0, dim=1)

        embeddings = self.embedding(inputs.data)
        packed_embeddings = pack_padded_sequence(embeddings, lengths.cpu(), batch_first=True)

        packed_output, hidden = self.encoder(packed_embeddings)

        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        u = torch.tanh(torch.matmul(output, self.weight_W))
        att = torch.matmul(u, self.weight_proj)
        att_score = F.softmax(att, dim=1)
        scored_x = output * att_score
        encoding = torch.sum(scored_x, dim=1)

        outputs = self.decoder1(encoding)
        outputs = self.decoder2(outputs)

        return outputs
    
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, drop_keep_prob, n_class):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(p=drop_keep_prob)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, dropout=0)
        self.dense = nn.Linear(hidden_dim, n_class)
        if n_class > 2: 
            self.out = nn.Softmax()
        elif n_class == 2:
            self.out = nn.Sigmoid()
        else:
            raise CustomError("n_class less than 2")
            

    def forward(self, inputs):

        lengths = torch.sum(inputs != 0, dim=1)

        embedded = self.embedding(inputs)
        embedded = self.dropout(embedded)
        
        packed_embedded = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True)
        
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        final_hidden = output[torch.arange(output.size(0)), lengths-1]
        
        dense_output = self.dense(final_hidden)
        output = self.out(dense_output)
        
        return output

