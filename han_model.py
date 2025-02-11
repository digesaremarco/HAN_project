import torch.nn as nn
import torch.nn.functional as F
import torch


class WordAttention(nn.Module):
    def __init__(self, opts):
        super(WordAttention, self).__init__()
        self.gru = nn.GRU(opts.gru_input_dim, opts.gru_hidden_dim, opts.gru_num_layers, batch_first=True, bidirectional=opts.gru_bidirectional)
        self.attention = nn.Linear(opts.context_vectors_dim * 2, 1) # bidirectional
        self.context_vector = nn.Parameter(torch.randn(opts.context_vectors_dim)) # random initialization

    def forward(self, word_embedding):
        """ Forward pass of the model, which is used to get the sentence vector. """
        h, _ = self.gru(word_embedding) # gru encoding
        u = torch.tanh(self.attention(h))
        attention_weights = torch.softmax(torch.matmul(u, self.context_vector), dim=1)  # Attention weights
        sentence_vector = torch.sum(attention_weights * h, dim=1) # Sentence vector obtained by weighted sum
        return sentence_vector

