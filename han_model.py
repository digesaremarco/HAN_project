import torch.nn as nn
import torch.nn.functional as F
import torch


class WordAttention(nn.Module):
    def __init__(self, opts):
        super(WordAttention, self).__init__()
        self.gru = nn.GRU(opts.gru_input_dim, opts.gru_hidden_dim, opts.gru_num_layers, batch_first=True,
                          bidirectional=opts.gru_bidirectional)
        self.attention = nn.Linear(opts.context_vectors_dim * 2, 1)  # bidirectional
        self.context_vector = nn.Parameter(torch.randn(opts.context_vectors_dim))  # random initialization

    def forward(self, word_embedding):
        """ Forward pass of the model, which is used to get the sentence vector. """
        h, _ = self.gru(word_embedding)  # gru encoding
        u = torch.tanh(self.attention(h))
        attention_weights = torch.softmax(torch.matmul(u, self.context_vector), dim=1)  # Attention weights
        sentence_vector = torch.sum(attention_weights * h, dim=1)  # Sentence vector obtained by weighted sum
        return sentence_vector


class SentenceAttention(nn.Module):
    def __init__(self, opts):
        super(SentenceAttention, self).__init__()
        self.gru = nn.GRU(opts.gru_input_dim, opts.gru_hidden_dim, opts.gru_num_layers, batch_first=True,
                          bidirectional=opts.gru_bidirectional)
        self.attention = nn.Linear(opts.context_vectors_dim * 2, 1)  # bidirectional
        self.context_vector = nn.Parameter(torch.randn(opts.context_vectors_dim))  # random initialization
        self.output_layer = nn.Linear(opts.context_vectors_dim * 2, opts.gru_output_dim) # output layer

    def forward(self, sentence_embedding):
        """ Forward pass of the model, which is used to get the document vector. """
        h, _ = self.gru(sentence_embedding)  # gru encoding
        u = torch.tanh(self.attention(h))
        attention_weights = torch.softmax(torch.matmul(u, self.context_vector), dim=1)  # Attention weights
        document_vector = torch.sum(attention_weights * h, dim=1)  # Document vector obtained by weighted sum
        return self.output_layer(document_vector)


class HAN(nn.Module):
    def __init__(self, opts, embedding_matrix):
        super(HAN, self).__init__()
        self.word_attention = WordAttention(opts)
        self.sentence_attention = SentenceAttention(opts)
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False) # embedding layer

    def forward(self, x):
        """ Forward pass of the model, which is used to get the output. """
        word_embedding = self.embedding(x) # word embedding
        word_context = self.word_attention(word_embedding)
        sentence_context = self.sentence_attention(word_context)
        return sentence_context