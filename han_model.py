import torch.nn as nn
import torch
import torch.nn.functional as F


class WordAttention(nn.Module):
    def __init__(self, opts):
        super(WordAttention, self).__init__()
        self.gru = nn.GRU(opts.gru_input_dim, opts.gru_hidden_dim, opts.gru_num_layers, batch_first=True,
                          bidirectional=opts.gru_bidirectional)
        self.attention = nn.Linear(opts.gru_output_dim, opts.context_vectors_dim)  # bidirectional
        self.context_vector = nn.Parameter(torch.randn(opts.context_vectors_dim))  # random initialization
        self.frozen_attention = opts.frozen_attention

    def forward(self, word_embedding):
        """ Forward pass of the model, which is used to get the sentence vector. """
        h, _ = self.gru(word_embedding)  # gru encoding

        if self.frozen_attention:
            B, T, _ = h.size()
            attention_weights = torch.ones(B, T, device=h.device) / T  # uniform attention weights
        else:
            u = torch.tanh(self.attention(h))
            attention_weights = torch.softmax(torch.matmul(u, self.context_vector), dim=1)  # Attention weights

        sentence_vector = torch.sum(attention_weights.unsqueeze(-1) * h,
                                    dim=1)  # Sentence vector obtained by weighted sum

        return sentence_vector


class SentenceAttention(nn.Module):
    def __init__(self, opts):
        super(SentenceAttention, self).__init__()
        self.gru = nn.GRU(opts.gru_output_dim, opts.gru_hidden_dim, opts.gru_num_layers, batch_first=True,
                          bidirectional=opts.gru_bidirectional)
        self.attention = nn.Linear(opts.gru_output_dim, opts.context_vectors_dim)  # bidirectional
        self.context_vector = nn.Parameter(torch.randn(opts.context_vectors_dim))  # random initialization
        self.output_layer = nn.Linear(opts.context_vectors_dim, opts.num_classes)
        self.frozen_attention = opts.frozen_attention

    def forward(self, sentence_embedding):
        """ Forward pass of the model, which is used to get the document vector. """
        h, _ = self.gru(sentence_embedding)  # gru encoding

        if self.frozen_attention:
            B, T, _ = h.size()
            attention_weights = torch.ones(B, T, device=h.device) / T  # uniform attention weights
        else:
            u = torch.tanh(self.attention(h))
            attention_weights = torch.softmax(torch.matmul(u, self.context_vector), dim=1)  # Attention weights
        document_vector = torch.sum(attention_weights.unsqueeze(-1) * h,
                                    dim=1)  # Document vector obtained by weighted sum

        logits = self.output_layer(document_vector)
        return F.log_softmax(logits, dim=1)  # Apply softmax to the output layer


class HAN(nn.Module):
    def __init__(self, opts, embedding_matrix):
        super(HAN, self).__init__()
        self.word_attention = WordAttention(opts)
        self.sentence_attention = SentenceAttention(opts)
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float32), freeze=False)

    def forward(self, x):
        """ Forward pass of the model, which is used to get the output. """
        word_embedding = self.embedding(x)  # Word embedding

        batch_size, num_sentences, num_words, embedding_dim = word_embedding.shape
        word_embedding = word_embedding.view(batch_size * num_sentences, num_words, embedding_dim)

        word_context = self.word_attention(word_embedding)  # Word-level encoding
        word_context = word_context.view(batch_size, num_sentences,
                                         -1)  # Restore the original shape after word-level encoding

        sentence_context = self.sentence_attention(word_context)  # Sentence-level encoding
        return sentence_context
