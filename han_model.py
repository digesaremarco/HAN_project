import torch.nn as nn
import torch


class WordAttention(nn.Module):
    def __init__(self, opts):
        super(WordAttention, self).__init__()
        self.gru = nn.GRU(opts.gru_input_dim, opts.gru_hidden_dim, opts.gru_num_layers, batch_first=True,
                          bidirectional=opts.gru_bidirectional)
        self.attention = nn.Linear(opts.gru_output_dim, opts.context_vectors_dim ) #* 2, 1)  # bidirectional
        self.context_vector = nn.Parameter(torch.randn(opts.context_vectors_dim))  # random initialization

    def forward(self, word_embedding):
        """ Forward pass of the model, which is used to get the sentence vector. """
        h, _ = self.gru(word_embedding)  # gru encoding

        '''print("=== DEBUG: WordAttention ===")
        print("word_embedding shape:", word_embedding.shape)  # Deve essere (batch_size, seq_len, 200)
        print("GRU output shape:", h.shape)  # Deve essere (batch_size, seq_len, 100)
        print("============================")'''

        u = torch.tanh(self.attention(h))
        #print("u shape:", u.shape)
        attention_weights = torch.softmax(torch.matmul(u, self.context_vector), dim=1)  # Attention weights
        #print("attention_weights shape:", attention_weights.shape)
        sentence_vector = torch.sum(attention_weights.unsqueeze(-1) * h, dim=1)  # Sentence vector obtained by weighted sum
        #print("sentence_vector shape:", sentence_vector.shape)
        return sentence_vector


class SentenceAttention(nn.Module):
    def __init__(self, opts):
        super(SentenceAttention, self).__init__()
        self.gru = nn.GRU(opts.gru_output_dim, opts.gru_hidden_dim, opts.gru_num_layers, batch_first=True,
                          bidirectional=opts.gru_bidirectional)
        self.attention = nn.Linear(opts.gru_output_dim, opts.context_vectors_dim)  #* 2, 1)  # bidirectional
        self.context_vector = nn.Parameter(torch.randn(opts.context_vectors_dim))  # random initialization
        self.output_layer = nn.Linear(opts.context_vectors_dim, opts.num_classes)

    def forward(self, sentence_embedding):
        """ Forward pass of the model, which is used to get the document vector. """

        '''print("=== DEBUG: SentenceAttention ===")
        print("sentence_embedding shape:", sentence_embedding.shape)  # Deve essere (batch_size, num_sentences, 100)
        print("==============================")'''

        h, _ = self.gru(sentence_embedding)  # gru encoding
        u = torch.tanh(self.attention(h))
        #print("document u shape:", u.shape)
        attention_weights = torch.softmax(torch.matmul(u, self.context_vector.unsqueeze(0).transpose(0, 1)), dim=1) # Attention weights
        #print("document attention_weights shape:", attention_weights.shape)
        document_vector = torch.sum(attention_weights.expand(-1, -1, 100) * u, dim=1)  # Document vector obtained by weighted sum
        #print("document_vector shape:", document_vector.shape)
        return self.output_layer(document_vector)


class HAN(nn.Module):
    def __init__(self, opts, embedding_matrix):
        super(HAN, self).__init__()
        self.word_attention = WordAttention(opts)
        self.sentence_attention = SentenceAttention(opts)
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float32), freeze=False)

    def forward(self, x):
        """ Forward pass of the model, which is used to get the output. """
        '''word_embedding = self.embedding(x.long()) # word embedding
        word_embedding = word_embedding.squeeze(1)
        #word_embedding = word_embedding.squeeze(2)
        word_context = self.word_attention(word_embedding)

        print("=== DEBUG: HAN ===")
        print("word_context shape:", word_context.shape)  # Deve essere (batch_size, num_sentences, 100)
        print("===================")

        sentence_context = self.sentence_attention(word_context)
        return sentence_context'''
        word_embedding = self.embedding(x)  # Word embedding

        batch_size, num_sentences, num_words, embedding_dim = word_embedding.shape
        word_embedding = word_embedding.view(batch_size * num_sentences, num_words, embedding_dim)

        word_context = self.word_attention(word_embedding)  # Word-level encoding
        word_context = word_context.view(batch_size, num_sentences, -1)  # Restore the original shape after word-level encoding

        sentence_context = self.sentence_attention(word_context)  # Sentence-level encoding
        return sentence_context