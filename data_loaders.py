# cd C:/Users/diges/Desktop/stanford-corenlp-4.5.8
# java -mx1g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
import numpy as np
import requests
from collections import Counter
from gensim.models import Word2Vec
from datasets import load_dataset



class HANTextDataset:
    def __init__(self, texts, min_frequency, embedding_dim):
        self.texts = texts
        self.min_frequency = min_frequency
        self.embedding_dim = embedding_dim

        # Tokenize all texts once
        self.processed_texts = self.tokenize_and_process_texts()

        # Build vocabulary
        self.vocab = self.build_vocabulary()
        #print vocabulary
        print(self.vocab)

        # Train Word2Vec
        self.word2vec_model = self.train_word2vec()

        # Build embedding matrix
        self.embedding_matrix = self.build_embedding_matrix()

    def tokenize_with_stanford(self, text):
        """Tokenizes the text using the Stanford CoreNLP server."""
        server_url = "http://localhost:9000"
        params = {"annotators": "tokenize,ssplit"}

        response = requests.post(server_url, params=params, data=text.encode("utf-8"))  # Send the request
        if response.status_code == 200:  # If the request is successful (status code 200)
            return response.json()
        else:
            raise Exception("Error in CoreNLP request")

    def tokenize_and_process_texts(self):
        """Tokenizes texts and stores them in a list."""
        tokenized_texts = []  # Store tokenized texts
        for text in self.texts:
            json_data = self.tokenize_with_stanford(text)
            tokenized_doc = []  # Store tokenized sentences
            for sentence in json_data["sentences"]:  # Iterate over sentences
                tokenized_sentence = [token["word"].lower() for token in sentence["tokens"]]
                tokenized_doc.append(tokenized_sentence)
            tokenized_texts.append(tokenized_doc)
        return tokenized_texts

    def build_vocabulary(self):
        """Creates a vocabulary dictionary mapping words to indices."""
        word_counter = Counter()
        for doc in self.processed_texts:  # Iterate over documents
            for sentence in doc:
                word_counter.update(sentence)  # Update word counts

        # Keep words appearing more than min_frequency
        vocab = {word: idx + 1 for idx, (word, count) in enumerate(word_counter.items()) if count > self.min_frequency}
        vocab['UNK'] = 0  # Assign index 0 to UNK token
        return vocab

    def train_word2vec(self):
        """Trains a Word2Vec model on the processed text."""
        sentences = [sentence for text in self.processed_texts for sentence in text]  # Flattens the list
        model = Word2Vec(sentences, vector_size=self.embedding_dim, window=5, min_count=5, sg=1, workers=4)
        return model

    def build_embedding_matrix(self):
        """Creates an embedding matrix using the trained Word2Vec model."""
        vocab_size = len(self.vocab)
        embedding_matrix = np.zeros((vocab_size, self.embedding_dim))

        for word, index in self.vocab.items():
            if word in self.word2vec_model.wv.key_to_index:
                embedding_matrix[index] = self.word2vec_model.wv[word]
            else:
                embedding_matrix[index] = np.random.uniform(-0.25, 0.25, self.embedding_dim)  # Random init for unknown words

        return embedding_matrix


# Example usage
texts = [
    "Stanford CoreNLP is a powerful tool for NLP.",
    "We train Word2Vec on the training and validation set.",
    "The embedding dimension is set to 200."
    "Stanford Stanford Stanford Stanford Stanford Stanford"
]

dataset = HANTextDataset(texts, min_frequency=5, embedding_dim=200)

print("Processed Texts:", dataset.processed_texts)
print("Vocabulary:", dataset.vocab)
print("Embedding Matrix Shape:", dataset.embedding_matrix.shape)

#load Stanford Sentiment Treebank (SST) dataset
sst_dataset = load_dataset("glue", "sst2")
train_texts = sst_dataset['train']['sentence']
print(train_texts[:5])

dataset = HANTextDataset(train_texts, min_frequency=5, embedding_dim=200)
print("Processed Texts:", dataset.processed_texts)
print("Vocabulary:", dataset.vocab)
print("Embedding Matrix Shape:", dataset.embedding_matrix.shape)
