import numpy as np
import requests
from collections import Counter

from gensim.models import Word2Vec
import torch
from torch.utils.data import DataLoader

from gensim.models.callbacks import CallbackAny2Vec


class EpochPrinter(CallbackAny2Vec):
    """Callback to print each epoch."""

    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        """Called at the end of each epoch."""
        self.epoch += 1
        print(f"Epoch {self.epoch}/{model.epochs}")


class HANTextDataset:
    def __init__(self, texts, labels, opts, vocab=None, embedding_matrix=None):
        self.texts = texts
        self.labels = labels
        self.min_frequency = opts.word_embedding_min_frequency
        self.embedding_dim = opts.word_embedding_dim

        # Tokenize all texts once
        print("Tokenizing and processing texts...")
        self.processed_texts = self.tokenize_and_process_texts()

        if vocab is None:
            # Build vocabulary
            self.vocab = self.build_vocabulary()
            # Train Word2Vec
            self.word2vec_model = self.train_word2vec()
            # Build embedding matrix
            self.embedding_matrix = self.build_embedding_matrix()
        else:
            # Use precomputed vocab and embedding matrix for validation/test
            self.vocab = vocab
            self.embedding_matrix = embedding_matrix

    def tokenize_with_stanford(self, text):
        """Tokenizes the text using the Stanford CoreNLP server."""
        server_url = "http://localhost:9030"
        params = {"annotators": "tokenize, ssplit"}

        response = requests.post(server_url, params=params, data=text.encode("utf-8"))  # Send the request
        if response.status_code == 200:  # If the request is successful (status code 200)
            return response.json()
        else:
            raise Exception("Error in CoreNLP request")

    def tokenize_and_process_texts(self):
        """Tokenizes texts and stores them in a list, printing progress every 100 texts."""
        tokenized_texts = []  # Store tokenized texts
        total_texts = len(self.texts)

        for idx, text in enumerate(self.texts):
            # Print progress every 1000 texts
            if (idx + 1) % 1000 == 0:
                progress = (idx + 1) / total_texts * 100
                print(f"Progress: {idx + 1}/{total_texts} ({progress:.2f}%)")

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
        """Trains a Word2Vec model on the processed text, printing progress during training."""
        sentences = [sentence for text in self.processed_texts for sentence in text]  # Flattens the list
        model = Word2Vec(sentences, vector_size=self.embedding_dim, window=5, min_count=5, sg=1, workers=4, epochs=10)

        # Create a callback to print the epoch
        epoch_printer = EpochPrinter()

        # Train the model with the callback
        model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs, report_delay=1,
                    compute_loss=True, callbacks=[epoch_printer])

        return model

    def build_embedding_matrix(self):
        """Creates an embedding matrix using the trained Word2Vec model."""
        vocab_size = len(self.vocab)
        embedding_matrix = np.zeros((vocab_size, self.embedding_dim))

        for word, index in self.vocab.items():
            if index < vocab_size:  # Verifica che l'indice non superi i limiti
                if word in self.word2vec_model.wv.key_to_index:
                    embedding_matrix[index] = self.word2vec_model.wv[word]
                else:
                    embedding_matrix[index] = np.random.uniform(-0.25, 0.25,
                                                                self.embedding_dim)  # Random init for unknown words

        return embedding_matrix

    # documents of similar length (in terms of the number of sentences in the documents) are organized to be a batch. This is done to ensure that the model can be trained efficiently using mini-batch training.
    def custom_collate(batch):
        """Collate function to organize documents into batches of similar length and pad the documents."""
        docs, labels = zip(*batch)  # Unzip the batch

        # Find the maximum number of sentences and words in the batch
        max_num_sentences = max(len(doc) for doc in docs)
        max_num_words = max(
            max(len(sentence) for sentence in doc) for doc in docs)

        # Pad the documents to have the same number of sentences and words
        padded_docs = []
        for doc in docs:
            padded_sentences = []
            for sentence in doc:
                # Sentence padding
                padded_sentence = torch.tensor(sentence, dtype=torch.long)
                if padded_sentence.size(0) < max_num_words:
                    pad_length = max_num_words - padded_sentence.size(0)
                    padded_sentence = torch.cat([padded_sentence, torch.zeros(pad_length, dtype=torch.long)])
                padded_sentences.append(padded_sentence)

            # If the document has fewer sentences than the maximum, pad the sentences
            if len(padded_sentences) < max_num_sentences:
                pad_sentences = torch.zeros(
                    (max_num_sentences - len(padded_sentences), max_num_words), dtype=torch.long
                )
                padded_sentences.extend(pad_sentences)

            padded_docs.append(torch.stack(padded_sentences))

        # Stack the padded documents to create a batch tensor
        batch_tensor = torch.stack(padded_docs)  # (batch_size, max_sentences, max_words)

        return (batch_tensor, torch.tensor(labels, dtype=torch.long))

    def __len__(self):
        return len(self.processed_texts)

    def __getitem__(self, idx):
        """Return the indexed document and its label."""
        indexed_doc = [
            [min(self.vocab.get(word, self.vocab['UNK']), len(self.vocab) - 1) for word in sentence]
            for sentence in self.processed_texts[idx]
        ]
        label = self.labels[idx]
        return indexed_doc, label


class MakeDataLoader:

    def __init__(self, opts, train_set, validation_set, test_set):
        self.train_loader = DataLoader(train_set, batch_size=opts.batch_size, shuffle=True,
                                       collate_fn=HANTextDataset.custom_collate)
        self.validation_loader = DataLoader(validation_set, batch_size=opts.batch_size, shuffle=True,
                                            collate_fn=HANTextDataset.custom_collate)
        self.test_loader = DataLoader(test_set, batch_size=opts.batch_size, shuffle=True,
                                     collate_fn=HANTextDataset.custom_collate)