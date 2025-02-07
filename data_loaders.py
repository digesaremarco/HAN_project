#cd C:/Users/diges/Desktop/stanford-corenlp-4.5.8
#java -mx1g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000


import requests
from collections import Counter

class HANTextDataset:
    def __init__(self, texts, min_frequency=5):
        self.texts = texts
        self.min_frequency = min_frequency
        self.vocab, self.word_counter = self.build_vocabulary()
        self.processed_texts = self.process_texts()

    def tokenize_with_stanford(self, text):
        """Tokenizes the text using the Stanford CoreNLP server."""
        server_url = "http://localhost:9000"  # URL of the server
        params = {
            "annotators": "tokenize,ssplit",  # Enables tokenization and sentence splitting
            "outputFormat": "json"  # Specifies JSON output format
        }

        # Sends a POST request to the server with the text
        response = requests.post(server_url, params=params, data=text.encode("utf-8"))

        # Checks if the request was successful
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception("Error in CoreNLP request")

    def build_vocabulary(self):
        """Builds the vocabulary, keeping only words with frequency > min_frequency."""
        word_counter = Counter()

        # Tokenization and word counting
        for text in self.texts:
            json_data = self.tokenize_with_stanford(text)

            for sentence in json_data["sentences"]:
                for token in sentence["tokens"]:
                    word = token["word"].lower()  # Converts the word to lowercase for uniformity
                    word_counter[word] += 1

        # Filters words appearing more than min_frequency times
        vocab = {word: count for word, count in word_counter.items() if count > self.min_frequency}

        # Adds the UNK token for rare words
        vocab['UNK'] = 0

        return vocab, word_counter

    def process_texts(self):
        """Replaces rare words with 'UNK' and tokenizes the text."""
        processed_texts = []

        for text in self.texts:
            json_data = self.tokenize_with_stanford(text)
            processed_text = []

            for sentence in json_data["sentences"]:
                processed_sentence = []
                for token in sentence["tokens"]:
                    word_text = token["word"].lower()  # Converts the word to lowercase for uniformity
                    if word_text in self.vocab:
                        processed_sentence.append(word_text)
                    else:
                        processed_sentence.append('UNK')  # Rare word, replaces with UNK
                processed_text.append(processed_sentence)

            processed_texts.append(processed_text)

        return processed_texts


# Example usage

# A list of documents (texts)
texts = [
    "Stanford CoreNLP is a powerful tool for NLP.",
    "It is used for various tasks like tokenization, part-of-speech tagging, and named entity recognition.",
    "The Stanford CoreNLP toolkit includes multiple components for different NLP tasks.",
    "Stanford Stanford Stanford Stanford Stanford"
]

# 1. Build the dataset
dataset = HANTextDataset(texts, min_frequency=5)

# Print the vocabulary
print("Vocabulary:", dataset.vocab)
# Print the processed texts (rare words replaced by UNK)
print("Processed Texts:", dataset.processed_texts)