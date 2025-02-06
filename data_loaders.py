from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from stanfordcorenlp import StanfordCoreNLP

stanford_corenlp_path = '/path/to/stanford-corenlp-full-2018-10-05'  # path to unzipped stanford corenlp folder
nlp = StanfordCoreNLP(stanford_corenlp_path)  # initialize stanford corenlp


class HANTextDataset(Dataset):

    def tokenize_with_corenlp(self, text):
        return nlp.word_tokenize(text)  # tokenize text using stanford corenlp

    def build_vocabulary(self, texts, min_frequency=5):  # build vocabulary from texts, keep only words that appear more than min_frequency times and substitute the rest with <UNK>
        word_counter = Counter()  # initialize counter to count word frequency

        for text in texts:
            tokens = self.tokenize_with_corenlp(text)
            word_counter.update(tokens)

        vocabbulary = {word: count for word, count in word_counter.items() if count > min_frequency}
        vocabbulary['<UNK>'] = 0

        return vocabbulary

    def process_text(self, text, vocabulary): # process text by tokenizing it and converting tokens to vocabulary indices
        tokens = self.tokenize_with_corenlp(text)
        return [vocabulary.get(token, vocabulary['<UNK>']) for token in tokens]

