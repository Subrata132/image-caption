import spacy
from collections import Counter


class Vocabulary:
    def __init__(self, threshold=5):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = threshold
        self.spacy_eng = spacy.load("en_core_web_lg")

    def __len__(self):
        return len(self.itos)

    def tokenize(self, text):
        tokenized_text = self.spacy_eng.tokenizer(text)
        tokenized_text_lower = []
        for token in tokenized_text:
            tokenized_text_lower.append(token.text.lower())
        return tokenized_text_lower

    def build_vocab(self, text_list):
        frequencies = Counter()
        idx = 4
        for text in text_list:
            tokenized_text = self.tokenize(text)
            for word in tokenized_text:
                frequencies[word] += 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def string_to_numerical(self, text):
        tokenized_text = self.tokenize(text)
        numeric_list = []
        for token in tokenized_text:
            if token in self.stoi:
                numeric_list.append(self.stoi[token])
            else:
                numeric_list.append(self.stoi['<UNK>'])
        return numeric_list
