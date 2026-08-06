"""
vocabulary.py

Build Vocabulary From Scratch
No pretrained tokenizer is used.


"""

import pickle
from collections import Counter

from tokenizer import Tokenizer


class Vocabulary:

    def __init__(self, min_frequency=2):

        self.min_frequency = min_frequency

        self.word2idx = {
            "<PAD>": 0,
            "<UNK>": 1
        }

        self.idx2word = {
            0: "<PAD>",
            1: "<UNK>"
        }

    def build(self, texts):

        tokenizer = Tokenizer()

        counter = Counter()

        print("Building Vocabulary...")

        for text in texts:

            tokens = tokenizer.tokenize(text)

            counter.update(tokens)

        index = 2

        for word, freq in counter.items():

            if freq >= self.min_frequency:

                self.word2idx[word] = index
                self.idx2word[index] = word

                index += 1

        print("Vocabulary Created Successfully")
        print(f"Vocabulary Size : {len(self.word2idx)}")

    def numericalize(self, text):

        tokenizer = Tokenizer()

        tokens = tokenizer.tokenize(text)

        return [
            self.word2idx.get(word, self.word2idx["<UNK>"])
            for word in tokens
        ]

    def save(self, path):

        with open(path, "wb") as f:

            pickle.dump(self, f)

        print(f"Vocabulary Saved -> {path}")

    @staticmethod
    def load(path):

        with open(path, "rb") as f:

            vocab = pickle.load(f)

        print(f"Vocabulary Loaded <- {path}")

        return vocab


if __name__ == "__main__":

    sample_texts = [

        "Machine Learning is Amazing",

        "Deep Learning uses Neural Networks",

        "Python is the best programming language",

        "Machine Learning with Python"
    ]

    vocab = Vocabulary(min_frequency=1)

    vocab.build(sample_texts)

    print()

    print(vocab.word2idx)

    print()

    sentence = "Machine Learning with AI"

    print(sentence)

    print(vocab.numericalize(sentence))

    vocab.save("vocabulary.pkl")

    loaded_vocab = Vocabulary.load("vocabulary.pkl")

    print()

    print(loaded_vocab.word2idx)