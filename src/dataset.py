"""
dataset.py

PyTorch Dataset for Topic Classification


"""

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

from tokenizer import Tokenizer
from vocabulary import Vocabulary


class TopicDataset(Dataset):

    def __init__(
        self,
        texts,
        labels,
        vocabulary,
        max_length=128
    ):

        self.texts = texts.tolist()
        self.labels = labels.tolist()

        self.vocab = vocabulary

        self.max_length = max_length

        self.tokenizer = Tokenizer()

        self.label_encoder = LabelEncoder()

        self.encoded_labels = self.label_encoder.fit_transform(
            self.labels
        )

    def __len__(self):

        return len(self.texts)

    def pad_sequence(self, sequence):

        if len(sequence) > self.max_length:

            sequence = sequence[:self.max_length]

        else:

            sequence += [0] * (
                self.max_length - len(sequence)
            )

        return sequence

    def __getitem__(self, index):

        text = self.texts[index]

        label = self.encoded_labels[index]

        token_ids = self.vocab.numericalize(text)

        token_ids = self.pad_sequence(token_ids)

        return {

            "input_ids": torch.tensor(
                token_ids,
                dtype=torch.long
            ),

            "label": torch.tensor(
                label,
                dtype=torch.long
            )
        }


def create_dataloader(
    texts,
    labels,
    vocabulary,
    batch_size=64,
    shuffle=True,
    max_length=128
):

    dataset = TopicDataset(

        texts=texts,

        labels=labels,

        vocabulary=vocabulary,

        max_length=max_length
    )

    loader = DataLoader(

        dataset,

        batch_size=batch_size,

        shuffle=shuffle,

        num_workers=2,

        pin_memory=True
    )

    return loader, dataset