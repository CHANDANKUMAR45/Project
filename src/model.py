"""
model.py

Custom Deep Learning Model
Embedding + BiLSTM + Attention + Fully Connected Layer


"""

import torch
import torch.nn as nn


# ============================================================
# Attention Layer
# ============================================================

class Attention(nn.Module):

    def __init__(self, hidden_size):

        super().__init__()

        self.attention = nn.Linear(hidden_size * 2, 1)

    def forward(self, lstm_output):

        # lstm_output
        # (batch_size, sequence_length, hidden_size*2)

        scores = self.attention(lstm_output)

        weights = torch.softmax(scores, dim=1)

        context = torch.sum(weights * lstm_output, dim=1)

        return context


# ============================================================
# Topic Classification Model
# ============================================================

class TopicClassifier(nn.Module):

    def __init__(

        self,

        vocab_size,

        embedding_dim=128,

        hidden_dim=256,

        output_dim=10,

        num_layers=2,

        dropout=0.3

    ):

        super().__init__()

        # Embedding Layer

        self.embedding = nn.Embedding(

            num_embeddings=vocab_size,

            embedding_dim=embedding_dim,

            padding_idx=0

        )

        # BiLSTM

        self.lstm = nn.LSTM(

            input_size=embedding_dim,

            hidden_size=hidden_dim,

            num_layers=num_layers,

            batch_first=True,

            dropout=dropout,

            bidirectional=True

        )

        # Attention

        self.attention = Attention(hidden_dim)

        # Dropout

        self.dropout = nn.Dropout(dropout)

        # Fully Connected

        self.fc = nn.Linear(

            hidden_dim * 2,

            output_dim

        )

    def forward(self, x):

        # x
        # (batch_size, sequence_length)

        embedding = self.embedding(x)

        # (batch_size, sequence_length, embedding_dim)

        lstm_output, _ = self.lstm(embedding)

        # Attention

        context = self.attention(lstm_output)

        context = self.dropout(context)

        output = self.fc(context)

        return output


# ============================================================
# Count Parameters
# ============================================================

def count_parameters(model):

    return sum(

        p.numel()

        for p in model.parameters()

        if p.requires_grad

    )


# ============================================================
# Test Model
# ============================================================

if __name__ == "__main__":

    VOCAB_SIZE = 50000

    EMBEDDING_DIM = 128

    HIDDEN_DIM = 256

    OUTPUT_CLASSES = 20

    model = TopicClassifier(

        vocab_size=VOCAB_SIZE,

        embedding_dim=EMBEDDING_DIM,

        hidden_dim=HIDDEN_DIM,

        output_dim=OUTPUT_CLASSES

    )

    print(model)

    print()

    print("Trainable Parameters")

    print(count_parameters(model))

    print()

    sample = torch.randint(

        0,

        VOCAB_SIZE,

        (32, 128)

    )

    output = model(sample)

    print("Output Shape")

    print(output.shape)