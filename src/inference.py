"""
=========================================================
inference.py

Inference Script for Topic Classification


=========================================================
"""

import os
import json
import pickle
import torch
import torch.nn.functional as F

from tokenizer import Tokenizer
from model import TopicClassifier


# ==========================================================
# Configuration
# ==========================================================

MODEL_DIR = "../final_models"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------------------------------------

with open(os.path.join(MODEL_DIR, "config.json")) as f:
    config = json.load(f)

with open(os.path.join(MODEL_DIR, "vocabulary.pkl"), "rb") as f:
    vocab = pickle.load(f)

with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
    label_encoder = pickle.load(f)

# ----------------------------------------------------------

model = TopicClassifier(

    vocab_size=config["vocab_size"],

    embedding_dim=config["embedding_dim"],

    hidden_dim=config["hidden_dim"],

    output_dim=config["num_classes"],

    num_layers=config["num_layers"],

    dropout=config["dropout"]

)

model.load_state_dict(

    torch.load(

        os.path.join(
            MODEL_DIR,
            "topic_classifier.pth"
        ),

        map_location=DEVICE

    )

)

model.to(DEVICE)

model.eval()

tokenizer = Tokenizer()

def predict(text):

    tokens = vocab.numericalize(text)

    max_length = config["max_length"]

    if len(tokens) > max_length:

        tokens = tokens[:max_length]

    else:

        tokens += [0] * (
            max_length - len(tokens)
        )

    tensor = torch.tensor(
        [tokens],
        dtype=torch.long
    ).to(DEVICE)

    with torch.no_grad():

        output = model(tensor)

        probabilities = F.softmax(
            output,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    label = label_encoder.inverse_transform(
        prediction.cpu().numpy()
    )[0]

    return label, confidence.item()

print("=" * 60)
print("Topic Classification Inference")
print("=" * 60)

while True:

    text = input("\nEnter Text (type exit to quit): ")

    if text.lower() == "exit":

        break

    label, confidence = predict(text)

    print("\nPrediction")

    print("---------------------------")

    print("Topic      :", label)

    print(
        "Confidence :",
        f"{confidence*100:.2f}%"
    )