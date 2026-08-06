"""
=========================================================
evaluate.py

Evaluate Topic Classification Model


=========================================================
"""

# ==========================================================
# Imports
# ==========================================================

import os
import json
import pickle
import warnings

import torch
import pandas as pd

from sklearn.model_selection import train_test_split

from vocabulary import Vocabulary
from dataset import create_dataloader
from model import TopicClassifier

warnings.filterwarnings("ignore")


# ==========================================================
# Configuration
# ==========================================================

class Config:

    DATASET_PATH = "../dataset/dataset_10M.parquet"

    MODEL_DIR = "../final_models"

    MODEL_PATH = os.path.join(
        MODEL_DIR,
        "topic_classifier.pth"
    )

    CONFIG_PATH = os.path.join(
        MODEL_DIR,
        "config.json"
    )

    VOCAB_PATH = os.path.join(
        MODEL_DIR,
        "vocabulary.pkl"
    )

    LABEL_ENCODER_PATH = os.path.join(
        MODEL_DIR,
        "label_encoder.pkl"
    )

    BATCH_SIZE = 64

    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


config = Config()


print("=" * 60)
print("Evaluation Started")
print("=" * 60)

print("Device :", config.DEVICE)


# ==========================================================
# Load Configuration
# ==========================================================

print("\nLoading Configuration...")

with open(config.CONFIG_PATH, "r") as f:

    model_config = json.load(f)

print("Configuration Loaded")


# ==========================================================
# Load Vocabulary
# ==========================================================

print("\nLoading Vocabulary...")

vocab = Vocabulary.load(config.VOCAB_PATH)

print("Vocabulary Loaded")

print("Vocabulary Size :", len(vocab.word2idx))


# ==========================================================
# Load Label Encoder
# ==========================================================

print("\nLoading Label Encoder...")

with open(config.LABEL_ENCODER_PATH, "rb") as f:

    label_encoder = pickle.load(f)

print("Label Encoder Loaded")

print("Classes :", len(label_encoder.classes_))


# ==========================================================
# Load Dataset
# ==========================================================

print("\nLoading Dataset...")

df = pd.read_parquet(
    config.DATASET_PATH
)

print("Dataset Loaded")

print(df.shape)


# ==========================================================
# Basic Cleaning
# ==========================================================

df = df.dropna()

df = df.drop_duplicates()

df = df.reset_index(drop=True)

print("Shape After Cleaning")

print(df.shape)


# ==========================================================
# Create Test Dataset
# ==========================================================

_, test_df = train_test_split(

    df,

    test_size=0.20,

    random_state=42,

    stratify=df["TOPIC"]

)

print()

print("Test Samples")

print(test_df.shape)


# ==========================================================
# DataLoader
# ==========================================================

test_loader, test_dataset = create_dataloader(

    texts=test_df["DATA"],

    labels=test_df["TOPIC"],

    vocabulary=vocab,

    batch_size=config.BATCH_SIZE,

    shuffle=False

)

print("\nTest DataLoader Ready")

print("Number of Batches :", len(test_loader))


# ==========================================================
# Build Model
# ==========================================================

print("\nBuilding Model...")

model = TopicClassifier(

    vocab_size=model_config["vocab_size"],

    embedding_dim=model_config["embedding_dim"],

    hidden_dim=model_config["hidden_dim"],

    output_dim=model_config["num_classes"],

    num_layers=model_config["num_layers"],

    dropout=model_config["dropout"]

)

model.load_state_dict(

    torch.load(

        config.MODEL_PATH,

        map_location=config.DEVICE

    )

)

model = model.to(config.DEVICE)

model.eval()

print("Model Loaded Successfully")

# ==========================================================
# Evaluation
# ==========================================================

from tqdm.auto import tqdm

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns

predictions = []
true_labels = []

print("\nStarting Evaluation...\n")

with torch.no_grad():

    progress_bar = tqdm(
        test_loader,
        desc="Evaluating"
    )

    for batch in progress_bar:

        inputs = batch["input_ids"].to(config.DEVICE)

        labels = batch["label"].to(config.DEVICE)

        outputs = model(inputs)

        preds = torch.argmax(outputs, dim=1)

        predictions.extend(
            preds.cpu().numpy()
        )

        true_labels.extend(
            labels.cpu().numpy()
        )

        # ==========================================================
# Metrics
# ==========================================================

accuracy = accuracy_score(
    true_labels,
    predictions
)

precision = precision_score(
    true_labels,
    predictions,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    true_labels,
    predictions,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    true_labels,
    predictions,
    average="weighted",
    zero_division=0
)

print("\nEvaluation Results")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

# ==========================================================
# Classification Report
# ==========================================================

report = classification_report(
    true_labels,
    predictions,
    target_names=label_encoder.classes_,
    zero_division=0
)

print("\nClassification Report\n")

print(report)

report_path = os.path.join(
    config.MODEL_DIR,
    "classification_report.txt"
)

with open(report_path, "w") as f:

    f.write(report)

print("\nClassification Report Saved")

# ==========================================================
# Confusion Matrix
# ==========================================================

cm = confusion_matrix(
    true_labels,
    predictions
)

plt.figure(figsize=(12,10))

sns.heatmap(
    cm,
    cmap="Blues"
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    os.path.join(
        config.MODEL_DIR,
        "confusion_matrix.png"
    )
)

plt.show()

# ==========================================================
# Save Metrics
# ==========================================================

metrics = {

    "accuracy": float(accuracy),

    "precision": float(precision),

    "recall": float(recall),

    "f1_score": float(f1)

}

metrics_path = os.path.join(
    config.MODEL_DIR,
    "evaluation_metrics.json"
)

with open(metrics_path, "w") as f:

    json.dump(
        metrics,
        f,
        indent=4
    )

print("\nMetrics Saved")

print("\n" + "="*60)

print("Evaluation Completed")

print("="*60)

print(f"Accuracy : {accuracy:.4f}")

print(f"Precision: {precision:.4f}")

print(f"Recall   : {recall:.4f}")

print(f"F1 Score : {f1:.4f}")

print("="*60)