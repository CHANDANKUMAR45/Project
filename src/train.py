"""
==========================================================
train.py

Training Script for Topic Classification


==========================================================
"""

# ==========================================================
# Imports
# ==========================================================

import os
import random
import warnings

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split

from tokenizer import Tokenizer
from vocabulary import Vocabulary
from dataset import create_dataloader
from model import TopicClassifier


warnings.filterwarnings("ignore")


# ==========================================================
# Configuration
# ==========================================================

class Config:

    # Dataset
    DATASET_PATH = "../dataset/dataset_10M.parquet"

    # Model
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    DROPOUT = 0.30

    # Training
    MAX_LENGTH = 128
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    EPOCHS = 5

    # Vocabulary
    MIN_FREQUENCY = 2

    # Random Seed
    SEED = 42

    # Save Directory
    SAVE_DIR = "../final_models"

    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


config = Config()

print("=" * 60)
print("Device :", config.DEVICE)
print("=" * 60)


# ==========================================================
# Reproducibility
# ==========================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


set_seed(config.SEED)


# ==========================================================
# Create Save Directory
# ==========================================================

os.makedirs(
    config.SAVE_DIR,
    exist_ok=True
)


# ==========================================================
# Load Dataset
# ==========================================================

print("\nLoading Dataset...\n")

df = pd.read_parquet(
    config.DATASET_PATH
)

print("Dataset Loaded Successfully")

print()

print("Shape")

print(df.shape)

print()

print(df.head())

print()

print("Columns")

print(df.columns.tolist())

print()

print("Missing Values")

print(df.isnull().sum())

print()

print("Duplicate Rows")

print(df.duplicated().sum())


# ==========================================================
# Basic Cleaning
# ==========================================================

df = df.dropna()

df = df.drop_duplicates()

df = df.reset_index(drop=True)

print()

print("Shape After Cleaning")

print(df.shape)


# ==========================================================
# Train / Validation Split
# ==========================================================

train_df, valid_df = train_test_split(

    df,

    test_size=0.20,

    random_state=config.SEED,

    stratify=df["TOPIC"]

)

print()

print("Training Samples")

print(train_df.shape)

print()

print("Validation Samples")

print(valid_df.shape)
# ==========================================================
# Build Vocabulary
# ==========================================================

print("\n" + "=" * 60)
print("Building Vocabulary...")
print("=" * 60)

vocab = Vocabulary(
    min_frequency=config.MIN_FREQUENCY
)

vocab.build(train_df["DATA"])

VOCAB_SIZE = len(vocab.word2idx)

print(f"\nVocabulary Size : {VOCAB_SIZE}")


# ==========================================================
# Save Vocabulary
# ==========================================================

VOCAB_PATH = os.path.join(
    config.SAVE_DIR,
    "vocabulary.pkl"
)

vocab.save(VOCAB_PATH)


# ==========================================================
# Create Train DataLoader
# ==========================================================

print("\nCreating Train DataLoader...")

train_loader, train_dataset = create_dataloader(

    texts=train_df["DATA"],

    labels=train_df["TOPIC"],

    vocabulary=vocab,

    batch_size=config.BATCH_SIZE,

    shuffle=True,

    max_length=config.MAX_LENGTH

)

print("Train Loader Created")


# ==========================================================
# Create Validation DataLoader
# ==========================================================

print("\nCreating Validation DataLoader...")

valid_loader, valid_dataset = create_dataloader(

    texts=valid_df["DATA"],

    labels=valid_df["TOPIC"],

    vocabulary=vocab,

    batch_size=config.BATCH_SIZE,

    shuffle=False,

    max_length=config.MAX_LENGTH

)

print("Validation Loader Created")


# ==========================================================
# Number of Classes
# ==========================================================

NUM_CLASSES = len(train_dataset.label_encoder.classes_)

print()

print("Number of Classes :", NUM_CLASSES)


# ==========================================================
# Check First Batch
# ==========================================================

print("\nChecking DataLoader...\n")

batch = next(iter(train_loader))

print("Input Shape :", batch["input_ids"].shape)

print("Label Shape :", batch["label"].shape)

print()

print("Sample Labels")

print(batch["label"][:10])

# ==========================================================
# Initialize Model
# ==========================================================

print("\n" + "=" * 60)
print("Initializing Model...")
print("=" * 60)

model = TopicClassifier(
    vocab_size=VOCAB_SIZE,
    embedding_dim=config.EMBEDDING_DIM,
    hidden_dim=config.HIDDEN_DIM,
    output_dim=NUM_CLASSES,
    num_layers=config.NUM_LAYERS,
    dropout=config.DROPOUT
)

model = model.to(config.DEVICE)

print(model)


# ==========================================================
# Count Parameters
# ==========================================================

def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

print()

print("Trainable Parameters :")

print(f"{count_parameters(model):,}")

# ==========================================================
# Loss Function
# ==========================================================

criterion = nn.CrossEntropyLoss()

print("\nLoss Function")

print(criterion)

# ==========================================================
# Optimizer
# ==========================================================

optimizer = torch.optim.Adam(

    model.parameters(),

    lr=config.LEARNING_RATE,

    weight_decay=1e-5

)

print()

print("Optimizer")

print(optimizer)

# ==========================================================
# Learning Rate Scheduler
# ==========================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="max",

    factor=0.5,

    patience=2

)

print()

print("Scheduler Ready")

# ==========================================================
# Mixed Precision Training
# ==========================================================

from torch.cuda.amp import GradScaler

scaler = GradScaler()

print("AMP Enabled")

# ==========================================================
# Best Model
# ==========================================================

best_accuracy = 0.0

best_epoch = 0

# ==========================================================
# GPU Information
# ==========================================================

print("\nDevice :", config.DEVICE)

if torch.cuda.is_available():

    print("GPU :", torch.cuda.get_device_name(0))

    print(
        "Memory :",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / 1024**3,
            2
        ),
        "GB"
    )

    # ==========================================================
# Training Function
# ==========================================================

from tqdm.auto import tqdm
from torch.cuda.amp import autocast


def train_one_epoch(model,
                    dataloader,
                    optimizer,
                    criterion,
                    scaler,
                    device):

    model.train()

    running_loss = 0.0
    running_correct = 0
    total = 0

    progress_bar = tqdm(
        dataloader,
        desc="Training",
        leave=False
    )

    for batch in progress_bar:

        inputs = batch["input_ids"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        # Mixed Precision Forward Pass
        with autocast():

            outputs = model(inputs)

            loss = criterion(outputs, labels)

        # Backward Pass
        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        # Predictions
        predictions = torch.argmax(outputs, dim=1)

        running_correct += (predictions == labels).sum().item()

        total += labels.size(0)

        running_loss += loss.item()

        accuracy = running_correct / total

        progress_bar.set_postfix(

            loss=f"{loss.item():.4f}",

            accuracy=f"{accuracy*100:.2f}%"

        )

    epoch_loss = running_loss / len(dataloader)

    epoch_accuracy = running_correct / total

    return epoch_loss, epoch_accuracy

# ==========================================================
# Start Training
# ==========================================================

print("\n" + "="*60)
print("Starting Training")
print("="*60)

for epoch in range(config.EPOCHS):

    print(f"\nEpoch {epoch+1}/{config.EPOCHS}")

    train_loss, train_accuracy = train_one_epoch(

        model=model,

        dataloader=train_loader,

        optimizer=optimizer,

        criterion=criterion,

        scaler=scaler,

        device=config.DEVICE

    )

    history["train_loss"].append(train_loss)

    history["train_accuracy"].append(train_accuracy)

    print(f"Train Loss     : {train_loss:.4f}")

    print(f"Train Accuracy : {train_accuracy*100:.2f}%")

    # ==========================================================
# Save Checkpoint
# ==========================================================

checkpoint_path = os.path.join(

    config.SAVE_DIR,

    "checkpoint_last.pth"

)

torch.save({

    "epoch": epoch,

    "model_state_dict": model.state_dict(),

    "optimizer_state_dict": optimizer.state_dict(),

    "loss": train_loss

}, checkpoint_path)

print("Checkpoint Saved")

# ==========================================================
# Validation Function
# ==========================================================

from sklearn.metrics import accuracy_score


def validate_one_epoch(
    model,
    dataloader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0

    predictions = []

    labels_list = []

    with torch.no_grad():

        progress_bar = tqdm(
            dataloader,
            desc="Validation",
            leave=False
        )

        for batch in progress_bar:

            inputs = batch["input_ids"].to(device)

            labels = batch["label"].to(device)

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            predictions.extend(
                preds.cpu().numpy()
            )

            labels_list.extend(
                labels.cpu().numpy()
            )

    epoch_loss = running_loss / len(dataloader)

    epoch_accuracy = accuracy_score(
        labels_list,
        predictions
    )

    return epoch_loss, epoch_accuracy

print("\n" + "=" * 60)
print("Training Started")
print("=" * 60)

for epoch in range(config.EPOCHS):

    print(f"\nEpoch {epoch+1}/{config.EPOCHS}")

    # ----------------------------
    # Train
    # ----------------------------

    train_loss, train_accuracy = train_one_epoch(

        model,

        train_loader,

        optimizer,

        criterion,

        scaler,

        config.DEVICE

    )

    # ----------------------------
    # Validation
    # ----------------------------

    valid_loss, valid_accuracy = validate_one_epoch(

        model,

        valid_loader,

        criterion,

        config.DEVICE

    )

    history["train_loss"].append(train_loss)

    history["train_accuracy"].append(train_accuracy)

    history["valid_loss"].append(valid_loss)

    history["valid_accuracy"].append(valid_accuracy)

    print(f"Train Loss      : {train_loss:.4f}")
    print(f"Train Accuracy  : {train_accuracy*100:.2f}%")

    print(f"Valid Loss      : {valid_loss:.4f}")
    print(f"Valid Accuracy  : {valid_accuracy*100:.2f}%")

    # ==========================================================
# Save Best Model
# ==========================================================

if valid_accuracy > best_accuracy:

    best_accuracy = valid_accuracy

    best_epoch = epoch + 1

    torch.save(

        model.state_dict(),

        os.path.join(
            config.SAVE_DIR,
            "topic_classifier.pth"
        )

    )

    print("Best Model Saved")

    scheduler.step(valid_accuracy)

    print("\n" + "=" * 60)
print("Training Completed")
print("=" * 60)

print(f"Best Validation Accuracy : {best_accuracy*100:.2f}%")

print(f"Best Epoch : {best_epoch}")

# ==========================================================
# Early Stopping
# ==========================================================

class EarlyStopping:

    def __init__(self, patience=3):

        self.patience = patience
        self.best_score = None
        self.counter = 0
        self.stop = False

    def __call__(self, score):

        if self.best_score is None:

            self.best_score = score

        elif score <= self.best_score:

            self.counter += 1

            print(
                f"EarlyStopping Counter : {self.counter}/{self.patience}"
            )

            if self.counter >= self.patience:

                self.stop = True

        else:

            self.best_score = score
            self.counter = 0


early_stopping = EarlyStopping(patience=3)

scheduler.step(valid_accuracy)

early_stopping(valid_accuracy)

if early_stopping.stop:

    print("\nEarly Stopping Triggered")



# ==========================================================
# Save Training History
# ==========================================================

import json

history_path = os.path.join(
    config.SAVE_DIR,
    "training_history.json"
)

with open(history_path, "w") as f:

    json.dump(history, f, indent=4)

print("Training History Saved")

# ==========================================================
# Training Curves
# ==========================================================

import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))

plt.plot(
    history["train_loss"],
    label="Train Loss"
)

plt.plot(
    history["valid_loss"],
    label="Validation Loss"
)

plt.legend()

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Loss Curve")

plt.savefig(
    os.path.join(
        config.SAVE_DIR,
        "loss_curve.png"
    )
)

plt.show()

plt.figure(figsize=(10,5))

plt.plot(
    history["train_accuracy"],
    label="Train Accuracy"
)

plt.plot(
    history["valid_accuracy"],
    label="Validation Accuracy"
)

plt.legend()

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title("Accuracy Curve")

plt.savefig(
    os.path.join(
        config.SAVE_DIR,
        "accuracy_curve.png"
    )
)

plt.show()

metrics = {

    "best_epoch": best_epoch,

    "best_validation_accuracy": best_accuracy,

    "final_train_accuracy":
        history["train_accuracy"][-1],

    "final_validation_accuracy":
        history["valid_accuracy"][-1]

}

with open(

    os.path.join(
        config.SAVE_DIR,
        "metrics.json"
    ),

    "w"

) as f:

    json.dump(metrics, f, indent=4)

print("Metrics Saved")

# ==========================================================
# FINAL MODEL SAVE
# ==========================================================

print("\n" + "=" * 70)
print("Saving Final Model...")
print("=" * 70)

BEST_MODEL_PATH = os.path.join(
    config.SAVE_DIR,
    "topic_classifier.pth"
)

torch.save(
    model.state_dict(),
    BEST_MODEL_PATH
)

print("Model Saved Successfully")
print(BEST_MODEL_PATH)

# ==========================================================
# Save Configuration
# ==========================================================

config_dict = {

    "embedding_dim": config.EMBEDDING_DIM,

    "hidden_dim": config.HIDDEN_DIM,

    "num_layers": config.NUM_LAYERS,

    "dropout": config.DROPOUT,

    "batch_size": config.BATCH_SIZE,

    "learning_rate": config.LEARNING_RATE,

    "epochs": config.EPOCHS,

    "max_length": config.MAX_LENGTH,

    "vocab_size": VOCAB_SIZE,

    "num_classes": NUM_CLASSES

}

import json

with open(

    os.path.join(
        config.SAVE_DIR,
        "config.json"
    ),

    "w"

) as f:

    json.dump(config_dict, f, indent=4)

print("Configuration Saved")

# ==========================================================
# Save Label Encoder
# ==========================================================

import pickle

with open(

    os.path.join(
        config.SAVE_DIR,
        "label_encoder.pkl"
    ),

    "wb"

) as f:

    pickle.dump(
        train_dataset.label_encoder,
        f
    )

print("Label Encoder Saved")

# ==========================================================
# Save Class Names
# ==========================================================

class_names = list(
    train_dataset.label_encoder.classes_
)

with open(

    os.path.join(
        config.SAVE_DIR,
        "class_names.json"
    ),

    "w"

) as f:

    json.dump(
        class_names,
        f,
        indent=4
    )

print("Class Names Saved")


# ==========================================================
# Training Summary
# ==========================================================

summary = f"""

==========================================================
Training Summary
==========================================================

Vocabulary Size :

{VOCAB_SIZE}

Number of Classes :

{NUM_CLASSES}

Embedding Dimension :

{config.EMBEDDING_DIM}

Hidden Dimension :

{config.HIDDEN_DIM}

Epochs :

{config.EPOCHS}

Batch Size :

{config.BATCH_SIZE}

Learning Rate :

{config.LEARNING_RATE}

Best Validation Accuracy :

{best_accuracy:.4f}

Best Epoch :

{best_epoch}

==========================================================

"""

print(summary)

with open(

    os.path.join(
        config.SAVE_DIR,
        "training_summary.txt"
    ),

    "w"

) as f:

    f.write(summary)

    # ==========================================================
# Verify Saved Files
# ==========================================================

print("\nGenerated Files\n")

for file in sorted(os.listdir(config.SAVE_DIR)):

    print(file)