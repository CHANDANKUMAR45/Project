"""
=========================================================
utils.py

Utility Functions


=========================================================
"""

import os
import json
import random
import numpy as np
import torch
import matplotlib.pyplot as plt


# ==========================================================
# Random Seed
# ==========================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ==========================================================
# Save Model
# ==========================================================

def save_model(model, path):

    torch.save(
        model.state_dict(),
        path
    )

    print(f"Model Saved -> {path}")


# ==========================================================
# Load Model
# ==========================================================

def load_model(model, path, device):

    model.load_state_dict(
        torch.load(
            path,
            map_location=device
        )
    )

    return model


# ==========================================================
# Save JSON
# ==========================================================

def save_json(data, path):

    with open(path, "w") as f:

        json.dump(
            data,
            f,
            indent=4
        )


# ==========================================================
# Save Checkpoint
# ==========================================================

def save_checkpoint(

    model,

    optimizer,

    epoch,

    loss,

    path

):

    checkpoint = {

        "epoch": epoch,

        "model_state_dict": model.state_dict(),

        "optimizer_state_dict": optimizer.state_dict(),

        "loss": loss

    }

    torch.save(
        checkpoint,
        path
    )

    print("Checkpoint Saved")


# ==========================================================
# Load Checkpoint
# ==========================================================

def load_checkpoint(

    path,

    model,

    optimizer,

    device

):

    checkpoint = torch.load(

        path,

        map_location=device

    )

    model.load_state_dict(

        checkpoint["model_state_dict"]

    )

    optimizer.load_state_dict(

        checkpoint["optimizer_state_dict"]

    )

    epoch = checkpoint["epoch"]

    loss = checkpoint["loss"]

    return model, optimizer, epoch, loss


# ==========================================================
# Plot Loss
# ==========================================================

def plot_loss(history, save_path):

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

    plt.savefig(save_path)

    plt.close()


# ==========================================================
# Plot Accuracy
# ==========================================================

def plot_accuracy(history, save_path):

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

    plt.savefig(save_path)

    plt.close()


# ==========================================================
# Count Parameters
# ==========================================================

def count_parameters(model):

    return sum(

        p.numel()

        for p in model.parameters()

        if p.requires_grad

    )


# ==========================================================
# Print Model Summary
# ==========================================================

def print_model_summary(model):

    print(model)

    print()

    print("Trainable Parameters")

    print(count_parameters(model))