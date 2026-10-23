
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from model import BiLSTMSentimentClassifier
from dataset import get_dataloaders, PAD_IDX, LABEL_NAMES

# Configuration
SEED = 42
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 128
EPOCHS = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
PATIENCE = 3
CHECKPOINT_PATH = Path("checkpoints/bilstm_best.pt")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# Evaluation
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for input_ids, lengths, labels in loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            # Model returns raw logits of shape (batch_size, num_classes).
            logits = model(input_ids, lengths)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)

            predictions = logits.argmax(dim=1)
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    average_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_predictions)
    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        labels=list(range(len(LABEL_NAMES))),
        average="macro",
        zero_division=0,
    )

    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "labels": all_labels,
        "predictions": all_predictions,
    }


# One epoch training
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for input_ids, lengths, labels in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(input_ids, lengths)
        loss = criterion(logits, labels)

        loss.backward()

        # Helps prevent exploding gradients in recurrent networks.
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += batch_size

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


# Main
def main():
    set_seed(SEED)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, vocab, max_length = (
        get_dataloaders(
            train_path="dataset/Corona_NLP_train.csv",
            test_path="dataset/Corona_NLP_test.csv",
            batch_size=BATCH_SIZE,
            eval_batch_size=EVAL_BATCH_SIZE,
            random_state=SEED,
        )
    )

    model_config = {
        "vocab_size": len(vocab),
        "embedding_dim": 128,
        "hidden_dim": 128,
        "num_layers": 2,
        "num_classes": len(LABEL_NAMES),
        "pad_idx": PAD_IDX,
        "dropout": 0.3,
    }

    model = BiLSTMSentimentClassifier(**model_config).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    best_val_f1 = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_metrics = evaluate(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss: {train_metrics['loss']:.4f} | "
            f"train acc: {train_metrics['accuracy']:.4f} | "
            f"val loss: {val_metrics['loss']:.4f} | "
            f"val acc: {val_metrics['accuracy']:.4f} | "
            f"val macro-F1: {val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config,
                    "vocab": vocab,
                    "max_length": max_length,
                    "label_names": LABEL_NAMES,
                    "best_val_macro_f1": best_val_f1,
                    "epoch": epoch,
                },
                CHECKPOINT_PATH,
            )
            print(f"  Saved best checkpoint: {CHECKPOINT_PATH}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping: validation macro-F1 did not improve.")
            break

    # Test only once, after model selection is complete.
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(model, test_loader, criterion, device)

    print("\nBest checkpoint")
    print(f"Epoch: {checkpoint['epoch']}")
    print(f"Validation macro-F1: {checkpoint['best_val_macro_f1']:.4f}")

    print("\nHeld-out test results")
    print(f"Test loss:     {test_metrics['loss']:.4f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test macro-F1: {test_metrics['macro_f1']:.4f}")

    print("\nPer-class report")
    print(
        classification_report(
            test_metrics["labels"],
            test_metrics["predictions"],
            labels=list(range(len(LABEL_NAMES))),
            target_names=LABEL_NAMES,
            digits=4,
            zero_division=0,
        )
    )

    print("Confusion matrix (rows = true, columns = predicted)")
    print(
        confusion_matrix(
            test_metrics["labels"],
            test_metrics["predictions"],
            labels=list(range(len(LABEL_NAMES))),
        )
    )


if __name__ == "__main__":
   main()