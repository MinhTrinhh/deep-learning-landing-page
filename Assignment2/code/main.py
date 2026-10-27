import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from baseline_model import BiLSTMSentimentClassifier
from dataset import (
    LABEL2ID,
    LABEL_NAMES,
    get_dataloaders,
)

SEED = 42
TRAIN_PATH = "dataset/Corona_NLP_train.csv"
TEST_PATH = "dataset/Corona_NLP_test.csv"
PATIENCE = 3

BASE_BATCH_SIZE = 64
BASE_EVAL_BATCH_SIZE = 128
BASE_EPOCHS = 2
BASE_LEARNING_RATE = 1e-3
BASE_WEIGHT_DECAY = 0.0

PRETRAINED_MODEL_NAME = "distilbert/distilbert-base-uncased"
PRETRAINED_BATCH_SIZE = 16
PRETRAINED_EVAL_BATCH_SIZE = 32
PRETRAINED_EPOCHS = 2
PRETRAINED_LEARNING_RATE = 5e-3
PRETRAINED_WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def forward_batch(model, batch, mode, criterion, device):
    batch = {key: value.to(device) for key, value in batch.items()}
    labels = batch["labels"]
    
    if mode == "base":
        input_ids = batch["input_ids"]
        # Derive lengths from attention mask for the BiLSTM
        lengths = batch["attention_mask"].sum(dim=1).cpu() 
        logits = model(input_ids, lengths)
        loss = criterion(logits, labels)
    else:
        outputs = model(**batch)
        loss, logits = outputs.loss, outputs.logits
        
    return loss, logits, labels


def evaluate(model, loader, criterion, device, mode):
    model.eval()
    total_loss = 0.0
    all_labels, all_predictions = [], []

    with torch.no_grad():
        for batch in loader:
            loss, logits, labels = forward_batch(model, batch, mode, criterion, device)
            total_loss += loss.item() * labels.size(0)
            predictions = logits.argmax(dim=1)
            all_labels.extend(labels.detach().cpu().tolist())
            all_predictions.extend(predictions.detach().cpu().tolist())

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(all_labels, all_predictions),
        "macro_f1": f1_score(
            all_labels, all_predictions,
            labels=list(range(len(LABEL_NAMES))),
            average="macro", zero_division=0,
        ),
        "labels": all_labels,
        "predictions": all_predictions,
    }


def train_one_epoch(model, loader, criterion, optimizer, device, mode, scheduler=None):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    start = time.perf_counter()

    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        loss, logits, labels = forward_batch(model, batch, mode, criterion, device)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        n = labels.size(0)
        total_loss += loss.item() * n
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += n

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
        "seconds": time.perf_counter() - start,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Corona_NLP sentiment classifier.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--base", action="store_true", help="Train BiLSTM from scratch.")
    group.add_argument("--pretrained", action="store_true", help="Fine-tune DistilBERT.")
    args = parser.parse_args()

    mode = "base" if args.base else "pretrained"
    set_seed()
    device = get_device()
    print(f"Mode: {mode} | Device: {device}")

    scheduler = None
    tokenizer = None

    if mode == "base":
        epochs = BASE_EPOCHS
        checkpoint_path = Path("checkpoints/bilstm_best.pt")
        
        # Load dataloaders with HF Tokenizer
        train_loader, val_loader, test_loader, tokenizer, max_length = get_dataloaders(
            tokenizer_name=PRETRAINED_MODEL_NAME,
            train_path=TRAIN_PATH,
            test_path=TEST_PATH,
            batch_size=BASE_BATCH_SIZE,
            eval_batch_size=BASE_EVAL_BATCH_SIZE,
            random_state=SEED,
        )
        
        model_config = {
            "vocab_size": tokenizer.vocab_size, # Use HF tokenizer vocab size
            "embedding_dim": 128,
            "hidden_dim": 128,
            "num_layers": 2,
            "num_classes": len(LABEL_NAMES),
            "pad_idx": tokenizer.pad_token_id,  # Use HF tokenizer pad ID
            "dropout": 0.3,
        }
        model = BiLSTMSentimentClassifier(**model_config).to(device)
        learning_rate, weight_decay = BASE_LEARNING_RATE, BASE_WEIGHT_DECAY
    else:
        from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

        epochs = PRETRAINED_EPOCHS
        checkpoint_path = Path("checkpoints/distilbert_best")
        
        # Load dataloaders with HF Tokenizer
        train_loader, val_loader, test_loader, tokenizer, max_length = get_dataloaders(
            tokenizer_name=PRETRAINED_MODEL_NAME,
            train_path=TRAIN_PATH,
            test_path=TEST_PATH,
            batch_size=PRETRAINED_BATCH_SIZE,
            eval_batch_size=PRETRAINED_EVAL_BATCH_SIZE,
            random_state=SEED,
        )
        
        id2label = {idx: label for label, idx in LABEL2ID.items()}
        model = AutoModelForSequenceClassification.from_pretrained(
            PRETRAINED_MODEL_NAME,
            num_labels=len(LABEL_NAMES),
            id2label=id2label,
            label2id=LABEL2ID,
        ).to(device)
        learning_rate, weight_decay = PRETRAINED_LEARNING_RATE, PRETRAINED_WEIGHT_DECAY

        # Freeze the backbone, train the classifier head only
        for param in model.distilbert.parameters():
            param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if mode == "pretrained":
        total_steps = epochs * len(train_loader)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(WARMUP_RATIO * total_steps),
            num_training_steps=total_steps,
        )

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_f1 = -1.0
    best_epoch = 0
    no_improvement = 0
    total_training_seconds = 0.0

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, mode, scheduler
        )
        total_training_seconds += train_metrics["seconds"]
        val_metrics = evaluate(model, val_loader, criterion, device, mode)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train loss {train_metrics['loss']:.4f} | "
            f"train acc {train_metrics['accuracy']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} | "
            f"val acc {val_metrics['accuracy']:.4f} | "
            f"val macro-F1 {val_metrics['macro_f1']:.4f} | "
            f"{train_metrics['seconds']:.1f}s"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            no_improvement = 0
            if mode == "base":
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config,
                    "max_length": max_length,
                    "label_names": LABEL_NAMES,
                    "best_val_macro_f1": best_val_f1,
                    "epoch": best_epoch,
                }, checkpoint_path)
            else:
                model.save_pretrained(checkpoint_path)
                tokenizer.save_pretrained(checkpoint_path)
            print(f"  Saved best checkpoint: {checkpoint_path}")
        else:
            no_improvement += 1
            print(f"  No improvement: {no_improvement}/{PATIENCE}")

        if no_improvement >= PATIENCE:
            print("Early stopping: validation macro-F1 did not improve.")
            break

    if mode == "base":
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        from transformers import AutoModelForSequenceClassification
        del model
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path).to(device)

    test_metrics = evaluate(model, test_loader, criterion, device, mode)

    print("\n========== FINAL TEST RESULTS ==========")
    print(f"Model: {mode}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation macro-F1: {best_val_f1:.4f}")
    print(f"Test loss: {test_metrics['loss']:.4f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"Total training time: {total_training_seconds:.1f}s")

    print("\nPer-class classification report:")
    print(classification_report(
        test_metrics["labels"], test_metrics["predictions"],
        labels=list(range(len(LABEL_NAMES))),
        target_names=LABEL_NAMES, digits=4, zero_division=0,
    ))
    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(
        test_metrics["labels"], test_metrics["predictions"],
        labels=list(range(len(LABEL_NAMES))),
    ))


if __name__ == "__main__":
    main()