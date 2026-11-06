import argparse
import random
import time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from metrics import MetricCalculator
from plotter import Plotter
from qualititative import run_qualitative, decode_texts, InferenceWrapper
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
BASE_EPOCHS = 15
BASE_LEARNING_RATE = 1e-3
BASE_WEIGHT_DECAY = 0.01

PRETRAINED_MODEL_NAME = "distilbert/distilbert-base-uncased"
PRETRAINED_BATCH_SIZE = 16
PRETRAINED_EVAL_BATCH_SIZE = 32
PRETRAINED_EPOCHS = 5
PRETRAINED_LEARNING_RATE = 2e-5
PRETRAINED_WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

# Output / checkpoint locations and display names
BASE_CHECKPOINT = Path("checkpoints/bilstm_best.pt")
PRETRAINED_CHECKPOINT = Path("checkpoints/distilbert_best")
BASE_NAME = "BiLSTM"
PRETRAINED_NAME = "DistilBERT"

CLASS_NAMES = list(LABEL_NAMES)
MAX_PROJECTION_SAMPLES = 3000

# Helpers
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

# EDA helpers
def collect_eda_data(loader, tokenizer):
    """One pass over the training loader: class counts + 15 random example tweets."""
    counts = np.zeros(len(CLASS_NAMES), dtype=int)
    texts, labels = [], []
    for i, batch in enumerate(loader):
        batch_labels = batch["labels"]
        counts += np.bincount(batch_labels.numpy(), minlength=len(CLASS_NAMES))
        if i < 2:  # first two (shuffled) batches are plenty to sample from
            texts.extend(decode_texts(tokenizer, batch["input_ids"]))
            labels.extend(batch_labels.tolist())

    rng = np.random.RandomState(SEED)
    pick = rng.choice(len(texts), size=min(15, len(texts)), replace=False)
    return counts, [texts[i] for i in pick], [labels[i] for i in pick]


def extract_features(model, loader, criterion, device, mode):
    """Sentence representations = the input of the model's final nn.Linear layer."""
    head = [m for m in model.modules() if isinstance(m, nn.Linear)][-1]
    captured = []
    handle = head.register_forward_hook(
        lambda module, inputs, output: captured.append(inputs[0].detach().cpu())
    )
    features, labels = [], []
    model.eval()
    try:
        with torch.no_grad():
            for batch in loader:
                captured.clear()
                _, _, batch_labels = forward_batch(model, batch, mode, criterion, device)
                features.append(captured[-1].numpy())
                labels.extend(batch_labels.cpu().tolist())
    finally:
        handle.remove()
    return np.concatenate(features, axis=0), np.array(labels)


def compute_projections(features, labels):
    """PCA / t-SNE / UMAP 2D projections on (at most) MAX_PROJECTION_SAMPLES points."""
    if len(labels) > MAX_PROJECTION_SAMPLES:
        rng = np.random.RandomState(SEED)
        idx = rng.choice(len(labels), MAX_PROJECTION_SAMPLES, replace=False)
        features, labels = features[idx], labels[idx]

    pca_2d = PCA(n_components=2, random_state=SEED).fit_transform(features)
    tsne_2d = TSNE(
        n_components=2, init="pca", perplexity=30, random_state=SEED
    ).fit_transform(features)
    try:
        import umap

        umap_2d = umap.UMAP(n_components=2, random_state=SEED).fit_transform(features)
    except ImportError:
        print("umap-learn not installed (pip install umap-learn); skipping UMAP panel.")
        umap_2d = None

    return {
        "labels": labels.tolist(),
        "pca_2d": pca_2d,
        "tsne_2d": tsne_2d,
        "umap_2d": umap_2d,
    }


def class_similarity(features, labels):
    """Cosine similarity between per-class mean feature vectors."""
    centroids = np.stack([
        features[labels == c].mean(axis=0) if np.any(labels == c)
        else np.zeros(features.shape[1])
        for c in range(len(CLASS_NAMES))
    ])
    return cosine_similarity(centroids)

def run_training(mode, device):
    calculator = MetricCalculator()
    plotter = Plotter()
    model_name = BASE_NAME if mode == "base" else PRETRAINED_NAME

    scheduler = None
    tokenizer = None

    if mode == "base":
        epochs = BASE_EPOCHS
        checkpoint_path = BASE_CHECKPOINT

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
            "vocab_size": tokenizer.vocab_size,  # Use HF tokenizer vocab size
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
        checkpoint_path = PRETRAINED_CHECKPOINT

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
            seq_classif_dropout=0.3,
        ).to(device)
        learning_rate, weight_decay = PRETRAINED_LEARNING_RATE, PRETRAINED_WEIGHT_DECAY

    # EDA plots (class distribution + random example tweets)
    class_counts, example_texts, example_labels = collect_eda_data(train_loader, tokenizer)
    plotter.plot_class_distribution(class_counts)
    plotter.plot_examples(example_texts, example_labels)
    set_seed()  # iterating the shuffled loader consumed RNG; restore a clean state

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

    epoch_train_loss, epoch_train_acc = [], []
    epoch_val_loss, epoch_val_acc = [], []

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, mode, scheduler
        )
        total_training_seconds += train_metrics["seconds"]
        val_metrics = evaluate(model, val_loader, criterion, device, mode)

        epoch_train_loss.append(train_metrics["loss"])
        epoch_train_acc.append(train_metrics["accuracy"])
        epoch_val_loss.append(val_metrics["loss"])
        epoch_val_acc.append(val_metrics["accuracy"])

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

    # restore best checkpoint
    if mode == "base":
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        from transformers import AutoModelForSequenceClassification
        del model
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path).to(device)

    test_metrics = evaluate(model, test_loader, criterion, device, mode)

    # resource metrics (FLOPs, params, size, inference time)
    sample_batch = next(iter(test_loader))
    sample_inputs = (
        sample_batch["input_ids"].to(device),
        sample_batch["attention_mask"].to(device),
    )
    resource_metrics = calculator.calculate_resource_metrics(
        InferenceWrapper(model, mode), sample_inputs, total_training_seconds
    )

    print("\n========== FINAL TEST RESULTS ==========")
    print(f"Model: {mode}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation macro-F1: {best_val_f1:.4f}")

    # accuracy, macro precision/recall/F1, per-class report, confusion matrix
    results = calculator.report(
        test_metrics["labels"],
        test_metrics["predictions"],
        model_name,
        test_loss=test_metrics["loss"],
        resource_metrics=resource_metrics,
    )

    # plots that need the trained model / test results
    plotter.plot_confusion_matrix(results["confusion_matrix"], model_name)
    plotter.plot_learning_curves(
        epoch_train_loss, epoch_train_acc, epoch_val_loss, epoch_val_acc, model_name
    )

    features, feature_labels = extract_features(model, test_loader, criterion, device, mode)
    plotter.plot_dimensionality_reduction(
        compute_projections(features, feature_labels), model_name
    )
    plotter.plot_similarity_matrix(class_similarity(features, feature_labels), model_name)


def main():
    parser = argparse.ArgumentParser(description="Train Corona_NLP sentiment classifier.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--base", action="store_true", help="Train BiLSTM from scratch.")
    group.add_argument("--pretrained", action="store_true", help="Fine-tune DistilBERT.")
    parser.add_argument(
        "--infer",
        action="store_true",
        help="Qualitative inference with the saved best checkpoints. "
             "Use alone, or after training (e.g. --base --infer).",
    )
    args = parser.parse_args()

    if not (args.base or args.pretrained or args.infer):
        parser.error("choose --base, --pretrained and/or --infer.")

    set_seed()
    device = get_device()

    if args.base or args.pretrained:
        mode = "base" if args.base else "pretrained"
        print(f"Mode: {mode} | Device: {device}")
        run_training(mode, device)

    if args.infer:
        if not (args.base or args.pretrained):
            print(f"Mode: infer | Device: {device}")
        run_qualitative(device)


if __name__ == "__main__":
    main()