import random
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn as nn
from plotter import Plotter
from baseline_model import BiLSTMSentimentClassifier
from dataset import (
    get_dataloaders
)

SEED = 42
PRETRAINED_MODEL_NAME = "distilbert/distilbert-base-uncased"
BASE_CHECKPOINT = Path("checkpoints/bilstm_best.pt")
BASE_NAME = "BiLSTM"
PRETRAINED_CHECKPOINT = Path("checkpoints/distilbert_best")
PRETRAINED_NAME = "DistilBERT"

TRAIN_PATH = "dataset/Corona_NLP_train.csv"
TEST_PATH = "dataset/Corona_NLP_test.csv"
PRETRAINED_BATCH_SIZE = 16
PRETRAINED_EVAL_BATCH_SIZE = 32

class InferenceWrapper(nn.Module):
    """Gives both models the same (input_ids, attention_mask) -> logits interface.

    Used for FLOPs profiling, inference timing and qualitative inference.
    """

    def __init__(self, model, mode):
        super().__init__()
        self.model = model
        self.mode = mode

    def forward(self, input_ids, attention_mask):
        if self.mode == "base":
            lengths = attention_mask.sum(dim=1).cpu()
            return self.model(input_ids, lengths)
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits
    
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
def decode_texts(tokenizer, input_ids):
    return tokenizer.batch_decode(input_ids, skip_special_tokens=True)

def load_checkpoint_models(device):
    wrappers = {}

    if BASE_CHECKPOINT.exists():
        checkpoint = torch.load(BASE_CHECKPOINT, map_location=device, weights_only=True)
        model = BiLSTMSentimentClassifier(**checkpoint["model_config"]).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        wrappers[BASE_NAME] = InferenceWrapper(model, "base")
    else:
        print(f"Skipping {BASE_NAME}: no checkpoint at {BASE_CHECKPOINT}")

    if PRETRAINED_CHECKPOINT.exists():
        from transformers import AutoModelForSequenceClassification

        model = AutoModelForSequenceClassification.from_pretrained(
            PRETRAINED_CHECKPOINT
        ).to(device)
        wrappers[PRETRAINED_NAME] = InferenceWrapper(model, "pretrained")
    else:
        print(f"Skipping {PRETRAINED_NAME}: no checkpoint at {PRETRAINED_CHECKPOINT}")

    for wrapper in wrappers.values():
        wrapper.eval()
    return wrappers


def collect_predictions(wrappers, test_loader, tokenizer, device):
    targets, texts = [], []
    predictions = {name: [] for name in wrappers}
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets.extend(batch["labels"].tolist())
            texts.extend(decode_texts(tokenizer, batch["input_ids"]))
            for name, wrapper in wrappers.items():
                predictions[name].extend(
                    wrapper(input_ids, attention_mask).argmax(dim=1).tolist()
                )
    return (
        texts,
        np.asarray(targets),
        {name: np.asarray(values) for name, values in predictions.items()},
    )


def select_examples(targets, predictions):
    correct = {name: values == targets for name, values in predictions.items()}
    all_correct = np.logical_and.reduce(list(correct.values()))
    any_correct = np.logical_or.reduce(list(correct.values()))

    masks = [("All models correct", all_correct)]
    if len(correct) > 1:
        for name, mask in correct.items():
            masks.append((f"{name} correct, another model wrong", mask & ~all_correct))
    masks.append(("All models incorrect", ~any_correct))

    selected = []
    used_classes = set()
    for category, mask in masks:
        candidates = np.flatnonzero(mask)
        chosen = []
        for index in candidates:  # first pass: prefer classes not shown yet
            label = int(targets[index])
            if label not in used_classes:
                chosen.append(int(index))
                used_classes.add(label)
            if len(chosen) == 2:
                break
        if len(chosen) < 2:  # second pass: fill up with anything left
            taken = {i for _, i in selected} | set(chosen)
            for index in candidates:
                if int(index) not in taken:
                    chosen.append(int(index))
                    taken.add(int(index))
                if len(chosen) == 2:
                    break
        selected.extend((category, index) for index in chosen)
    return selected


def run_qualitative(device):
    print("\n========== QUALITATIVE INFERENCE ==========")
    set_seed()
    wrappers = load_checkpoint_models(device)
    if not wrappers:
        print("No checkpoints found. Train with --base and/or --pretrained first.")
        return

    # same tokenizer and test split for both models
    _, _, test_loader, tokenizer, _ = get_dataloaders(
        tokenizer_name=PRETRAINED_MODEL_NAME,
        train_path=TRAIN_PATH,
        test_path=TEST_PATH,
        batch_size=PRETRAINED_BATCH_SIZE,
        eval_batch_size=PRETRAINED_EVAL_BATCH_SIZE,
        random_state=SEED,
    )
    texts, targets, predictions = collect_predictions(
        wrappers, test_loader, tokenizer, device
    )
    selected = select_examples(targets, predictions)
    if not selected:
        print("No examples matched any category.")
        return
    Plotter().plot_qualitative(texts, targets, predictions, selected)