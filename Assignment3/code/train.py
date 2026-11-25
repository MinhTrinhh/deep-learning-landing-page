"""Train one baseline head on cached features and evaluate it on val/test.

  python code/train.py --features FEAT_DIR --model text   --text-feat roberta_cls --seed 0 --out RUNS_DIR
  python code/train.py --features FEAT_DIR --model image  --image-feat vit        --seed 0 --out RUNS_DIR
  python code/train.py --features FEAT_DIR --model concat --text-feat roberta_cls --image-feat vit --seed 0 --out RUNS_DIR

The checkpoint is selected on validation macro-F1 (early stopping); the test
split is only scored once, with that checkpoint. Everything lands in
RUNS_DIR/<run name>/seed<k>/: model.pt, metrics.json, history.csv, test_probs.npy.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn

from models import MODALITIES, build_model, count_parameters


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", choices=sorted(MODALITIES), required=True)
    p.add_argument("--text-feat", default="roberta_cls", help="roberta_cls or roberta_mean")
    p.add_argument("--image-feat", default="vit",
                   help="vit (ImageNet, the unimodal baseline) or clip (language-aligned, side experiment only)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=8, help="epochs without val macro-F1 gain before stopping")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--hidden", type=int, default=512, help="0 = linear probe")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--debug", action="store_true", help="fixed 10%% stratified train sample, never for reported numbers")
    p.add_argument("--name", default=None, help="override the run name")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_name(args) -> str:
    if args.name:
        return args.name
    parts = {"text": args.text_feat, "image": args.image_feat}
    return "_".join([args.model] + [parts[m] for m in MODALITIES[args.model]])


def load_features(args, device):
    index = pd.read_csv(args.features / "index.csv", dtype={"image_id": str})
    files = {"text": args.text_feat, "image": args.image_feat}
    feats = {}
    for modality in MODALITIES[args.model]:
        array = np.load(args.features / f"{files[modality]}.npy").astype(np.float32)
        if len(array) != len(index):
            raise ValueError(f"{files[modality]}.npy has {len(array)} rows, index.csv has {len(index)}")
        feats[modality] = torch.from_numpy(array).to(device)
    labels = torch.from_numpy(index["y"].to_numpy().copy()).long().to(device)
    return index, feats, labels


def debug_subset(index: pd.DataFrame, train_idx: np.ndarray) -> np.ndarray:
    # fixed seed so every model sees the same debug sample
    sample = index.iloc[train_idx].groupby("label", group_keys=False).sample(frac=0.1, random_state=42)
    return np.sort(sample.index.to_numpy())


@torch.no_grad()
def predict(model, feats, idx, batch_size=4096) -> torch.Tensor:
    model.eval()
    chunks = []
    for start in range(0, len(idx), batch_size):
        rows = idx[start:start + batch_size]
        chunks.append(model(*[feats[m][rows] for m in feats]).float())
    return torch.cat(chunks)


def scores(y_true, y_pred, num_classes) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=range(num_classes), zero_division=0)),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    index, feats, labels = load_features(args, device)
    classes_file = args.features / "classes.json"
    classes = json.loads(classes_file.read_text()) if classes_file.exists() else sorted(index["label"].unique())
    num_classes = len(classes)

    split = index["split"].to_numpy()
    train_idx = np.flatnonzero(split == "train")
    if args.debug:
        train_idx = debug_subset(index, train_idx)
    val_idx, test_idx = np.flatnonzero(split == "val"), np.flatnonzero(split == "test")
    train_t, val_t, test_t = (torch.from_numpy(i).to(device) for i in (train_idx, val_idx, test_idx))
    y_val, y_test = labels[val_t].cpu().numpy(), labels[test_t].cpu().numpy()

    dims = {m: f.shape[1] for m, f in feats.items()}
    model = build_model(args.model, dims, num_classes, args.hidden, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    name = run_name(args)
    run_dir = args.out / name / f"seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"{name} seed={args.seed} | train/val/test={len(train_idx)}/{len(val_idx)}/{len(test_idx)} "
          f"| dims={dims} | params={count_parameters(model):,} | device={device}")

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    best_f1, best_epoch, best_state, history = -1.0, 0, None, []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = train_t[torch.randperm(len(train_t), generator=generator).to(device)]
        total, seen = 0.0, 0
        for b in range(0, len(order), args.batch_size):
            rows = order[b:b + args.batch_size]
            loss = criterion(model(*[feats[m][rows] for m in feats]), labels[rows])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(rows)
            seen += len(rows)
        scheduler.step()

        val_logits = predict(model, feats, val_t)
        val_loss = criterion(val_logits, labels[val_t]).item()
        val = scores(y_val, val_logits.argmax(1).cpu().numpy(), num_classes)
        history.append({"epoch": epoch, "train_loss": total / seen, "val_loss": val_loss,
                        "val_accuracy": val["accuracy"], "val_macro_f1": val["macro_f1"]})
        improved = val["macro_f1"] > best_f1
        if improved:
            best_f1, best_epoch = val["macro_f1"], epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch:3d} | train_loss {total / seen:.4f} | val_loss {val_loss:.4f} "
              f"| val_acc {val['accuracy']:.4f} | val_macroF1 {val['macro_f1']:.4f}{' *' if improved else ''}")
        if epoch - best_epoch >= args.patience:
            print(f"early stop: no val macro-F1 gain for {args.patience} epochs")
            break
    train_seconds = time.time() - start

    model.load_state_dict(best_state)
    val_pred = predict(model, feats, val_t).argmax(1).cpu().numpy()
    test_logits = predict(model, feats, test_t)
    test_pred = test_logits.argmax(1).cpu().numpy()
    test_probs = torch.softmax(test_logits, 1).cpu().numpy()

    metrics = {
        "run": name, "model": args.model, "seed": args.seed, "debug": args.debug,
        "text_feat": args.text_feat if "text" in MODALITIES[args.model] else None,
        "image_feat": args.image_feat if "image" in MODALITIES[args.model] else None,
        "best_epoch": best_epoch, "epochs_run": len(history),
        "train_seconds": round(train_seconds, 2), "parameters": count_parameters(model),
        "n_train": int(len(train_idx)), "n_val": int(len(val_idx)), "n_test": int(len(test_idx)),
        "val": scores(y_val, val_pred, num_classes),
        "test": scores(y_test, test_pred, num_classes),
        "test_per_class_f1": dict(zip(classes, f1_score(
            y_test, test_pred, average=None, labels=range(num_classes), zero_division=0).round(4).tolist())),
        "test_confusion_matrix": confusion_matrix(y_test, test_pred, labels=range(num_classes)).tolist(),
        "classes": classes,
        "hyperparameters": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    torch.save({"state_dict": best_state, "model": args.model, "input_dims": model.input_dims,
                "num_classes": num_classes, "hidden": args.hidden, "dropout": args.dropout,
                "classes": classes, "text_feat": metrics["text_feat"], "image_feat": metrics["image_feat"],
                "seed": args.seed, "best_epoch": best_epoch, "val_macro_f1": metrics["val"]["macro_f1"]},
               run_dir / "model.pt")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    np.save(run_dir / "test_probs.npy", test_probs.astype(np.float16))
    np.save(run_dir / "test_image_ids.npy", index["image_id"].to_numpy()[test_idx].astype(str))

    print(f"best epoch {best_epoch} | val macro-F1 {metrics['val']['macro_f1']:.4f} "
          f"| TEST acc {metrics['test']['accuracy']:.4f} macro-F1 {metrics['test']['macro_f1']:.4f} -> {run_dir}")


if __name__ == "__main__":
    main()
