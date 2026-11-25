"""Encode every N24News pair once with a frozen pretrained encoder.

Outputs (float16, one row per line of splits.csv, same order):
  roberta  -> roberta_cls.npy, roberta_mean.npy   (headline/abstract as a sentence pair)
  vit      -> vit.npy                             (ViT-B/16 ImageNet, CLS token)
plus index.csv (image_id, label, split, y) and <name>.json with the settings used.

Example:
  python code/extract_features.py --data /content/data --out /content/drive/MyDrive/DL_A3/features --encoder vit
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import ENCODERS, IMAGE_SIZE, MAX_LENGTH
from dataset import ImageDataset, load_records


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True, help="folder that contains the extracted N24News archive")
    p.add_argument("--out", type=Path, required=True, help="feature folder (put it on Drive)")
    p.add_argument("--encoder", choices=sorted(ENCODERS), required=True)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    return p.parse_args()


def write_index(records, out: Path) -> None:
    """index.csv pins the row order; refuse to mix features from different orders."""
    (out / "classes.json").write_text(json.dumps(sorted(records["label"].unique()), indent=2))
    index = records[["image_id", "label", "split", "y"]]
    path = out / "index.csv"
    if path.exists():
        existing = pd.read_csv(path, dtype={"image_id": str})
        if not existing["image_id"].equals(index["image_id"].reset_index(drop=True)):
            raise ValueError(f"{path} has a different row order; use a fresh --out folder or delete it")
    else:
        index.to_csv(path, index=False)


@torch.no_grad()
def encode_text(records, device, batch_size):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(ENCODERS["roberta"])
    model = AutoModel.from_pretrained(ENCODERS["roberta"], add_pooling_layer=False).to(device).eval()
    headlines, abstracts = records["headline"].tolist(), records["abstract"].tolist()
    cls_rows, mean_rows = [], []
    for start in tqdm(range(0, len(records), batch_size), desc="roberta"):
        batch = tokenizer(
            headlines[start:start + batch_size], abstracts[start:start + batch_size],
            truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt",
        ).to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            hidden = model(**batch).last_hidden_state.float()
        mask = batch["attention_mask"].unsqueeze(-1).float()
        cls_rows.append(hidden[:, 0].cpu())
        mean_rows.append(((hidden * mask).sum(1) / mask.sum(1)).cpu())
    return {
        "roberta_cls": torch.cat(cls_rows).numpy(),
        "roberta_mean": torch.cat(mean_rows).numpy(),
    }


@torch.no_grad()
def encode_images(records, encoder, device, batch_size, num_workers):
    from transformers import AutoImageProcessor

    name = ENCODERS[encoder]
    processor = AutoImageProcessor.from_pretrained(name)
    from transformers import ViTModel
    model = ViTModel.from_pretrained(name, add_pooling_layer=False).to(device).eval()

    dataset = ImageDataset(records["img_path"].tolist(), processor.image_mean, processor.image_std, IMAGE_SIZE)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=device.type == "cuda")
    rows = []
    for pixels in tqdm(loader, desc=encoder):
        pixels = pixels.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            output = model(pixel_values=pixels)
        pooled = output.last_hidden_state[:, 0]  # CLS token of the final layernormed sequence
        rows.append(pooled.float().cpu())
    return {encoder: torch.cat(rows).numpy()}


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    outputs = ["roberta_cls", "roberta_mean"] if args.encoder == "roberta" else [args.encoder]
    if all((args.out / f"{n}.npy").exists() for n in outputs):
        print(f"{outputs} already in {args.out}; skipping (delete the .npy files to redo)")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = load_records(args.data)
    write_index(records, args.out)
    print(f"{len(records)} records | device={device} | encoder={ENCODERS[args.encoder]}")

    start = time.time()
    if args.encoder == "roberta":
        features = encode_text(records, device, args.batch_size)
    else:
        features = encode_images(records, args.encoder, device, args.batch_size, args.num_workers)
    seconds = time.time() - start

    for name, array in features.items():
        assert array.shape[0] == len(records) and np.isfinite(array).all(), name
        np.save(args.out / f"{name}.npy", array.astype(np.float16))
        meta = {
            "feature": name, "checkpoint": ENCODERS[args.encoder], "shape": list(array.shape),
            "rows": len(records), "seconds": round(seconds, 1),
            "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
            "max_length" if args.encoder == "roberta" else "image_size":
                MAX_LENGTH if args.encoder == "roberta" else IMAGE_SIZE,
        }
        (args.out / f"{name}.json").write_text(json.dumps(meta, indent=2))
        print(f"saved {name}.npy {array.shape} in {seconds:.0f}s")


if __name__ == "__main__":
    main()
