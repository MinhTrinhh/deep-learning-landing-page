"""N24News loading shared by every A3 script.

The duplicate-aware split from the EDA (outputs/eda/splits.csv) is the single
source of truth for which articles exist and which split they belong to; every
script keeps rows in that file's order so cached features line up by index.
"""
from __future__ import annotations

import html
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

ASSIGNMENT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SPLITS = ASSIGNMENT_DIR / "outputs" / "eda" / "splits.csv"
IMAGE_SIZE = 224


def clean_text(value) -> str:
    """Same minimal normalization as the EDA: HTML entities and whitespace only."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return re.sub(r"\s+", " ", html.unescape(str(value))).strip()


def find_metadata(data_root: Path) -> Path:
    direct = data_root / "news" / "nytimes.json"
    if direct.exists():
        return direct
    found = next(data_root.rglob("nytimes.json"), None)
    if found is None:
        raise FileNotFoundError(f"nytimes.json not found under {data_root}")
    return found


def find_image_dir(data_root: Path) -> Path:
    for candidate in (data_root / "imgs", data_root / "images", data_root / "news" / "imgs"):
        if candidate.is_dir():
            return candidate
    first = next(data_root.rglob("*.jpg"), None)
    if first is None:
        raise FileNotFoundError(f"no .jpg images found under {data_root}")
    return first.parent


def load_split_index(splits_csv: Path = DEFAULT_SPLITS) -> pd.DataFrame:
    """image_id, label, split, group_id plus an integer target y (classes sorted by name)."""
    index = pd.read_csv(splits_csv, dtype={"image_id": str})
    classes = sorted(index["label"].unique())
    index["y"] = index["label"].map({c: i for i, c in enumerate(classes)}).astype(int)
    return index


def class_names(index: pd.DataFrame) -> list[str]:
    return sorted(index["label"].unique())


def load_records(data_root: Path, splits_csv: Path = DEFAULT_SPLITS, limit: int | None = None) -> pd.DataFrame:
    """Split index joined with headline, abstract and image path, in splits.csv order."""
    data_root = Path(data_root)
    index = load_split_index(splits_csv)
    raw = pd.read_json(find_metadata(data_root))
    raw["image_id"] = raw["image_id"].astype(str)
    raw = raw.drop_duplicates("image_id").set_index("image_id")

    missing = set(index["image_id"]) - set(raw.index)
    if missing:
        raise ValueError(f"{len(missing)} ids from splits.csv are not in nytimes.json, e.g. {next(iter(missing))}")

    records = index.copy()
    records["headline"] = raw.loc[records["image_id"], "headline"].map(clean_text).to_numpy()
    records["abstract"] = raw.loc[records["image_id"], "abstract"].map(clean_text).to_numpy()
    image_dir = find_image_dir(data_root)
    records["img_path"] = [str(image_dir / f"{i}.jpg") for i in records["image_id"]]
    if limit:
        records = records.head(limit).copy()
    return records


def load_image(path: str, mean, std, size: int = IMAGE_SIZE) -> torch.Tensor:
    """Shorter side -> size, then a centre crop.

    On the 1050x550 thumbnails the crop drops the left and right edges, which
    removes the "NYT | Published YYYY" badge in the top-right corner (a year
    shortcut, see the proposal). Non-RGB images are converted to RGB.
    """
    with Image.open(path) as im:
        im.draft("RGB", (size, size))  # JPEG fast path: decode at a reduced scale that stays >= size
        im = im.convert("RGB")
        w, h = im.size
        scale = size / min(w, h)
        im = im.resize((max(size, round(w * scale)), max(size, round(h * scale))), Image.BICUBIC)
        w, h = im.size
        left, top = (w - size) // 2, (h - size) // 2
        im = im.crop((left, top, left + size, top + size))
        array = np.asarray(im, dtype=np.float32) / 255.0
    array = (array - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


class ImageDataset(Dataset):
    """Preprocessed images in record order, used for frozen-feature extraction."""

    def __init__(self, paths: list[str], mean, std, size: int = IMAGE_SIZE):
        self.paths, self.mean, self.std, self.size = list(paths), mean, std, size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        return load_image(self.paths[i], self.mean, self.std, self.size)
