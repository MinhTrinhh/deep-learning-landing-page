"""Shared settings for every A3 script (command-line flags override the training defaults)."""
from pathlib import Path

ASSIGNMENT_DIR = Path(__file__).resolve().parents[1]
SPLITS_CSV = ASSIGNMENT_DIR / "outputs" / "eda" / "splits.csv"  # duplicate-aware split from the EDA (seed 42)

# Frozen pretrained encoders
ENCODERS = {
    "roberta": "roberta-base",                # text: headline + abstract as a sentence pair
    "vit": "google/vit-base-patch16-224",     # image: ViT-B/16, ImageNet
}
MAX_LENGTH = 128   # tokens; longest headline + abstract pair is 87
IMAGE_SIZE = 224   # shorter side -> 224, centre crop 224

# Feature variants compared per modality (the text one is chosen on val macro-F1)
TEXT_FEATS = ["roberta_cls", "roberta_mean"]
IMAGE_FEATS = ["vit"]

# MLP head training
SEEDS = [0, 1, 2]
EPOCHS = 60
PATIENCE = 8          # epochs without val macro-F1 gain before early stopping
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-2
HIDDEN = 512
DROPOUT = 0.3
