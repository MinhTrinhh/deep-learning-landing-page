# Assignment 3 Codebase

N24News, 24-class news classification from headline + abstract (text) and the article image.
All scripts are run from `Assignment3/`; on Colab use [`notebook/02_colab_runner.ipynb`](../notebook/02_colab_runner.ipynb).

| File | Role |
|---|---|
| `dataset.py` | Loads `nytimes.json` in the order of `outputs/eda/splits.csv` (duplicate-aware split); text cleaning; image preprocessing (shorter side 224 → centre crop 224, removes the year badge) |
| `extract_features.py` | Frozen encoders → cached features: `roberta_cls`, `roberta_mean` (RoBERTa-base, sentence pair, max 128 tokens), `vit` (ViT-B/16, ImageNet; image baseline), `clip` (CLIP ViT-B/16; optional side experiment) |
| `models.py` | `MLPHead`: per-modality LayerNorm → concat → MLP. Same head for text-only, image-only and concat fusion |
| `train.py` | Trains one head on cached features; AdamW + cosine, early stopping on **val macro-F1**; test scored once with the selected checkpoint |
| `evaluate.py` | mean ± std over seeds, per-class F1, confusion matrices, text/image/fusion agreement (support vs. conflict), copies best checkpoints |

## M2 pipeline

```bash
# 1. features (GPU, once)
python code/extract_features.py --data DATA_DIR --out FEAT_DIR --encoder roberta
python code/extract_features.py --data DATA_DIR --out FEAT_DIR --encoder vit

# 2. unimodal baselines (repeat for seeds 0 1 2)
python code/train.py --features FEAT_DIR --out RUNS_DIR --model text  --text-feat roberta_cls --seed 0
python code/train.py --features FEAT_DIR --out RUNS_DIR --model image --image-feat vit       --seed 0

# 3. simple fusion with the text variant chosen on validation (image is always vit)
python code/evaluate.py --runs RUNS_DIR --best text     # -> e.g. roberta_mean
python code/train.py --features FEAT_DIR --out RUNS_DIR --model concat --text-feat roberta_mean --image-feat vit --seed 0

# 4. tables + figures + checkpoints
python code/evaluate.py --runs RUNS_DIR --out outputs/metrics/m2 --checkpoints checkpoints --data DATA_DIR
```

Smoke test without a GPU: add `--limit 600` to `extract_features.py` and `--debug` (10 % train sample) to `train.py`.
Debug runs are excluded from `evaluate.py` unless `--include-debug` is passed, and are never reported.

**Why ViT and not CLIP for the image baseline.** CLIP's vision tower was pretrained with language supervision and can read text rendered in images (OCR), so an "image-only" CLIP model would partly carry language information. Unimodal B therefore uses ImageNet ViT-B/16, the same kind of encoder as the N24News paper. `--encoder clip` remains available for an optional, clearly labelled side experiment.
