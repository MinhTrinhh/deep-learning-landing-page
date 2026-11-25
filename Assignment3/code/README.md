# Assignment 3 Codebase

N24News, 24-class news classification from headline + abstract (text) and the article image.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MinhTrinhh/deep-learning-landing-page/blob/main/Assignment3/notebook/02_colab_runner.ipynb)
· [Assignment page](../index.html) · [Dataset proposal](../dataproposal/dataproposal-a3.pdf) · [Checkpoints](../checkpoints/README.md)

| File | Role |
|---|---|
| `config.py` | All shared settings: split file, encoder checkpoints, max length / image size, feature variants, seeds, head hyperparameters |
| `dataset.py` | Loads `nytimes.json` in the order of `outputs/eda/splits.csv` (duplicate-aware split); text cleaning; image preprocessing (shorter side 224 → centre crop 224, removes the year badge) |
| `extract_features.py` | Frozen encoders → cached features: `roberta_cls`, `roberta_mean` (RoBERTa-base, sentence pair, max 128 tokens), `vit` (ViT-B/16, ImageNet) |
| `models.py` | `MLPHead`: per-modality LayerNorm → concat → MLP. Same head for text-only, image-only and concat fusion |
| `train.py` | Trains one head on cached features; AdamW + cosine, early stopping on **val macro-F1**; test scored once with the selected checkpoint |
| `evaluate.py` | mean ± std over seeds, per-class F1, confusion matrices, text/image/fusion agreement (support vs. conflict), copies best checkpoints |

## Run on Colab (any account)

1. Open the notebook with the badge above, then *Runtime → Change runtime type → T4 GPU*.
2. *Runtime → Run all* and allow Google Drive access. Features and runs are written to `MyDrive/DL_A3/` of the account running it.
3. The last cell downloads `a3_m2_results.zip` (tables, figures, checkpoints); unzip it into `Assignment3/`.

The first session takes about 30–60 min on a T4 (download + feature extraction); later sessions reuse the cached features.
If `gdown` hits a quota error, add [N24News.zip](https://drive.google.com/file/d/1OS1fXwZ1Vsj70lEQajccyssxQRYp5X9D) to your Drive, place it in `MyDrive/DL_A3/` and rerun the data cell.

## Run locally

```bash
pip install -r requirements_assignment3.txt     # from the repository root
cd Assignment3

# 1. features (GPU, once)
python code/extract_features.py --data DATA_DIR --out FEAT_DIR --encoder roberta
python code/extract_features.py --data DATA_DIR --out FEAT_DIR --encoder vit

# 2. unimodal baselines (repeat for seeds 0 1 2)
python code/train.py --features FEAT_DIR --out RUNS_DIR --model text  --text-feat roberta_cls --seed 0
python code/train.py --features FEAT_DIR --out RUNS_DIR --model text  --text-feat roberta_mean --seed 0
python code/train.py --features FEAT_DIR --out RUNS_DIR --model image --image-feat vit --seed 0

# 3. simple fusion with the text variant chosen on validation
python code/evaluate.py --runs RUNS_DIR --best text     # -> e.g. roberta_mean
python code/train.py --features FEAT_DIR --out RUNS_DIR --model concat --text-feat roberta_mean --image-feat vit --seed 0

# 4. tables + figures + checkpoints
python code/evaluate.py --runs RUNS_DIR --out outputs/metrics/m2 --checkpoints checkpoints --data DATA_DIR
```

Every run stores its full configuration (all hyperparameters, seed, features) in `RUNS_DIR/<run>/seed<k>/metrics.json`,
and each feature file has a `<name>.json` with the encoder checkpoint, GPU and extraction time.

**Why ViT and not CLIP for the image baseline.** CLIP's vision tower was pretrained with language supervision and can read text rendered in images (OCR), so an "image-only" CLIP model would partly carry language information. Unimodal B therefore uses ImageNet ViT-B/16, the same kind of encoder as the N24News paper.
