# Assignment 1 — Foundations of Deep Learning Pipelines and Architectures

**Course:** CO3133 - Deep Learning and Its Applications (Semester-261)  
**Institution:** Ho Chi Minh City University of Technology (HCMUT), VNU-HCM  
**Faculty:** Faculty of Computer Science and Engineering  
**Instructor:** Lê Thành Sách  
**Group:** G-10  
**Assignment Weight:** 40% of total course grade  

| Member | Student ID | GitHub | Role |
|--------|-----------|--------|------|
| Trịnh Lê Minh | 2352765 | [@MinhTrinhh](https://github.com/MinhTrinhh) | Pipeline Design & PyTorch Architecture |
| Trần Phước Sang | 2353044 | [@phuocsangcs](https://github.com/phuocsangcs) | Data Preprocessing & EDA Lead |
| Đặng Trần Thái Bảo | 2452120 | [@baodangtran](https://github.com/baodangtran) | Model Training & Hyperparameter Tuning |

**Links:**  
🌐 [Assignment 1 Page](https://MinhTrinhh.github.io/deep-learning-landing-page/Assignment1/)  
🌐 [Course Landing Page](https://MinhTrinhh.github.io/deep-learning-landing-page/docs/)  
📄 [AI Usage Disclosure](AI_USAGE.md)  
📁 [Full Repository](https://github.com/MinhTrinhh/deep-learning-landing-page)

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Dataset Preparation](#2-dataset-preparation)
3. [Training](#3-training)
4. [Evaluation](#4-evaluation)
5. [Reproducibility](#5-reproducibility)
6. [Checkpoints](#6-checkpoints)
7. [Result Traceability](#7-result-traceability)

---

## 1. Environment Setup

### Prerequisites

- **Python:** 3.10 or higher
- **OS:** Linux, macOS (Apple Silicon supported), or Windows
- **Hardware:** CPU-only (no GPU required). Tested on Intel-based laptops and Apple MacBook (ARM via CPU fallback).

### Install Dependencies

From the **repository root** (`deep-learning-landing-page/`):

```bash
pip install -r requirements.txt
```

**Pinned dependency versions** (from `requirements.txt`):

| Package | Version |
|---------|---------|
| torch | 2.14.0 |
| torchvision | 0.29.0 |
| numpy | 2.5.3 |
| scikit-learn | 1.9.1 |
| umap-learn | 0.5.12 |
| scipy | 1.18.1 |
| matplotlib | 3.11.2 |
| seaborn | 0.13.2 |
| thop | 0.1.1.post2209072238 |

> **Note:** All commands below must be run from `Assignment1/code/` unless otherwise specified.

---

## 2. Dataset Preparation

**Primary dataset:** Fashion-MNIST  
**Development/debugging dataset:** MNIST  

Fashion-MNIST and MNIST are **automatically downloaded** via `torchvision.datasets` on first run. No manual download is needed.

```bash
# Dataset is downloaded automatically on first training run.
# Default download path:
#   Assignment1/data/FashionMNIST/
#   Assignment1/data/MNIST/
```

**Dataset split policy** (configured in `config.py`):

| Split | Size | Method |
|-------|------|--------|
| Train | 54,000 | Stratified, seed 42 |
| Validation | 6,000 (10% of original train) | Stratified, seed 42 |
| Test | 10,000 | Official torchvision test set |

**Preprocessing:**
- Pixel normalization: `[0, 255]` → `[0.0, 1.0]` (via `ToTensor`)
- Data augmentation (train only): `RandomHorizontalFlip`, `RandomRotation(10°)`
- Input tensor shape: `[B, 1, 28, 28]`
- Class names: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

---

## 3. Training

All training commands are run from `Assignment1/code/`:

```bash
# EDA only (no training) — generates plots under Assignment1/outputs/eda/
PYTHONHASHSEED=42 python main.py --eda

# Train Linear/Softmax model
PYTHONHASHSEED=42 python main.py --model linear

# Train MLP model
PYTHONHASHSEED=42 python main.py --model mlp

# Train both Linear and MLP (sequential)
PYTHONHASHSEED=42 python main.py --model both

# Run EDA then train both models
PYTHONHASHSEED=42 python main.py --run-eda --model both
```

**Training configuration** (see `config.py` for all settings):

| Hyperparameter | Value |
|---------------|-------|
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.1 |
| Batch size | 256 |
| Max epochs | 5 (with early stopping patience 10) |
| Dropout (MLP) | 0.2 |
| Random seed | 42 |
| Deterministic mode | Enabled |
| Mixed precision | Not used |
| Scheduler | None |
| Checkpoint criterion | Lowest validation loss |

**Generated training artifacts** in `Assignment1/outputs/`:

```
outputs/
├── eda/
│   ├── class_distribution.png
│   ├── class_similarity_matrix.png
│   ├── class_dendrogram.png
│   ├── dimensionality_reduction.png
│   ├── random_examples.png
│   └── eda_summary.json
├── learning_curves/
│   ├── linear_learning_curves.png
│   ├── linear_no_aug_learning_curves.png
│   ├── mlp_learning_curves.png
│   └── mlp_no_aug_learning_curves.png
└── metrics/
    ├── linear_metrics.json
    ├── linear_no_aug_metrics.json
    ├── mlp_metrics.json
    └── mlp_no_aug_metrics.json
```

---

## 4. Evaluation

Evaluation runs **automatically** at the end of each training run. The best checkpoint (lowest validation loss) is restored before the test set is scored.

To verify results, inspect the generated JSON metric files:

```bash
# Example: check linear model test metrics
cat Assignment1/outputs/metrics/linear_metrics.json

# Example: check MLP model test metrics
cat Assignment1/outputs/metrics/mlp_metrics.json
```

**Key metrics** reported per run:
- `test_loss` — cross-entropy loss on test set
- `accuracy` — top-1 accuracy
- `macro_precision`, `macro_recall`, `macro_f1`
- `resource_metrics` — train time (seconds), inference ms/sample, FLOPs, model size MB, param count
- `confusion_matrix` — 10×10 matrix
- `classification_report` — per-class precision/recall/F1

**Current results** (Linear and MLP — CPU, seed 42, with augmentation):

| Model | Test Accuracy | Macro F1 | Params | Train Time |
|-------|-------------|---------|--------|-----------|
| Linear/Softmax | 73.51% | 0.7280 | 7,850 | 176.4 s |
| MLP | 83.19% | 0.8269 | 567,434 | 178.1 s |
| CNN | TBD | TBD | TBD | TBD |
| LSTM/GRU | TBD | TBD | TBD | TBD |
| Vision Transformer | TBD | TBD | TBD | TBD |

---

## 5. Reproducibility

### Seed & Determinism

The codebase sets **seed 42** globally at startup for:
- Python `random` module
- NumPy
- PyTorch (CPU and CUDA)
- DataLoader worker seeds
- Model parameter initialization

PyTorch deterministic algorithms are **enabled** (`torch.use_deterministic_algorithms(True)`).

To reproduce results, always prefix commands with `PYTHONHASHSEED=42`:

```bash
PYTHONHASHSEED=42 python main.py --model linear
```

### Config File

All experiment settings are centralized in [`code/config.py`](code/config.py):

```python
SEED        = 42
VAL_SIZE    = 0.1      # 10% of 60k train → 6,000 val samples
BATCH_SIZE  = 256
NUM_WORKERS = 1
LEARNING_RATE = 0.001
WD          = 0.1      # weight decay
NUM_EPOCHS  = 5
MAX_ATTEMPT = 10       # early stopping patience
DROP_OUT    = 0.2      # MLP dropout
```

### Hardware Note

Results were produced on **CPU-only** hardware:
- Intel-based laptop (Windows)
- Apple MacBook (ARM CPU via PyTorch CPU backend)

Exact numeric reproducibility across different hardware/OS platforms is **not guaranteed** due to floating-point non-determinism in CPU backends. For best reproducibility, match the OS, Python version, and dependency versions in `requirements.txt`.

### Dependency Versions

Exact versions are pinned in the repository root `requirements.txt`. Install with:

```bash
pip install -r requirements.txt
```

---

## 6. Checkpoints

Best-validation-loss checkpoints are saved automatically to `Assignment1/checkpoints/`:

```
checkpoints/
├── best_linear.pt           # Linear model (with augmentation)
├── best_linear_aug.pt       # Linear model (augmentation variant)
├── best_linear_no_aug.pt    # Linear model (no augmentation)
├── best_mlp.pt              # MLP model (with augmentation)
└── best_mlp_no_aug.pt       # MLP model (no augmentation)
```

Each `.pt` file is a PyTorch checkpoint dictionary containing:

| Key | Description |
|-----|-------------|
| `model_state_dict` | Learned parameter tensors |
| `optimizer_state_dict` | Optimizer state and parameter groups |
| `model_name` | Architecture identifier string |
| `epoch` | Epoch at which best validation loss was achieved |
| `val_loss` | Validation loss at the best epoch |
| `val_accuracy` | Validation accuracy at the best epoch |
| `seed` | Random seed used for the experiment |

### Loading a Checkpoint

```python
import torch
from models import LinearModel  # or MLPModel

checkpoint = torch.load("Assignment1/checkpoints/best_linear.pt", weights_only=True)
model = LinearModel()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(f"Best epoch:    {checkpoint['epoch']}")
print(f"Val loss:      {checkpoint['val_loss']:.4f}")
print(f"Val accuracy:  {checkpoint['val_accuracy']:.4f}")
```

---

## 7. Result Traceability

Every result can be traced back to its config, split, checkpoint, and log:

| Model | Config File | Split Seed | Checkpoint File | Metric Log |
|-------|-------------|-----------|-----------------|------------|
| Linear (aug) | `code/config.py` | `SEED=42, VAL_SIZE=0.1` | `checkpoints/best_linear.pt` | `outputs/metrics/linear_metrics.json` |
| Linear (no aug) | `code/config.py` | `SEED=42, VAL_SIZE=0.1` | `checkpoints/best_linear_no_aug.pt` | `outputs/metrics/linear_no_aug_metrics.json` |
| MLP (aug) | `code/config.py` | `SEED=42, VAL_SIZE=0.1` | `checkpoints/best_mlp.pt` | `outputs/metrics/mlp_metrics.json` |
| MLP (no aug) | `code/config.py` | `SEED=42, VAL_SIZE=0.1` | `checkpoints/best_mlp_no_aug.pt` | `outputs/metrics/mlp_no_aug_metrics.json` |

> **Verification:** To verify a repeated run, re-run with the same command and compare the generated metric JSON files against the committed versions in `outputs/metrics/`.
