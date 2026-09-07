# Deep Learning and Its Applications (CO3133) - Course Project Repository

**Course:** CO3133 - Deep Learning and Its Applications (Semester-261)  
**Institution:** Ho Chi Minh City University of Technology (HCMUT), VNU-HCM  
**Faculty:** Faculty of Computer Science and Engineering  
**Instructor:** Lê Thành Sách  
**Group:** Group [ID]  
**Live GitHub Pages Site:** [https://your-username.github.io/deep-learning-repo/docs/](https://your-username.github.io/deep-learning-repo/docs/)

---

## 📁 Repository Structure

```text
[group-name].github.io/
├── README.md                 # Top-level reproducibility & project documentation (Sec 4.2)
├── docs/                     # GitHub Pages live site root (Sec 2.1)
│   ├── index.html            # Shared Landing Page
│   ├── AI_USAGE.md           # Repository-wide AI disclosure log (Sec 5.1)
│   └── assets/               # CSS styles & JS scripts
├── Assignment1/              # Assignment 1 Package (40% Weight)
│   ├── index.html            # Assignment 1 GitHub Pages detail page (Sec 2.2)
│   ├── code/                 # PyTorch pipeline & model implementations
│   ├── checkpoints/          # Model weights & reconstruction scripts
│   └── Presentation/         # Slides & presentation materials
├── Assignment2/              # Assignment 2 Package (30% Weight)
│   ├── index.html            # Assignment 2 GitHub Pages detail page
│   ├── code/                 # Specialized task track codebase
│   ├── checkpoints/          # Fine-tuned model checkpoints
│   └── Presentation/         # Slides & report
└── Assignment3/              # Assignment 3 Package (30% Weight)
    ├── index.html            # Assignment 3 GitHub Pages detail page
    ├── code/                 # Multimodal encoders & fusion models
    ├── checkpoints/          # Multimodal model checkpoints
    └── Presentation/         # Slides & report
```

---

## 🛠️ Reproducibility Requirements (Section 4.2)

### 1. Environment Setup & Installation
```bash
# Clone the repository
git clone https://github.com/your-username/deep-learning-repo.git
cd deep-learning-repo

# Create and activate a conda environment
conda create -n dl-co3133 python=3.10 -y
conda activate dl-co3133

# Install PyTorch and required dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 2. Dependency Versions & Hardware Specifications
- **Python:** `3.10.12`
- **PyTorch:** `2.1.0+cu118`
- **Torchvision:** `0.16.0+cu118`
- **Hardware Used:** NVIDIA RTX 3090 (24GB VRAM) / Apple M2 Max / Google Colab T4
- **Random Seed:** `42` (Enforced across numpy, torch, and Python random for all experiments)

---

## 🚀 Execution & Evaluation Commands

### Assignment 1: Foundations of Deep Learning Pipelines
```bash
# Data preparation & training linear / MLP / CNN / LSTM / ViT
python Assignment1/code/train.py --model cnn --dataset FashionMNIST --epochs 20 --seed 42
python Assignment1/code/evaluate.py --model cnn --checkpoint Assignment1/checkpoints/best_cnn.pt
```

### Assignment 2: Deep Learning on Large-Scale Data
```bash
# Train specialized track baseline and fine-tune pretrained backbone
python Assignment2/code/train.py --config Assignment2/code/configs/finetune_config.yaml --seed 42
python Assignment2/code/evaluate.py --checkpoint Assignment2/checkpoints/best_model.pth
```

### Assignment 3: Multimodal Deep Learning
```bash
# Train unimodal baselines and multimodal fusion pipeline
python Assignment3/code/train_fusion.py --fusion intermediate --seed 42
python Assignment3/code/evaluate_multimodal.py --checkpoint Assignment3/checkpoints/best_fusion.pth
```

---

## 🤖 AI Usage Disclosure (Section 5)
All generative AI tool usage across the project is logged in [`docs/AI_USAGE.md`](docs/AI_USAGE.md). Each entry records tool name, version, prompt summary, student verification source, and responsible member.
