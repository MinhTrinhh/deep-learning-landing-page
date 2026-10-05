# Assignment 1 — Foundations of Deep Learning Pipelines and Architectures

Fashion-MNIST classification experiments for the Linear/Softmax and multilayer perceptron models.

- [Assignment 1 website](https://MinhTrinhh.github.io/deep-learning-landing-page/Assignment1/)
- [AI usage disclosure](AI_USAGE.md)

## Directory Structure

```text
Assignment1/
├── AI_USAGE.md              # Assignment-specific AI disclosure
├── README.md                # Installation and execution guide
├── index.html               # Assignment website
├── code/
│   ├── config.py            # Paths and experiment configuration
│   ├── dataloader.py        # Dataset download, split, transforms, and loaders
│   ├── edaworker.py         # EDA calculations
│   ├── main.py              # Command-line entry point
│   ├── metrics.py           # Predictive and resource metrics
│   ├── models.py            # Linear and MLP architectures
│   ├── plotter.py           # EDA, learning-curve, and metric plots
│   └── trainer.py           # Training, validation, testing, and checkpoints
├── checkpoints/             # Best-validation-loss model checkpoints
├── data/                    # Automatically downloaded Fashion-MNIST data
├── diagram-pipeline/        # Methodology diagram assets
└── outputs/
    ├── eda/                 # EDA figures
    ├── learning_curves/     # Training and validation curves
    └── metrics/             # Confusion-matrix figures
```

## Installation

Create and activate the virtual environment by following the [repository setup guide](../README.md#local-environment-setup). Then install the dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

Move into the Assignment 1 code directory before running the commands below:

```bash
cd Assignment1/code
```

## Dataset Preparation

No manual download is required. `torchvision` downloads Fashion-MNIST into `Assignment1/data/` the first time an experiment runs. The loader then creates the training and validation subsets automatically.

To generate the EDA outputs before training both implemented models:

```bash
python main.py --eda --model all
```

`--eda` currently runs EDA first and then continues with the selected model training; there is no separate EDA-only command.

## Training Commands

Train the Linear/Softmax model with the default augmentation:

```bash
python main.py --model linear
```

Train the MLP with the default augmentation:

```bash
python main.py --model mlp
```

Train both implemented models sequentially:

```bash
python main.py --model all
```

Disable augmentation for any model selection:

```bash
python main.py --model linear --no-aug
python main.py --model mlp --no-aug
python main.py --model all --no-aug
```

Use `--aug` explicitly when desired; augmentation is enabled by default.

## Evaluation Commands

Evaluation is automatic after every training command. The trainer restores the checkpoint with the lowest validation loss, evaluates it on the official test set, prints the metrics, and writes the generated artifacts under `Assignment1/outputs/`.

For example, train and evaluate the MLP in one command:

```bash
python main.py --model mlp
```

There is currently no standalone evaluate-only command.
