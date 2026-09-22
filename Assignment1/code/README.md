# Assignment 1 Codebase

Place all PyTorch code for Assignment 1 here:
- `dataset.py` / `dataloader.py`: Dataset loading & augmentation for Fashion-MNIST (and MNIST / CIFAR-10)
- `models/`: Implementations for Linear/Softmax, MLP, CNN, LSTM/GRU, Vision Transformer
- `edaworker.py`: Reproducible dataset statistics and EDA inputs
- `trainer.py`: Unified PyTorch training, validation, and test passes
- `metrics.py`: Accuracy, macro-F1, class report, and confusion-matrix values
- `plotter.py`: All EDA, metric, and learning-curve figure generation

## Running the workflow

Run these commands from `Assignment1/code`:

```bash
# Generate EDA statistics and figures without training
PYTHONHASHSEED=42 python main.py --eda-only

# Train and evaluate one model
PYTHONHASHSEED=42 python main.py --model linear
PYTHONHASHSEED=42 python main.py --model mlp

# Generate EDA, then train and compare both required model families
PYTHONHASHSEED=42 python main.py --run-eda --model both
```

Generated artifacts are stored under `Assignment1/outputs`:

- `eda/`: class distribution and random dataset examples
- `learning_curves/`: training/validation loss and accuracy
- `metrics/`: test confusion matrices

The best validation-loss checkpoint for each model is stored under
`Assignment1/checkpoints` as `best_linear.pt` and `best_mlp.pt`. Each checkpoint
contains the model parameters (`model_state_dict`), optimizer state, best epoch,
validation loss, and validation accuracy. The best parameters are restored
automatically before test evaluation.

## Reproducibility

Experiment settings, including the seed, dataset split, normalization, batch
size, training hyperparameters, and model settings, are defined in `config.py`.
The dependency versions are pinned in the repository's `requirements.txt`.

The workflow uses seed `42` for Python, NumPy, PyTorch, DataLoader shuffling,
DataLoader workers, random augmentation, dropout, and model initialization. It
also enables PyTorch deterministic algorithms. Run with the same configuration,
dependency versions, command options, and hardware/software environment when
exact repeatability is required.

To verify a repeated run, compare the generated test metric JSON files.
