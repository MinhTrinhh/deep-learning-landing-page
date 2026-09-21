import os

import argparse
import json
import platform
import random
import sys
from importlib.metadata import PackageNotFoundError, version

import models
import dataloader
import trainer
import edaworker
import metrics
import torch
from config import (
    BATCH_SIZE,
    DROP_OUT,
    LEARNING_RATE,
    MEAN_TUP,
    NUM_EPOCHS,
    NUM_WORKERS,
    OUTPUT_PATH,
    SD_TUP,
    SEED,
    VAL_SIZE,
    WD,
)
import plotter
import numpy as np

def set_seed(seed):
    """Configure deterministic random-number generation for a complete run."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

def get_package_versions():
    packages = ("torch", "torchvision", "numpy", "scikit-learn", "matplotlib")
    versions = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not installed"
    return versions

def save_reproducibility_manifest(args, data_loader=None):
    """Save all settings needed to understand and repeat this run."""
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    manifest = {
        "command_options": vars(args),
        "seed": SEED,
        "deterministic_algorithms": True,
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": get_package_versions(),
        "configuration": {
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "validation_fraction": VAL_SIZE,
            "normalization_mean": MEAN_TUP,
            "normalization_std": SD_TUP,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WD,
            "epochs": NUM_EPOCHS,
            "dropout": DROP_OUT,
            "training_augmentation": {
                "random_horizontal_flip_probability": 0.5,
                "random_affine_degrees": [-10, 10],
                "random_affine_translation": [0.1, 0.1],
                "random_affine_scale": [0.9, 1.1],
            },
        },
        "dataset_split": (
            data_loader.get_split_metadata() if data_loader is not None else None
        ),
    }
    path = OUTPUT_PATH / "reproducibility_manifest.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    print(f"Reproducibility manifest saved to: {path}")

def process_linear_model(data_loader, model_trainer, result_plotter, metric_calculator):
    set_seed(SEED)
    data_loader.reset_train_generator(SEED)
    linear_model = models.LinearModel()
    linear_loss_func = torch.nn.CrossEntropyLoss()
    linear_optimizer = torch.optim.AdamW(linear_model.parameters(), lr=LEARNING_RATE, weight_decay=WD)

    epoch_train_loss,\
        epoch_train_accuracies,\
            epoch_val_loss,\
                epoch_val_accuracies = model_trainer.train_model(
        model=linear_model,
        train_loader=data_loader.get_train_loader(),
        val_loader=data_loader.get_val_loader(),
        loss_func=linear_loss_func,
        optimizer=linear_optimizer,
        epochs=NUM_EPOCHS,
        model_name="linear")

    test_loss, test_targets, test_predictions = model_trainer.test_model(
        model=linear_model,
        test_loader=data_loader.get_test_loader(),
        loss_func=linear_loss_func)
    metric_calculator.report(
        test_targets,
        test_predictions,
        model_name="Linear",
        test_loss=test_loss,
    )

    result_plotter.plot(
        epoch_train_loss,
        epoch_train_accuracies,
        epoch_val_loss,
        epoch_val_accuracies,
        model_name="Linear")

def process_mlp_model(data_loader, model_trainer, result_plotter, metric_calculator):
    set_seed(SEED)
    data_loader.reset_train_generator(SEED)
    mlp_model = models.MLPModel(dropout=DROP_OUT)
    mlp_loss_func = torch.nn.CrossEntropyLoss()
    mlp_optimizer = torch.optim.AdamW(mlp_model.parameters(), lr=LEARNING_RATE, weight_decay=WD)

    epoch_train_loss,\
        epoch_train_accuracies,\
            epoch_val_loss,\
                epoch_val_accuracies = model_trainer.train_model(
        model=mlp_model,
        train_loader=data_loader.get_train_loader(),
        val_loader=data_loader.get_val_loader(),
        loss_func=mlp_loss_func,
        optimizer=mlp_optimizer,
        epochs=NUM_EPOCHS,
        model_name="mlp")

    test_loss, test_targets, test_predictions = model_trainer.test_model(
        model=mlp_model,
        test_loader=data_loader.get_test_loader(),
        loss_func=mlp_loss_func)
    metric_calculator.report(
        test_targets,
        test_predictions,
        model_name="MLP",
        test_loss=test_loss,
    )

    result_plotter.plot(
        epoch_train_loss,
        epoch_train_accuracies,
        epoch_val_loss,
        epoch_val_accuracies,
        model_name="MLP")

def parse_args():
    parser = argparse.ArgumentParser(description="FashionMNIST experiment pipeline")
    parser.add_argument(
        "--run-eda",
        action="store_true",
        help="Run EDA and save its figures before model training",
    )
    parser.add_argument(
        "--eda-only",
        action="store_true",
        help="Run EDA without training a model",
    )
    parser.add_argument(
        "--model",
        choices=("linear", "mlp", "both"),
        default="mlp",
        help="Model family to train (default: mlp)",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    set_seed(SEED)
    save_reproducibility_manifest(args)

    if args.run_eda or args.eda_only:
        edaworker.EDAWorker().run()
    if args.eda_only:
        raise SystemExit(0)

    data_loader = dataloader.FashionMNISTDataLoader()
    save_reproducibility_manifest(args, data_loader)
    model_trainer = trainer.Trainer()
    result_plotter = plotter.Plotter()
    metric_calculator = metrics.MetricCalculator()

    if args.model in ("linear", "both"):
        process_linear_model(
            data_loader, model_trainer, result_plotter, metric_calculator
        )
    if args.model in ("mlp", "both"):
        process_mlp_model(
            data_loader, model_trainer, result_plotter, metric_calculator
        )
