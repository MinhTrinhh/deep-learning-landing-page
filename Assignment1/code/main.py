import os
import argparse
import random

import dataloader
import edaworker
import metrics
import models
import numpy as np
import plotter
import torch
import trainer
from config import (
    DROP_OUT,
    LEARNING_RATE,
    NUM_EPOCHS,
    SEED,
    WD,
)

def set_seed(seed):
    """Configure deterministic random-number generation for a complete run."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

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
    
    metric_results = metric_calculator.report(
        test_targets,
        test_predictions,
        model_name="Linear",
        test_loss=test_loss,
    )
    result_plotter.plot_confusion_matrix(
        metric_results["confusion_matrix"], model_name="Linear"
    )

    result_plotter.plot_learning_curves(
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
    metric_results = metric_calculator.report(
        test_targets,
        test_predictions,
        model_name="MLP",
        test_loss=test_loss,
    )
    result_plotter.plot_confusion_matrix(
        metric_results["confusion_matrix"], model_name="MLP"
    )

    result_plotter.plot_learning_curves(
        epoch_train_loss,
        epoch_train_accuracies,
        epoch_val_loss,
        epoch_val_accuracies,
        model_name="MLP")

def parse_args():
    parser = argparse.ArgumentParser(description="FashionMNIST experiment pipeline")
    parser.add_argument(
        "--eda",
        action="store_true",
        help="Run EDA",
    )
    parser.add_argument(
        "--model",
        choices=("linear", "mlp", "all"),
        help="Model family to train (default: mlp)",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    set_seed(SEED)
    result_plotter = plotter.Plotter()

    if args.eda:
        eda_results = edaworker.EDAWorker().run()
        result_plotter.plot_class_distribution(eda_results["class_counts"])
        result_plotter.plot_examples(
            eda_results["example_images"], eda_results["example_labels"]
        )

    data_loader = dataloader.FashionMNISTDataLoader()
    model_trainer = trainer.Trainer()
    metric_calculator = metrics.MetricCalculator()

    if args.model in ("linear", "all"):
        process_linear_model(
            data_loader, model_trainer, result_plotter, metric_calculator
        )
    if args.model in ("mlp", "all"):
        process_mlp_model(
            data_loader, model_trainer, result_plotter, metric_calculator
        )
