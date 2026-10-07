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

def process_linear_model(
    data_loader, model_trainer, result_plotter, metric_calculator, run_label,
    eval_only=False,
):
    set_seed(SEED)
    data_loader.reset_train_generator(SEED)
    linear_model = models.LinearModel()
    linear_loss_func = torch.nn.CrossEntropyLoss()
    model_name = f"linear_{run_label}"
    learning_history = None

    if eval_only:
        model_trainer.restore_best_checkpoint(linear_model, model_name)
    else:
        linear_optimizer = torch.optim.AdamW(
            linear_model.parameters(), lr=LEARNING_RATE, weight_decay=WD
        )
        learning_history = model_trainer.train_model(
            model=linear_model,
            train_loader=data_loader.get_train_loader(),
            val_loader=data_loader.get_val_loader(),
            loss_func=linear_loss_func,
            optimizer=linear_optimizer,
            epochs=NUM_EPOCHS,
            model_name=model_name)

    test_loss, test_targets, test_predictions = model_trainer.test_model(
        model=linear_model,
        test_loader=data_loader.get_test_loader(),
        loss_func=linear_loss_func)
    
    sample_batch_images, _ = next(iter(data_loader.get_test_loader())) #to get a sample batch of images for resource metrics calculation

    metric_results = metric_calculator.report(
        test_targets,
        test_predictions,
        model_name=f"Linear_{run_label}",
        test_loss=test_loss,
        resource_metrics=metric_calculator.calculate_resource_metrics(
            model=linear_model,
            sample_input=sample_batch_images,
            train_time_seconds=model_trainer.last_train_time,
        ),
    )
    result_plotter.plot_confusion_matrix(
        metric_results["confusion_matrix"], model_name=f"Linear_{run_label}"
    )

    if learning_history is not None:
        result_plotter.plot_learning_curves(
            *learning_history,
            model_name=f"Linear_{run_label}")

def process_mlp_model(
    data_loader, model_trainer, result_plotter, metric_calculator, run_label,
    eval_only=False,
):
    set_seed(SEED)
    data_loader.reset_train_generator(SEED)
    mlp_model = models.MLPModel(dropout=DROP_OUT)
    mlp_loss_func = torch.nn.CrossEntropyLoss()
    model_name = f"mlp_{run_label}"
    learning_history = None

    if eval_only:
        model_trainer.restore_best_checkpoint(mlp_model, model_name)
    else:
        mlp_optimizer = torch.optim.AdamW(
            mlp_model.parameters(), lr=LEARNING_RATE, weight_decay=WD
        )
        learning_history = model_trainer.train_model(
            model=mlp_model,
            train_loader=data_loader.get_train_loader(),
            val_loader=data_loader.get_val_loader(),
            loss_func=mlp_loss_func,
            optimizer=mlp_optimizer,
            epochs=NUM_EPOCHS,
            model_name=model_name)

    test_loss, test_targets, test_predictions = model_trainer.test_model(
        model=mlp_model,
        test_loader=data_loader.get_test_loader(),
        loss_func=mlp_loss_func)
    
    sample_batch_images, _ = next(iter(data_loader.get_test_loader()))

    metric_results = metric_calculator.report(
        test_targets,
        test_predictions,
        model_name=f"MLP_{run_label}",
        test_loss=test_loss,
        resource_metrics=metric_calculator.calculate_resource_metrics(
            model=mlp_model,
            sample_input=sample_batch_images,
            train_time_seconds=model_trainer.last_train_time,
        ),
    )
    result_plotter.plot_confusion_matrix(
        metric_results["confusion_matrix"], model_name=f"MLP_{run_label}"
    )

    if learning_history is not None:
        result_plotter.plot_learning_curves(
            *learning_history,
            model_name=f"MLP_{run_label}")

def process_cnn_model(
    data_loader, model_trainer, result_plotter, metric_calculator, run_label,
    eval_only=False,
):
    set_seed(SEED)
    data_loader.reset_train_generator(SEED)
    cnn_model = models.CNNModel(dropout=DROP_OUT)
    cnn_loss_func = torch.nn.CrossEntropyLoss()
    model_name = f"cnn_{run_label}"
    learning_history = None

    if eval_only:
        model_trainer.restore_best_checkpoint(cnn_model, model_name)
    else:
        cnn_optimizer = torch.optim.AdamW(
            cnn_model.parameters(), lr=LEARNING_RATE, weight_decay=WD
        )
        learning_history = model_trainer.train_model(
            model=cnn_model,
            train_loader=data_loader.get_train_loader(),
            val_loader=data_loader.get_val_loader(),
            loss_func=cnn_loss_func,
            optimizer=cnn_optimizer,
            epochs=NUM_EPOCHS,
            model_name=model_name)

    test_loss, test_targets, test_predictions = model_trainer.test_model(
        model=cnn_model,
        test_loader=data_loader.get_test_loader(),
        loss_func=cnn_loss_func)

    sample_batch_images, _ = next(iter(data_loader.get_test_loader()))

    metric_results = metric_calculator.report(
        test_targets,
        test_predictions,
        model_name=f"CNN_{run_label}",
        test_loss=test_loss,
        resource_metrics=metric_calculator.calculate_resource_metrics(
            model=cnn_model,
            sample_input=sample_batch_images,
            train_time_seconds=model_trainer.last_train_time,
        ),
    )
    result_plotter.plot_confusion_matrix(
        metric_results["confusion_matrix"], model_name=f"CNN_{run_label}"
    )

    if learning_history is not None:
        result_plotter.plot_learning_curves(
            *learning_history,
            model_name=f"CNN_{run_label}")

def process_gru_model(
    data_loader, model_trainer, result_plotter, metric_calculator, run_label,
    eval_only=False,
):
    set_seed(SEED)
    data_loader.reset_train_generator(SEED)
    gru_model = models.GRUModel(dropout=DROP_OUT)
    gru_loss_func = torch.nn.CrossEntropyLoss()
    model_name = f"gru_{run_label}"
    learning_history = None

    if eval_only:
        model_trainer.restore_best_checkpoint(gru_model, model_name)
    else:
        gru_optimizer = torch.optim.AdamW(
            gru_model.parameters(), lr=LEARNING_RATE, weight_decay=WD
        )
        learning_history = model_trainer.train_model(
            model=gru_model,
            train_loader=data_loader.get_train_loader(),
            val_loader=data_loader.get_val_loader(),
            loss_func=gru_loss_func,
            optimizer=gru_optimizer,
            epochs=NUM_EPOCHS,
            model_name=model_name)

    test_loss, test_targets, test_predictions = model_trainer.test_model(
        model=gru_model,
        test_loader=data_loader.get_test_loader(),
        loss_func=gru_loss_func)

    sample_batch_images, _ = next(iter(data_loader.get_test_loader()))

    metric_results = metric_calculator.report(
        test_targets,
        test_predictions,
        model_name=f"GRU_{run_label}",
        test_loss=test_loss,
        resource_metrics=metric_calculator.calculate_resource_metrics(
            model=gru_model,
            sample_input=sample_batch_images,
            train_time_seconds=model_trainer.last_train_time,
        ),
    )
    result_plotter.plot_confusion_matrix(
        metric_results["confusion_matrix"], model_name=f"GRU_{run_label}"
    )

    if learning_history is not None:
        result_plotter.plot_learning_curves(
            *learning_history,
            model_name=f"GRU_{run_label}")

def parse_args():
    parser = argparse.ArgumentParser(description="FashionMNIST experiment pipeline")
    parser.add_argument(
        "--eda",
        action="store_true",
        help="Run EDA",
    )
    parser.add_argument(
        "--model",
        default="all",
        choices=("linear", "mlp", "cnn", "gru", "all"),
        help="Model family to train (default: all)",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Load the matching best checkpoint and evaluate without training",
    )
    augmentation_group = parser.add_mutually_exclusive_group()
    augmentation_group.add_argument(
        "--aug",
        dest="use_augmentation",
        action="store_true",
        help="Train with data augmentation (default)",
    )
    augmentation_group.add_argument(
        "--no-aug",
        dest="use_augmentation",
        action="store_false",
        help="Train without data augmentation",
    )
    parser.set_defaults(use_augmentation=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    set_seed(SEED)
    result_plotter = plotter.Plotter()

    if args.eda:
        eda_results = edaworker.EDAWorker().run()
        core_eda = eda_results["core_eda"]
        task_specific_eda = eda_results["task_specific_eda"]
        result_plotter.plot_class_distribution(core_eda["class_counts"])
        result_plotter.plot_examples(eda_results["example_images"], eda_results["example_labels"])
        result_plotter.plot_dimensionality_reduction(task_specific_eda["instance_level"])
        result_plotter.plot_similarity_matrix(task_specific_eda["class_level"]["similarity_matrix"])

    run_label = "aug" if args.use_augmentation else "no_aug"
    data_loader = dataloader.FashionMNISTDataLoader(
        use_augmentation=args.use_augmentation
    )
    model_trainer = trainer.Trainer()
    metric_calculator = metrics.MetricCalculator()

    if args.model in ("linear", "all"):
        process_linear_model(
            data_loader, model_trainer, result_plotter, metric_calculator, run_label,
            eval_only=args.eval_only,
        )
    if args.model in ("mlp", "all"):
        process_mlp_model(
            data_loader, model_trainer, result_plotter, metric_calculator, run_label,
            eval_only=args.eval_only,
        )
    if args.model in ("cnn", "all"):
        process_cnn_model(
            data_loader, model_trainer, result_plotter, metric_calculator, run_label,
            eval_only=args.eval_only,
        )
    if args.model in ("gru", "all"):
        process_gru_model(
            data_loader, model_trainer, result_plotter, metric_calculator, run_label,
            eval_only=args.eval_only,
        )
