import json
import time
from pathlib import Path

import torch
from thop import profile
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from config import CLASS_NAMES, OUTPUT_PATH


class MetricCalculator:
    def __init__(self, output_dir=OUTPUT_PATH / "metrics"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def calculate(self, targets, predictions):
        """Calculate classification metrics from one complete dataset pass."""
        labels = list(range(len(CLASS_NAMES)))
        return {
            "accuracy": accuracy_score(targets, predictions),
            "macro_precision": precision_score(
                targets, predictions, average="macro", zero_division=0
            ),
            "macro_recall": recall_score(
                targets, predictions, average="macro", zero_division=0
            ),
            "macro_f1": f1_score(targets, predictions, average="macro"),
            "confusion_matrix": confusion_matrix(
                targets, predictions, labels=labels
            ),
            "classification_report": classification_report(
                targets,
                predictions,
                labels=labels,
                target_names=CLASS_NAMES,
                digits=4,
                zero_division=0,
            ),
        }

    @staticmethod
    def calculate_resource_metrics(model, sample_input, train_time_seconds):
        batch_size = sample_input.shape[0]

        # use sample_input to calculate FLOPs using thop library
        macs, _ = profile(model, inputs=(sample_input,), verbose=False)

        # total num of params
        parameters = sum(parameter.numel() for parameter in model.parameters())

        # model size in bytes
        total_bytes = sum(parameter.numel() * parameter.element_size()
                        for parameter in model.parameters())

        # measure inference time
        was_training = model.training
        model.eval()
        with torch.no_grad():
            for _ in range(10):
                model(sample_input) #warm up

            if sample_input.is_cuda:
                torch.cuda.synchronize() #make cpu wait for gpu to finish
            start_time = time.perf_counter() 
            for _ in range(50):
                model(sample_input)
            if sample_input.is_cuda:
                torch.cuda.synchronize() #make cpu wait for gpu to finish

        if was_training: #return the to previous state
            model.train()

        inference_ms_per_batch = (time.perf_counter() - start_time) / 50 * 1000 #in milliseconds

        return {
            "train_time_seconds": float(train_time_seconds),
            "inference_ms_per_sample": float(inference_ms_per_batch / batch_size),
            "flops_b": float((macs * 2 / batch_size) / 1e9),
            "model_size_mb": float(total_bytes / (1024 * 1024)),
            "params_m": int(parameters) / 1e6,
        }

    def report(
        self,
        targets,
        predictions,
        model_name,
        test_loss=None,
        resource_metrics=None,
    ):
        """Print and save numeric metrics for a named model."""
        results = self.calculate(targets, predictions)

        print(f"\n=== {model_name} test results ===")
        if test_loss is not None:
            print(f"Test loss: {test_loss:.4f}")
        print(f"Test accuracy: {results['accuracy']:.4f}")
        print(f"Macro-Precision: {results['macro_precision']:.4f}")
        print(f"Macro-Recall: {results['macro_recall']:.4f}")
        print(f"Macro-F1: {results['macro_f1']:.4f}")

        #for resource metrics
        if resource_metrics is not None:
            print(f"Train time: {resource_metrics['train_time_seconds']:.2f}s")
            print(
                "Inference time: "
                f"{resource_metrics['inference_ms_per_sample']:.6f} ms/sample"
            )
            print(f"FLOPs: {resource_metrics['flops_b']:.6f} B/sample")
            print(f"Parameters: {resource_metrics['params_m']:.6f} M")
            print(f"Model size: {resource_metrics['model_size_mb']:.3f} MB")

        print(results["classification_report"])

        serializable_results = {
            "test_loss": float(test_loss) if test_loss is not None else None,
            "accuracy": float(results["accuracy"]),
            "macro_precision": float(results["macro_precision"]),
            "macro_recall": float(results["macro_recall"]),
            "macro_f1": float(results["macro_f1"]),
            "resource_metrics": resource_metrics,
            "confusion_matrix": results["confusion_matrix"].tolist(),
            "classification_report": results["classification_report"],
        }
        metrics_path = self.output_dir / f"{model_name.lower()}_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as file:
            json.dump(serializable_results, file, indent=2)
        print(f"Numeric metrics saved to: {metrics_path}")

        return results
