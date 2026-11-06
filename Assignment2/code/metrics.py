import copy
import time
import matplotlib
matplotlib.use("Agg")
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from thop import profile
from dataset import (
    LABEL_NAMES
)

CLASS_NAMES = list(LABEL_NAMES)

def count_lstm_ops(module, inputs, output):
    """thop's default LSTM counter breaks on PackedSequence inputs, so use our own.

    Counts the multiply-accumulates of the four gate matmuls per valid token,
    for every layer and direction (elementwise gate ops are ignored).
    """
    x = inputs[0]
    if isinstance(x, nn.utils.rnn.PackedSequence):
        steps = x.data.shape[0]  # total valid tokens across the batch
    elif module.batch_first:
        steps = x.shape[0] * x.shape[1]
    else:
        steps = x.shape[0] * x.shape[1]
    directions = 2 if module.bidirectional else 1
    macs = 0
    for layer in range(module.num_layers):
        in_size = module.input_size if layer == 0 else module.hidden_size * directions
        macs += directions * steps * 4 * module.hidden_size * (in_size + module.hidden_size)
    module.total_ops += torch.DoubleTensor([int(macs)])


def _sync(tensor):
    if tensor.is_cuda:
        torch.cuda.synchronize()  # make cpu wait for gpu to finish
    elif tensor.device.type == "mps":
        torch.mps.synchronize()


class MetricCalculator:
    def calculate(self, targets, predictions):
        """Calculate classification metrics from one complete dataset pass."""
        labels = list(range(len(CLASS_NAMES)))
        return {
            "accuracy": accuracy_score(targets, predictions),
            "macro_precision": precision_score(
                targets, predictions, labels=labels, average="macro", zero_division=0
            ),
            "macro_recall": recall_score(
                targets, predictions, labels=labels, average="macro", zero_division=0
            ),
            "macro_f1": f1_score(
                targets, predictions, labels=labels, average="macro", zero_division=0
            ),
            "confusion_matrix": confusion_matrix(targets, predictions, labels=labels),
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
    def calculate_resource_metrics(model, sample_inputs, train_time_seconds):
        """model: an nn.Module taking *sample_inputs (e.g. InferenceWrapper)."""
        batch_size = sample_inputs[0].shape[0]

        # FLOPs via thop (profiled on a copy so the real model is never touched)
        macs = None
        try:
            macs, _ = profile(
                copy.deepcopy(model),
                inputs=tuple(sample_inputs),
                custom_ops={nn.LSTM: count_lstm_ops},
                verbose=False,
            )
        except Exception as error:  # thop can fail on exotic modules
            print(f"FLOPs could not be computed with thop: {error}")

        # total num of params
        parameters = sum(parameter.numel() for parameter in model.parameters())

        # model size in bytes
        total_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in model.parameters()
        )

        # measure inference time
        was_training = model.training
        model.eval()
        with torch.no_grad():
            for _ in range(10):
                model(*sample_inputs)  # warm up

            _sync(sample_inputs[0])
            start_time = time.perf_counter()
            for _ in range(50):
                model(*sample_inputs)
            _sync(sample_inputs[0])
            elapsed = time.perf_counter() - start_time

        if was_training:  # return to previous state
            model.train()

        inference_ms_per_batch = elapsed / 50 * 1000  # in milliseconds

        return {
            "train_time_seconds": (
                None if train_time_seconds is None else float(train_time_seconds)
            ),
            "inference_ms_per_sample": float(inference_ms_per_batch / batch_size),
            "flops_b": (
                None if macs is None else float((macs * 2 / batch_size) / 1e9)
            ),
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
        """Calculate and print numeric metrics for a named model."""
        results = self.calculate(targets, predictions)

        print(f"\n=== {model_name} test results ===")
        if test_loss is not None:
            print(f"Test loss: {test_loss:.4f}")
        print(f"Test accuracy: {results['accuracy']:.4f}")
        print(f"Macro-Precision: {results['macro_precision']:.4f}")
        print(f"Macro-Recall: {results['macro_recall']:.4f}")
        print(f"Macro-F1: {results['macro_f1']:.4f}")

        if resource_metrics is not None:
            if resource_metrics["train_time_seconds"] is None:
                print("Train time: not measured (evaluation-only)")
            else:
                print(f"Train time: {resource_metrics['train_time_seconds']:.2f}s")
            print(
                "Inference time: "
                f"{resource_metrics['inference_ms_per_sample']:.6f} ms/sample"
            )
            if resource_metrics["flops_b"] is None:
                print("FLOPs: not available")
            else:
                print(f"FLOPs: {resource_metrics['flops_b']:.6f} B/sample")
            print(f"Parameters: {resource_metrics['params_m']:.6f} M")
            print(f"Model size: {resource_metrics['model_size_mb']:.3f} MB")

        print("\nPer-class classification report:")
        print(results["classification_report"])
        print("Confusion matrix (rows=true, columns=predicted):")
        print(results["confusion_matrix"])

        return results
