import json
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
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

    def report(self, targets, predictions, model_name, test_loss=None):
        """Print metrics and save a confusion matrix for a named model."""
        results = self.calculate(targets, predictions)

        print(f"\n=== {model_name} test results ===")
        if test_loss is not None:
            print(f"Test loss: {test_loss:.4f}")
        print(f"Test accuracy: {results['accuracy']:.4f}")
        print(f"Macro-F1: {results['macro_f1']:.4f}")
        print(results["classification_report"])

        fig, ax = plt.subplots(figsize=(9, 8))
        display = ConfusionMatrixDisplay(
            confusion_matrix=results["confusion_matrix"],
            display_labels=CLASS_NAMES,
        )
        display.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
        ax.set_title(f"{model_name} confusion matrix")
        fig.tight_layout()
        output_path = self.output_dir / f"{model_name.lower()}_confusion_matrix.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"Confusion matrix saved to: {output_path}")

        serializable_results = {
            "test_loss": float(test_loss) if test_loss is not None else None,
            "accuracy": float(results["accuracy"]),
            "macro_f1": float(results["macro_f1"]),
            "confusion_matrix": results["confusion_matrix"].tolist(),
            "classification_report": results["classification_report"],
        }
        metrics_path = self.output_dir / f"{model_name.lower()}_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as file:
            json.dump(serializable_results, file, indent=2)
        print(f"Numeric metrics saved to: {metrics_path}")

        return results
