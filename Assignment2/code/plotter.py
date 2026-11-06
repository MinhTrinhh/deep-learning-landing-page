import textwrap
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.metrics import (
    ConfusionMatrixDisplay
)
from dataset import (
    LABEL_NAMES
)

CLASS_NAMES = list(LABEL_NAMES)
OUTPUT_PATH = Path("outputs")

class Plotter:
    def __init__(self, output_dir=OUTPUT_PATH):
        self.output_dir = Path(output_dir)

    def _save(self, fig, subdir, filename, dpi=150):
        output_dir = self.output_dir / subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename
        fig.savefig(output_path, dpi=dpi)
        plt.close(fig)
        return output_path

    def plot_learning_curves(
        self,
        epoch_train_loss,
        epoch_train_accuracies,
        epoch_val_loss,
        epoch_val_accuracies,
        model_name,
    ):
        """Plot and save training and validation learning curves."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].plot(epoch_train_loss, label="Train")
        axes[0].plot(epoch_val_loss, label="Validation")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()

        axes[1].plot(epoch_train_accuracies, label="Train")
        axes[1].plot(epoch_val_accuracies, label="Validation")
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()

        fig.tight_layout()
        path = self._save(
            fig, "learning_curves", f"{model_name.lower()}_learning_curves.png"
        )
        print(f"Learning curves saved to: {path}")

    def plot_class_distribution(self, class_counts):
        """Plot and save the training-split class distribution."""
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(CLASS_NAMES, class_counts)
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylabel("Number of tweets")
        ax.set_title("Corona_NLP training-split class distribution")
        fig.tight_layout()
        path = self._save(fig, "eda", "class_distribution.png")
        print(f"Class distribution saved to: {path}")

    def plot_examples(self, texts, labels):
        """Plot and save a 3x5 grid of random example tweets."""
        fig, axes = plt.subplots(3, 5, figsize=(16, 8))
        for ax in axes.flat:
            ax.axis("off")

        for ax, text, label in zip(axes.flat, texts, labels):
            shown = text if len(text) <= 220 else text[:217] + "..."
            ax.text(
                0.5,
                0.5,
                textwrap.fill(shown, 34),
                ha="center",
                va="center",
                fontsize=7,
                transform=ax.transAxes,
            )
            ax.set_title(CLASS_NAMES[label], fontsize=9)

        fig.tight_layout()
        path = self._save(fig, "eda", "random_examples.png")
        print(f"Random examples saved to: {path}")

    def plot_confusion_matrix(self, confusion_matrix, model_name):
        """Plot and save a confusion matrix for a named model."""
        fig, ax = plt.subplots(figsize=(9, 8))
        display = ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix,
            display_labels=CLASS_NAMES,
        )
        display.plot(
            ax=ax,
            cmap="Blues",
            xticks_rotation=45,
            colorbar=False,
            values_format="d",
        )
        ax.set_title(f"{model_name} confusion matrix")
        fig.tight_layout()
        path = self._save(
            fig, "metrics", f"{model_name.lower()}_confusion_matrix.png"
        )
        print(f"Confusion matrix saved to: {path}")

    def plot_dimensionality_reduction(self, instance_data, model_name):
        labels = np.array(instance_data["labels"])
        methods = [
            (name, np.array(instance_data[key]))
            for name, key in (("PCA", "pca_2d"), ("t-SNE", "tsne_2d"), ("UMAP", "umap_2d"))
            if instance_data.get(key) is not None
        ]

        fig, axes = plt.subplots(1, len(methods), figsize=(6.7 * len(methods), 6))
        axes = np.atleast_1d(axes)
        colors = plt.get_cmap("tab10")(np.arange(len(CLASS_NAMES)))

        for ax, (name, coords) in zip(axes, methods):
            # explicit per-point colors so the points always match the legend
            ax.scatter(coords[:, 0], coords[:, 1], c=colors[labels], s=5, alpha=0.7)
            ax.set_title(f"{model_name}: {name} 2D Projection")
            ax.set_xticks([])
            ax.set_yticks([])

        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=color,
                markersize=7,
                label=class_name,
            )
            for class_name, color in zip(CLASS_NAMES, colors)
        ]
        fig.legend(
            handles=legend_handles,
            title="Class",
            loc="center left",
            bbox_to_anchor=(0.89, 0.5),
            frameon=True,
            fontsize=11,
            title_fontsize=12,
            markerscale=1.3,
        )
        fig.tight_layout(rect=(0, 0, 0.88, 1))
        path = self._save(
            fig, "eda", f"{model_name.lower()}_dimensionality_reduction.png"
        )
        print(f"Dimensionality reduction plots saved to: {path}")

    def plot_similarity_matrix(self, similarity_matrix, model_name):
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(
            similarity_matrix,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            ax=ax,
        )
        ax.tick_params(axis="x", rotation=45)
        ax.set_title(f"{model_name} class similarity matrix (cosine)")
        fig.tight_layout()
        path = self._save(
            fig, "eda", f"{model_name.lower()}_class_similarity_matrix.png"
        )
        print(f"Class similarity matrix saved to: {path}")

    def plot_qualitative(self, texts, targets, predictions, selected):
        """Grid of example tweets with every model's prediction."""
        ncols = 4
        nrows = max(1, int(np.ceil(len(selected) / ncols)))
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.5 * nrows))
        axes = np.atleast_1d(axes)
        for ax in axes.flat:
            ax.axis("off")

        for ax, (category, index) in zip(axes.flat, selected):
            target_name = CLASS_NAMES[int(targets[index])]
            lines = [f"True: {target_name}"]
            for model_name, values in predictions.items():
                predicted = CLASS_NAMES[int(values[index])]
                mark = "correct" if predicted == target_name else "wrong"
                lines.append(f"{model_name}: {predicted} ({mark})")
            ax.set_title(category + "\n" + "\n".join(lines), fontsize=7.5)

            text = texts[index]
            shown = text if len(text) <= 300 else text[:297] + "..."
            ax.text(
                0.5,
                0.5,
                textwrap.fill(shown, 42),
                ha="center",
                va="center",
                fontsize=8,
                transform=ax.transAxes,
            )

        fig.suptitle(
            "Corona_NLP qualitative predictions from restored best checkpoints",
            fontsize=14,
            fontweight="bold",
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        path = self._save(fig, "metrics", "qualitative_predictions.png", dpi=180)
        print(f"Qualitative prediction examples saved to: {path}")
