from pathlib import Path

import matplotlib
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import ConfusionMatrixDisplay

from config import CLASS_NAMES, OUTPUT_PATH


class Plotter:
    def __init__(self, output_dir=OUTPUT_PATH):
        self.output_dir = Path(output_dir)

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
        output_dir = self.output_dir / "learning_curves"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{model_name.lower()}_learning_curves.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"Learning curves saved to: {output_path}")

    def plot_class_distribution(self, class_counts):
        """Plot and save the FashionMNIST class distribution."""
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(CLASS_NAMES, class_counts)
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylabel("Number of images")
        ax.set_title("FashionMNIST training-pool class distribution")
        fig.tight_layout()

        output_dir = self.output_dir / "eda"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "class_distribution.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"Class distribution saved to: {output_path}")

    def plot_examples(self, images, labels):
        """Plot and save a grid of FashionMNIST examples."""
        fig, axes = plt.subplots(3, 5, figsize=(10, 6))

        for ax, image, label in zip(axes.flat, images, labels):
            ax.imshow(image.squeeze(), cmap="gray")
            ax.set_title(CLASS_NAMES[label])
            ax.axis("off")

        fig.tight_layout()
        output_dir = self.output_dir / "eda"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "random_examples.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"Random examples saved to: {output_path}")

    def plot_confusion_matrix(self, confusion_matrix, model_name):
        """Plot and save a confusion matrix for a named model."""
        fig, ax = plt.subplots(figsize=(9, 8))
        display = ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix,
            display_labels=CLASS_NAMES,
        )
        display.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
        ax.set_title(f"{model_name} confusion matrix")
        fig.tight_layout()

        output_dir = self.output_dir / "metrics"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{model_name.lower()}_confusion_matrix.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"Confusion matrix saved to: {output_path}")

    def plot_dimensionality_reduction(self, instance_data):
        labels = np.array(instance_data["labels"])
        methods = [
            ("PCA", instance_data["pca_2d"]),
            ("t-SNE", instance_data["tsne_2d"]),
            ("UMAP", instance_data["umap_2d"]),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        colors = plt.get_cmap("tab10")(np.arange(len(CLASS_NAMES)))

        for ax, (name, coords) in zip(axes, methods):
            coords = np.array(coords)
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=labels,
                cmap="tab10",
                s=5,
                alpha=0.7,
                vmin=0,
                vmax=len(CLASS_NAMES) - 1,
            )
            ax.set_title(f"{name} 2D Projection")
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
        output_dir = self.output_dir / "eda"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "dimensionality_reduction.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"Dimensionality reduction plots saved to: {output_path}")

    def plot_similarity_matrix(self, similarity_matrix):
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            similarity_matrix,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            ax=ax,
        )
        ax.set_title("Class Similarity Matrix (Cosine)")
        fig.tight_layout()

        output_dir = self.output_dir / "eda"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "class_similarity_matrix.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"Class similarity matrix saved to: {output_path}")

    def plot_dendrogram(self, linkage_matrix):
        fig, ax = plt.subplots(figsize=(10, 5))
        dendrogram(
            linkage_matrix,
            labels=CLASS_NAMES,
            leaf_rotation=45,
            leaf_font_size=10,
            ax=ax,
        )
        ax.set_title("Agglomerative Clustering Dendrogram")
        ax.set_xlabel("Classes")
        ax.set_ylabel("Distance")
        fig.tight_layout()

        output_dir = self.output_dir / "eda"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "class_dendrogram.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"Dendrogram saved to: {output_path}")