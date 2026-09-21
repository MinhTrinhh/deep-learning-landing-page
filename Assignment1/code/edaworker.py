import json

from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from config import CLASS_NAMES, DATA_PATH, OUTPUT_PATH, SEED
import torch
import matplotlib.pyplot as plt

"""
How many images are available?
What is the shape and data type of each image?
What range do pixel values have?
What are the class names?
Is the dataset balanced?
Do the images and labels appear correct?
What are the pixel mean and standard deviation?
Which classes look visually similar?
Are there malformed, blank, or suspicious images?
What preprocessing and augmentation make sense?

"""

class EDAWorker():
    def __init__(self, output_dir=OUTPUT_PATH / "eda"):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.train_eval_data = FashionMNIST(
            root=DATA_PATH,
            train=True,
            download=True,
            transform=ToTensor()
        )

        self.test_data = FashionMNIST(
            root=DATA_PATH,
            train=False,
            download=True,
            transform=ToTensor()
        )

    def display_distribution(self):
        counts = torch.bincount(self.train_eval_data.targets)

        for class_id, count in enumerate(counts):
            print(CLASS_NAMES[class_id], count.item())

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(CLASS_NAMES, counts.numpy())
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylabel("Number of images")
        ax.set_title("FashionMNIST training-pool class distribution")
        fig.tight_layout()
        path = self.output_dir / "class_distribution.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return counts

    def show_examples(self):
        fig, axes = plt.subplots(3, 5, figsize=(10, 6))

        generator = torch.Generator().manual_seed(SEED)
        indices = torch.randperm(len(self.train_eval_data), generator=generator)[:15]
        for ax, index in zip(axes.flat, indices):
            image, label = self.train_eval_data[index.item()]
            ax.imshow(image.squeeze(), cmap="gray")
            ax.set_title(CLASS_NAMES[label])
            ax.axis("off")

        plt.tight_layout()
        path = self.output_dir / "random_examples.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return indices

    def calculate_data_statistics(self):
        print(f"Official training samples: {len(self.train_eval_data)}")
        print(f"Official test samples: {len(self.test_data)}")
        print(f"Image tensor shape: {self.train_eval_data[0][0].shape}")

        print(f"Label example: {self.train_eval_data[0][1]}")

        pixels = self.train_eval_data.data.float() / 255.0

        statistics = {
            "training_samples": len(self.train_eval_data),
            "test_samples": len(self.test_data),
            "image_shape": tuple(self.train_eval_data[0][0].shape),
            "minimum": pixels.min().item(),
            "maximum": pixels.max().item(),
            "mean": pixels.mean().item(),
            "standard_deviation": pixels.std().item(),
        }

        print("Minimum:", statistics["minimum"])
        print("Maximum:", statistics["maximum"])
        print("Mean:", statistics["mean"])
        print("Standard deviation:", statistics["standard_deviation"])
        return statistics

    def run(self):
        """Run the reproducible EDA steps and save their figures."""
        print("\n=== Exploratory data analysis ===")
        statistics = self.calculate_data_statistics()
        class_counts = self.display_distribution()
        self.show_examples()
        results = {
            "statistics": statistics,
            "class_counts": class_counts.tolist(),
            "seed": SEED,
        }
        with (self.output_dir / "eda_summary.json").open("w", encoding="utf-8") as file:
            json.dump(results, file, indent=2)
        print(f"EDA figures saved to: {self.output_dir}")
        return results
