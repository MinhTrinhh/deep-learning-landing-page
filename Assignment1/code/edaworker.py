import json

import torch
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

from config import CLASS_NAMES, DATA_PATH, OUTPUT_PATH, SEED

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

    def calculate_class_distribution(self):
        counts = torch.bincount(self.train_eval_data.targets)

        for class_id, count in enumerate(counts):
            print(CLASS_NAMES[class_id], count.item())

        return counts

    def select_examples(self):
        generator = torch.Generator().manual_seed(SEED)
        indices = torch.randperm(len(self.train_eval_data), generator=generator)[:15]
        images = []
        labels = []
        for index in indices:
            image, label = self.train_eval_data[index.item()]
            images.append(image)
            labels.append(label)
        return indices, images, labels

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
        """Calculate reproducible EDA results and save their numeric summary."""
        print("\n=== Exploratory data analysis ===")
        statistics = self.calculate_data_statistics()
        class_counts = self.calculate_class_distribution()
        example_indices, example_images, example_labels = self.select_examples()
        summary = {
            "statistics": statistics,
            "class_counts": class_counts.tolist(),
            "example_indices": example_indices.tolist(),
            "seed": SEED,
        }
        with (self.output_dir / "eda_summary.json").open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)
        print(f"EDA summary saved to: {self.output_dir / 'eda_summary.json'}")
        return {
            **summary,
            "example_images": example_images,
            "example_labels": example_labels,
        }
