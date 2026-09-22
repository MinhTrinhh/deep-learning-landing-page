import json

import torch
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage

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

    def calculate_instance_level_features(self, num_samples=2000):
        generator = torch.Generator().manual_seed(SEED)
        indices = torch.randperm(len(self.train_eval_data), generator=generator)[:num_samples]
        
        images, labels = [], []
        for idx in indices:
            img, lbl = self.train_eval_data[idx.item()]
            images.append(img.view(-1).numpy()) 
            labels.append(lbl)
            
        X = np.array(images)
        y = np.array(labels)

        pca = PCA(n_components=2, random_state=SEED)
        pca_result = pca.fit_transform(X)

        tsne = TSNE(n_components=2, random_state=SEED, init='pca', learning_rate='auto')
        tsne_result = tsne.fit_transform(X)

        umap_reducer = umap.UMAP(n_components=2, random_state=SEED)
        umap_result = umap_reducer.fit_transform(X)

        return {
            "labels": y.tolist(),
            "pca_2d": pca_result.tolist(),
            "tsne_2d": tsne_result.tolist(),
            "umap_2d": umap_result.tolist(),
        }

    def calculate_class_level_similarity(self):
        class_sums = {i: 0 for i in range(len(CLASS_NAMES))}
        class_counts = {i: 0 for i in range(len(CLASS_NAMES))}
        
        for img, label in self.train_eval_data:
            flat_img = img.view(-1).numpy()
            class_sums[label] = class_sums[label] + flat_img
            class_counts[label] += 1
            
        class_means = [class_sums[i] / class_counts[i] for i in range(len(CLASS_NAMES))]
        class_means = np.array(class_means)
        
        sim_matrix = cosine_similarity(class_means)
        
        sim_matrix_masked = sim_matrix.copy()
        np.fill_diagonal(sim_matrix_masked, -1) 
        top_indices = np.argsort(sim_matrix_masked, axis=None)[::-1]
        
        top_pairs = []
        seen_pairs = set()
        
        for idx in top_indices:
            r, c = divmod(idx, len(CLASS_NAMES))
            if r != c and frozenset([r, c]) not in seen_pairs:
                seen_pairs.add(frozenset([r, c]))
                top_pairs.append({
                    "class_1": CLASS_NAMES[r],
                    "class_2": CLASS_NAMES[c],
                    "similarity_score": float(sim_matrix[r, c])
                })
                if len(top_pairs) == 5:
                    break
                    
        dist_array = pdist(class_means, metric='cosine')
        linkage_matrix = linkage(dist_array, method='average')

        return {
            "similarity_matrix": sim_matrix.tolist(),
            "top_confusing_pairs": top_pairs,
            "linkage_matrix": linkage_matrix.tolist()
        }

    def run(self):
        """Calculate reproducible EDA results and save their numeric summary."""
        print("\n=== Exploratory data analysis ===")
        statistics = self.calculate_data_statistics()
        class_counts = self.calculate_class_distribution()
        example_indices, example_images, example_labels = self.select_examples()
        instance_level_data = self.calculate_instance_level_features(num_samples=2000)
        class_level_data = self.calculate_class_level_similarity()

        summary = {
            "seed": SEED,
            "core_eda": {
                "statistics": statistics,
                "class_counts": class_counts.tolist(),
                "example_indices": example_indices.tolist(),
            },
            "task_specific_eda": {
                "instance_level": instance_level_data,
                "class_level": {
                    "similarity_matrix": class_level_data["similarity_matrix"],
                    "top_confusing_pairs": class_level_data["top_confusing_pairs"],
                    "linkage_matrix": class_level_data["linkage_matrix"]
                }
            }
        }
        
        with (self.output_dir / "eda_summary.json").open("w",encoding="utf-8") as file:
            json.dump(summary, file, indent=2)
        print(f"EDA summary saved to: {self.output_dir / 'eda_summary.json'}")
        return {
            **summary,
            "example_images": example_images,
            "example_labels": example_labels,
        }
