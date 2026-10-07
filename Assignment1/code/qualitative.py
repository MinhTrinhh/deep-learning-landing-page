"""Generate deterministic qualitative prediction examples for the report."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

import models
from config import CHECKPOINT_PATH, CLASS_NAMES, DROP_OUT, OUTPUT_PATH
from dataloader import FashionMNISTDataLoader


def load_models():
    model_map = {
        "Linear": models.LinearModel(),
        "MLP": models.MLPModel(dropout=DROP_OUT),
        "CNN": models.CNNModel(dropout=DROP_OUT),
        "GRU": models.GRUModel(dropout=DROP_OUT),
    }
    for name, model in model_map.items():
        checkpoint = torch.load(
            CHECKPOINT_PATH / f"best_{name.lower()}_aug.pt",
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
    return model_map


def collect_predictions(model_map, test_loader):
    targets = []
    predictions = {name: [] for name in model_map}
    with torch.no_grad():
        for images, labels in test_loader:
            targets.extend(labels.tolist())
            for name, model in model_map.items():
                predictions[name].extend(model(images).argmax(dim=1).tolist())
    return np.asarray(targets), {
        name: np.asarray(values) for name, values in predictions.items()
    }


def select_examples(targets, predictions):
    correct = {
        name: values == targets for name, values in predictions.items()
    }
    masks = [
        ("All models correct", np.logical_and.reduce(list(correct.values()))),
        (
            "CNN and GRU recover",
            correct["CNN"] & correct["GRU"] &
            ~(correct["Linear"] & correct["MLP"]),
        ),
        (
            "CNN correct, GRU wrong",
            correct["CNN"] & ~correct["GRU"],
        ),
        ("All models incorrect", ~np.logical_or.reduce(list(correct.values()))),
    ]

    selected = []
    used_classes = set()
    for category, mask in masks:
        candidates = np.flatnonzero(mask)
        chosen = []
        for index in candidates:
            label = int(targets[index])
            if label not in used_classes:
                chosen.append(int(index))
                used_classes.add(label)
            if len(chosen) == 2:
                break
        if len(chosen) < 2:
            for index in candidates:
                if int(index) not in chosen and int(index) not in selected:
                    chosen.append(int(index))
                if len(chosen) == 2:
                    break
        selected.extend((category, index) for index in chosen)
    return selected


def plot_examples(raw_images, targets, predictions, selected):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, (category, index) in zip(axes.flat, selected):
        target_name = CLASS_NAMES[int(targets[index])]
        ax.imshow(raw_images[index], cmap="gray")
        ax.axis("off")

        lines = [f"True: {target_name}"]
        for model_name in ("Linear", "MLP", "CNN", "GRU"):
            predicted = CLASS_NAMES[int(predictions[model_name][index])]
            mark = "correct" if predicted == target_name else "wrong"
            lines.append(f"{model_name}: {predicted} ({mark})")
        ax.set_title(category + "\n" + "\n".join(lines), fontsize=9)

    fig.suptitle(
        "Fashion-MNIST qualitative predictions from the restored best checkpoints",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_dir = Path(OUTPUT_PATH) / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "qualitative_predictions.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Qualitative prediction examples saved to: {output_path}")


if __name__ == "__main__":
    torch.manual_seed(42)
    data = FashionMNISTDataLoader(num_workers=0, use_augmentation=False)
    models_by_name = load_models()
    y_true, y_pred = collect_predictions(models_by_name, data.get_test_loader())
    examples = select_examples(y_true, y_pred)
    plot_examples(data.test_set.data.numpy(), y_true, y_pred, examples)
