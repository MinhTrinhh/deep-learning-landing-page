"""Classification heads trained on cached frozen features.

All baselines share one head design so that the only thing that changes
between them is which modality goes in:
  text   -> MLPHead([d_text])            (unimodal A)
  image  -> MLPHead([d_image])           (unimodal B)
  concat -> MLPHead([d_text, d_image])   (simple feature-level fusion baseline; unlike the
                                          N24News paper there are no auxiliary unimodal heads/losses)

Each input gets its own LayerNorm before concatenation because RoBERTa and
CLIP/ViT features live on very different scales.
"""
from __future__ import annotations

import torch
from torch import nn


class MLPHead(nn.Module):
    def __init__(self, input_dims: list[int], num_classes: int, hidden: int = 512, dropout: float = 0.3):
        super().__init__()
        self.input_dims = list(input_dims)
        self.norms = nn.ModuleList(nn.LayerNorm(d) for d in self.input_dims)
        width = sum(self.input_dims)
        if hidden > 0:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(width, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, num_classes),
            )
        else:  # linear probe
            self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(width, num_classes))

    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        if len(features) != len(self.norms):
            raise ValueError(f"expected {len(self.norms)} inputs, got {len(features)}")
        return self.classifier(torch.cat([norm(x) for norm, x in zip(self.norms, features)], dim=-1))


MODALITIES = {"text": ("text",), "image": ("image",), "concat": ("text", "image")}


def build_model(kind: str, dims: dict[str, int], num_classes: int, hidden: int, dropout: float) -> MLPHead:
    return MLPHead([dims[m] for m in MODALITIES[kind]], num_classes, hidden, dropout)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
