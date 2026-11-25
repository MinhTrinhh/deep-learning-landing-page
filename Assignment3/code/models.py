"""Classification head trained on cached frozen features.

All baselines share one head so that the only thing that changes is which modality goes in:
  text   -> MLPHead([768])        (unimodal A, RoBERTa)
  image  -> MLPHead([768])        (unimodal B, ViT)
  concat -> MLPHead([768, 768])   (simple feature-level fusion; no auxiliary unimodal heads/losses)

Each input gets its own LayerNorm before concatenation because RoBERTa and
ViT features live on very different scales.
"""
import torch
from torch import nn

MODALITIES = {"text": ("text",), "image": ("image",), "concat": ("text", "image")}


class MLPHead(nn.Module):
    def __init__(self, input_dims, num_classes, hidden=512, dropout=0.3):
        super().__init__()
        self.input_dims = list(input_dims)
        self.norms = nn.ModuleList(nn.LayerNorm(d) for d in self.input_dims)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(sum(self.input_dims), hidden),  # 768 (or 1536 for concat) -> 512
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),           # 512 -> 24
        )

    def forward(self, *features):
        x = torch.cat([norm(f) for norm, f in zip(self.norms, features)], dim=-1)
        return self.classifier(x)


def build_model(kind, dims, num_classes, hidden, dropout):
    return MLPHead([dims[m] for m in MODALITIES[kind]], num_classes, hidden, dropout)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
