# Assignment 3 Checkpoints

Trained heads for Assignment 3 (copied here by `code/evaluate.py --checkpoints checkpoints`, best-val seed of each selected run):
- `unimodal_a_baseline.pt`: text only (RoBERTa features → MLP head)
- `unimodal_b_baseline.pt`: image only (ViT-B/16 ImageNet features → MLP head)
- `simple_fusion_model.pt`: concatenation fusion (text + image features → MLP head)
- `improved_multimodal_model.pt`: cross-attention + gated fusion (later milestone)

Each file stores `state_dict` plus the settings needed to rebuild the head (`input_dims`, `hidden`, `classes`, feature names).
They are `.pt` because the repo `.gitignore` ignores `*.pth`.
