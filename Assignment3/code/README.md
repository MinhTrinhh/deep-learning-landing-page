# Assignment 3 Codebase

Place all PyTorch code for Assignment 3 multimodal deep learning here:
- `dataset.py`: Paired multimodal dataset loader (visual, text, depth, etc.)
- `encoders/`: Modality A & Modality B feature encoders
- `fusion/`: Early, intermediate, and late fusion module implementations
- `train.py`: Multimodal training pipeline with per-modality and joint losses
- `evaluate.py`: Multimodal evaluation (Recall@K, CIDEr/BLEU, accuracy, modality support vs conflict analysis)
