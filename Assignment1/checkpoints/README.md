# Assignment 1 Model Checkpoints

The training workflow automatically writes:

- `best_linear_aug.pt`
- `best_mlp_aug.pt`
- `best_cnn_aug.pt`
- `best_gru_aug.pt`
- `best_vit_aug.pt`

The corresponding no-augmentation runs use the `_no_aug.pt` suffix, for
example `best_vit_no_aug.pt`. Checkpoints are created when the matching model
is trained; evaluation-only mode expects the matching checkpoint to exist.

Each file is a PyTorch checkpoint dictionary containing:

- `model_state_dict`: the learned parameter tensors
- `optimizer_state_dict`: optimizer state and parameter groups
- `model_name`: architecture identifier
- `epoch`: best validation epoch
- `val_loss`: validation loss at that epoch
- `val_accuracy`: validation accuracy at that epoch
- `seed`: random seed used for the experiment

The best checkpoint is selected by minimum validation loss and restored before
the model is evaluated on the test set. ViT checkpoint names use the `vit`
model identifier, matching the `python main.py --model vit` command.
All the model checkpoints can be seen in the repo: https://github.com/MinhTrinhh/deep-learning-landing-page/tree/main/Assignment1/checkpoints 
