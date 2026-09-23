# Assignment 1 Model Checkpoints

The training workflow automatically writes:

- `best_linear.pt`
- `best_mlp.pt`

Each file is a PyTorch checkpoint dictionary containing:

- `model_state_dict`: the learned parameter tensors
- `optimizer_state_dict`: optimizer state and parameter groups
- `model_name`: architecture identifier
- `epoch`: best validation epoch
- `val_loss`: validation loss at that epoch
- `val_accuracy`: validation accuracy at that epoch
- `seed`: random seed used for the experiment

The best checkpoint is selected by minimum validation loss and restored before
the model is evaluated on the test set.
