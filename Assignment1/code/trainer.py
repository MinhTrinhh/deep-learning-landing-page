import pickle
import time

import numpy
import torch
from sklearn.metrics import accuracy_score
from torch import no_grad, argmax
from config import CHECKPOINT_PATH, NUM_EPOCHS, SEED, MAX_ATTEMPT

# torch.utils.data.Dataset/Dataloader is already a class
class Trainer():
    def __init__(self, checkpoint_dir=CHECKPOINT_PATH):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.last_train_time = None
    
    def train_model(
        self,
        model,
        train_loader,
        val_loader,
        loss_func,
        optimizer,
        epochs=NUM_EPOCHS,
        model_name=None,
    ):

        epoch_train_loss = []
        epoch_val_loss = []
        epoch_train_accuracies = []
        epoch_val_accuracies = []
        model_name = model_name or model.__class__.__name__
        checkpoint_path = self.checkpoint_dir / f"best_{model_name.lower()}.pt"
        best_val_loss = float("inf")
        train_start = time.perf_counter()

        ticking_clock = MAX_ATTEMPT

        for epoch in range(epochs):
            train_losses, train_preds, train_targets = [], [], []
            model.train()

            for batch_images, batch_labels in train_loader:
                logits = model(batch_images) # use internal mechanism and finally trigger the model.forward()
                loss = loss_func(logits, batch_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())
                train_preds.extend(argmax(logits, dim=1).cpu().numpy())
                train_targets.extend(batch_labels.cpu().numpy())


            print(f"Epoch {epoch + 1}/{epochs}:")
            print(f"Mean train loss: {numpy.mean(train_losses)}")
            print(f"Train Accuracy: {accuracy_score(train_targets, train_preds)}")

            # check the correlation between val and train after each epoch
            mean_val_losses, val_accuracy = self.validate_model(model,
                                                                val_loader,
                                                                loss_func)

            print(f"Mean validate loss:  {mean_val_losses}")
            print(f"Validate Accuracy: {val_accuracy}")

            epoch_train_loss.append(numpy.mean(train_losses))
            epoch_val_loss.append(mean_val_losses)
            epoch_train_accuracies.append(accuracy_score(train_targets, train_preds))
            epoch_val_accuracies.append(val_accuracy)

            if mean_val_losses < best_val_loss:
                best_val_loss = mean_val_losses
                self.save_checkpoint(
                    checkpoint_path=checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    model_name=model_name,
                    epoch=epoch + 1,
                    val_loss=mean_val_losses,
                    val_accuracy=val_accuracy,
                )
                print(f"Saved new best checkpoint: {checkpoint_path}")
                ticking_clock = MAX_ATTEMPT
            else:
                ticking_clock = ticking_clock - 1
                if ticking_clock == 0:
                    break

        checkpoint = self.load_checkpoint(checkpoint_path, model)
        self.last_train_time = time.perf_counter() - train_start
        print(
            f"Restored best {model_name} checkpoint from epoch "
            f"{checkpoint['epoch']} "
            f"(validation loss={checkpoint['val_loss']:.4f})"
        )

        return epoch_train_loss, epoch_train_accuracies, epoch_val_loss, epoch_val_accuracies

    def save_checkpoint(
        self,
        checkpoint_path,
        model,
        optimizer,
        model_name,
        epoch,
        val_loss,
        val_accuracy,
    ):
        """Store the best model parameters and training metadata."""
        model_state = model.state_dict()
        torch.save(
            {
                "model_name": model_name,
                "epoch": int(epoch),
                "val_loss": float(val_loss),
                "val_accuracy": float(val_accuracy),
                "model_state_dict": model_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "seed": SEED,
            },
            checkpoint_path,
        )

    def load_checkpoint(self, checkpoint_path, model, optimizer=None):
        """Load model parameters and optionally optimizer state for resuming."""
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint

    def validate_model(self, model, val_loader, loss_func, return_predictions=False):

        val_losses, val_preds, val_targets = [], [], []

        model.eval()
        with no_grad():
            for batch_images, batch_labels in val_loader:
                logits = model(batch_images)
                loss = loss_func(logits, batch_labels)

                val_losses.append(loss.item())
                val_preds.extend(argmax(logits, dim=1).cpu().numpy()) #flatten to a single list only
                val_targets.extend(batch_labels.cpu().numpy())

        mean_loss = numpy.mean(val_losses)
        accuracy = accuracy_score(val_targets, val_preds)
        if return_predictions:
            return mean_loss, val_targets, val_preds
        return mean_loss, accuracy
    
    def test_model(self, model, test_loader, loss_func):
        return self.validate_model(
            model,
            test_loader,
            loss_func,
            return_predictions=True,
        )
