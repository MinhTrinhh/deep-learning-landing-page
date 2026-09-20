import numpy
from sklearn.metrics import accuracy_score
from torch import no_grad, argmax
import matplotlib.pyplot as plt
from config import NUM_EPOCHS

# torch.utils.data.Dataset/Dataloader is already a class
class Trainer():
    def __init__(self):
        pass
    
    def train_model(self, model, train_loader, val_loader, loss_func, optimizer, epochs=NUM_EPOCHS):

        epoch_train_loss = []
        epoch_val_loss = []
        epoch_train_accuracies = []
        epoch_val_accuracies = []

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

        return epoch_train_loss, epoch_train_accuracies, epoch_val_loss, epoch_val_accuracies

    def validate_model(self, model, val_loader, loss_func):

        val_losses, val_preds, val_targets = [], [], []

        model.eval()
        with no_grad():
            for batch_images, batch_labels in val_loader:
                logits = model(batch_images)
                loss = loss_func(logits, batch_labels)

                val_losses.append(loss.item())
                val_preds.extend(argmax(logits, dim=1).cpu().numpy()) #flatten to a single list only
                val_targets.extend(batch_labels.cpu().numpy())

        return numpy.mean(val_losses), accuracy_score(val_targets, val_preds)
    
    def test_model(self, model, test_loader, loss_func):
        mean_test_losses, test_accuracy = self.validate_model(model, test_loader, loss_func)

        print(f"Mean test loss: {mean_test_losses}")
        print(f"Test Accuracy: {test_accuracy}")





    