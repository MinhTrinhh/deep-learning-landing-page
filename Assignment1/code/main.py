import models
import dataloader
import trainer
import torch
from config import LEARNING_RATE, NUM_EPOCHS
import plotter

if __name__ == "__main__":

    data_loader = dataloader.FashionMNISTDataLoader()
    model_trainer = trainer.Trainer()
    result_plotter = plotter.Plotter()

    linear_model = models.LinearModel()
    linear_loss_func = torch.nn.CrossEntropyLoss()
    linear_optimizer = torch.optim.AdamW(linear_model.parameters(), lr=LEARNING_RATE)

    epoch_train_loss,\
        epoch_train_accuracies,\
            epoch_val_loss,\
                epoch_val_accuracies = model_trainer.train_model(
        model=linear_model,
        train_loader=data_loader.get_train_loader(),
        val_loader=data_loader.get_val_loader(),
        loss_func=linear_loss_func,
        optimizer=linear_optimizer,
        epochs=NUM_EPOCHS)
    
    model_trainer.test_model(
        model=linear_model,
        test_loader=data_loader.get_test_loader(),
        loss_func=linear_loss_func)

    result_plotter.plot(
        epoch_train_loss,
        epoch_train_accuracies,
        epoch_val_loss,
        epoch_val_accuracies)