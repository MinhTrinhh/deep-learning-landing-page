from dataloader import get_dataloaders
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import models
import torch
import numpy as np

if __name__ == "__main__":
    # Dataloaders
    train_loader, val_loader, test_loader = get_dataloaders()

    # Model
    model = models.LinearModel()
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    num_epoch = 5

    # Train and validate
    epoch_train_loss = []
    epoch_val_loss = []
    epoch_train_accuracies = []
    epoch_val_accuracies = []
    for epoch in range(num_epoch):
        # Training
        model.train()
        train_y_true = []
        train_y_pred = []
        train_losses = []

        for inputs, labels in train_loader:
            optimizer.zero_grad()

            # Forward pass
            out = model(inputs)
            loss = loss_func(out, labels) # out shape is (batch, 10) while label is (batch, 1)
            
            train_losses.append(loss.item())

            # Backward pass
            loss.backward()
            optimizer.step()

            # Accuracy
            train_y_pred.extend(torch.argmax(out, dim=1).numpy())
            train_y_true.extend(labels.numpy())

        # Validating
        model.eval()
        val_y_true = []
        val_y_pred = []
        val_losses = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                # Forward pass
                out = model(inputs)
                loss = loss_func(out, labels)
                
                val_losses.append(loss.item())

                # Accuracy
                val_y_pred.extend(torch.argmax(out, dim=1).numpy())
                val_y_true.extend(labels.numpy())

        mean_train_loss = np.mean(train_losses)
        mean_train_acc = accuracy_score(train_y_true, train_y_pred)
        mean_val_loss = np.mean(val_losses)
        mean_val_acc = accuracy_score(val_y_true, val_y_pred)

        epoch_train_loss.append(mean_train_loss)
        epoch_val_loss.append(mean_val_loss)
        epoch_train_accuracies.append(mean_train_acc)
        epoch_val_accuracies.append(mean_val_acc)

        print("Epoch [%d/%d] \nTrain loss: %.5f | Train Accuracy:%.5f\nVal loss: %.5f | Val Accuracy: %.5f" % (epoch + 1, num_epoch, mean_train_loss, mean_train_acc, mean_val_loss, mean_val_acc))

    # Testing
    model.eval()
    test_losses = []
    test_y_true = []
    test_y_pred = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            # Forward pass
            out = model(inputs)
            loss = loss_func(out, labels)
            test_losses.append(loss.item())
            test_y_true.extend(labels.numpy())
            test_y_pred.extend(torch.argmax(out, dim=1))

    print("Final test loss: %.5f\nFinal test accuracy: %.5f" % (np.mean(test_losses), accuracy_score(test_y_true, test_y_pred)))

    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].plot(epoch_train_loss, label='Train')
    ax[0].plot(epoch_val_loss, label='Validation')
    ax[0].set_title('Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].legend()

    ax[1].plot(epoch_train_accuracies, label='Train')
    ax[1].plot(epoch_val_accuracies, label='Validation')
    ax[1].set_title('Accuracy')
    ax[1].set_xlabel('Epoch')
    ax[1].legend()

    plt.tight_layout()
    plt.show()