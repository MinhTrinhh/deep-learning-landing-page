from dataloader import get_dataloaders
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
    num_epoch = 20

    # Train and validate
    val_losses = []
    for epoch in range(num_epoch):
        # Training
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()

            # Forward pass
            out = model(inputs)
            loss = loss_func(out, labels) # out and labels shape are (batch, 10)

            # Backward pass
            loss.backward()
            optimizer.step()

        # Validating
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                # Forward pass
                out = model(inputs)
                loss = loss_func(out, labels)
                val_losses.append(loss.item())

        print("Epoch [%d/%d] - Val loss: %.5f" % (epoch + 1, num_epoch, np.mean(val_losses)))
        val_losses.clear()

    # Testing
    model.eval()
    test_losses = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Forward pass
            out = model(inputs)
            loss = loss_func(out, labels)
            test_losses.append(loss.item())

    print("Final test loss: %.5f" % (np.mean(test_losses),))
