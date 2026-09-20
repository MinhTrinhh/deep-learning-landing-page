import matplotlib.pyplot as plt

class Plotter():
    def __init__(self):
        pass
    def plot(self,
             epoch_train_loss,
             epoch_train_accuracies,
             epoch_val_loss,
             epoch_val_accuracies):       
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