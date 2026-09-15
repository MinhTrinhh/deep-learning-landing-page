import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from config import SEED

class FashionMNISTDataset(Dataset):
    # This whole class is actually not needed but imma put in it anyway for educational purposes ;)
    def __init__(self, train: bool):
        self.train = train
        self.transform = self.get_base_transform()
        self.dataset = datasets.FashionMNIST(root='./data', train=train, download=True, transform=self.transform)
        self.targets = self.dataset.targets

    def get_base_transform(self):
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        return train_transform
    
    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)
    
    def __len__(self):
        return self.dataset.__len__()
    
def get_dataloaders(batch_size=16, num_workers=1):
    # Dataset creation
    nottest_set = FashionMNISTDataset(True)
    test_set = FashionMNISTDataset(False)

    # Stratification and splitting
    indices = np.arange(nottest_set.__len__())
    targets = np.array(nottest_set.targets)

    train_idx, val_idx = train_test_split(  
        indices,
        test_size=0.1,
        random_state=SEED,
        stratify=targets
    )

    train_set = Subset(nottest_set, train_idx)
    val_set = Subset(nottest_set, val_idx)

    # Dataloader
    train_loader = DataLoader(dataset=train_set(),
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True
                                  )
    val_loader = DataLoader(dataset=val_set,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  pin_memory=True
                                  )
    test_loader = DataLoader(dataset=test_set,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  pin_memory=True
                                  )
    
    return train_loader, val_loader, test_loader
