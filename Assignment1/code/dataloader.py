import random

import numpy
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import FashionMNIST
from torchvision.transforms import Compose, Normalize, RandomAffine, RandomHorizontalFlip, ToTensor

from config import DATA_PATH, VAL_SIZE, SEED, BATCH_SIZE, NUM_WORKERS

class FashionMNISTDataLoader():
    def __init__(self, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, use_augmentation=True):
        self.train_generator = torch.Generator()
        self.train_generator.manual_seed(SEED)

        train_data = FashionMNIST(
            root=DATA_PATH,
            train=True,
            download=True,
            transform=None,
            target_transform=None,
        )

        indices = numpy.arange(len(train_data))
        targets = numpy.array(train_data.targets)

        train_idx, val_idx = train_test_split(indices,
                                              test_size=VAL_SIZE,
                                              random_state=SEED,
                                              stratify=targets)

        train_pixels = train_data.data[train_idx].float() / 255.0
        train_mean = train_pixels.mean().item()
        train_std = train_pixels.std().item()

        val_transform = [ToTensor(), Normalize((train_mean,), (train_std,))]
        train_transform = []
        if use_augmentation:
            train_transform = [
                RandomHorizontalFlip(p=0.5),
                RandomAffine(
                    degrees=(-10, 10),
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1),
                ),
            ]

        self.test_set = FashionMNIST(
            root=DATA_PATH,
            train=False,
            download=True,
            transform=Compose(val_transform),
            target_transform=None,
        )

        self.val_set = FashionMNIST(
            root=DATA_PATH,
            train=True,
            download=True,
            transform=Compose(val_transform),
            target_transform=None,
        )

        train_data.transform = Compose(train_transform + val_transform)
        self.train_set = train_data

        self.train_set = Subset(self.train_set, train_idx)
        self.val_set = Subset(self.val_set, val_idx)

        # Dataloader
        self.train_loader = DataLoader(dataset=self.train_set,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=False,
                                  worker_init_fn=self.seed_worker,
                                  generator=self.train_generator)

        self.val_loader = DataLoader(dataset=self.val_set,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     num_workers=num_workers,
                                     pin_memory=False)

        self.test_loader = DataLoader(dataset=self.test_set,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      num_workers=num_workers,
                                      pin_memory=False)

    def get_train_loader(self):
        return self.train_loader
    
    def get_val_loader(self):
        return self.val_loader
    
    def get_test_loader(self):
        return self.test_loader

    def reset_train_generator(self, seed=SEED):
        """Reset shuffle and worker seeds before training a model."""
        self.train_generator.manual_seed(seed)

    def seed_worker(self, worker_id):
        """Seed Python and NumPy inside each DataLoader worker."""
        del worker_id
        worker_seed = torch.initial_seed() % (2 ** 32)
        random.seed(worker_seed)
        numpy.random.seed(worker_seed)
