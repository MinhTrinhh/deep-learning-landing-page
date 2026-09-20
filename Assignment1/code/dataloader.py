from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import FashionMNIST
from config import MEAN_TUP, SD_TUP, DATA_PATH, VAL_SIZE, SEED, BATCH_SIZE, NUM_WORKERS
from sklearn.model_selection import train_test_split
import numpy
from torch.utils.data import Subset, DataLoader

class FashionMNISTDataLoader():
    def __init__(self, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):

        def dataset_transform():
            return Compose([
                ToTensor(),
                Normalize(MEAN_TUP, SD_TUP)
            ])

        self.test_set = FashionMNIST(root=DATA_PATH,
                                     train=False,
                                     download=True,
                                     transform=dataset_transform(),
                                     target_transform=None)

        nottest_set = FashionMNIST(root=DATA_PATH,
                                   train=True,
                                   download=True,
                                   transform=dataset_transform(),
                                   target_transform=None)

        indices = numpy.arange(len(nottest_set))
        targets = numpy.array(nottest_set.targets)

        train_idx, val_idx = train_test_split(indices,
                                              test_size=VAL_SIZE,
                                              random_state=SEED,
                                              stratify=targets)

        self.train_set = Subset(nottest_set, train_idx)
        self.val_set = Subset(nottest_set, val_idx)

        # Dataloader
        self.train_loader = DataLoader(dataset=self.train_set,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=False)

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