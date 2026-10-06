import torch
from torch import nn
from config import INPUT_DIM, OUTPUT_DIM, DROP_STEP

# Linear model
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(in_features=INPUT_DIM, out_features=OUTPUT_DIM) # Since images are (1, 28, 28) flattening would return a 1 x 28*28 vector

    def forward(self, images): # receive a batch of flatten image
        vectors = torch.flatten(images, start_dim=1, end_dim=-1) # init shape is (batch, 1, h, w) -> (batch, h*w)
        return self.fc(vectors)
    
# MLP model
class MLPModel(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, 512)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(p=dropout)

        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(p=dropout)

        self.fc3 = nn.Linear(256, 128)
        self.relu3 = nn.ReLU()
        self.drop3 = nn.Dropout(p=dropout)

        self.fc4 = nn.Linear(128, OUTPUT_DIM)

    def forward(self, images):
        vectors = torch.flatten(images, start_dim=1, end_dim = -1)

        out = self.fc1(vectors)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.fc2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        out = self.fc3(out)
        out = self.relu3(out)
        out = self.drop3(out)

        out = self.fc4(out)

        return out

class CNNModel(nn.Module):
    def __init__(self, dropout):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1, stride=1), # (B, 1, 28, 28) -> (B, 32, 28, 28)
            nn.ReLU(), # (B, 32, 28, 28)
            nn.MaxPool2d(kernel_size=2), # (B, 32, 14, 14)

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1), # (B, 32, 14, 14) -> (B, 64, 14, 14)
            nn.ReLU(), # (B, 64, 14, 14)
            nn.MaxPool2d(kernel_size=2), # (B, 64, 7, 7)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, OUTPUT_DIM)
        )

    def forward(self, images):
        features = self.features(images)
        logits = self.classifier(features)

        return logits
