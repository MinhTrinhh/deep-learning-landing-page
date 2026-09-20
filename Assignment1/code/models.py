import torch
from torch import nn
from config import INPUT_DIM, OUTPUT_DIM

# Linear model
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(INPUT_DIM, OUTPUT_DIM) # Since images are (1, 28, 28) flattening would return a 1 x 28*28 vector

    def forward(self, images): # receive a batch of flatten image
        vectors = torch.flatten(images, start_dim=1, end_dim=-1) # init shape is (batch, 1, h, w) -> (batch, h*w)
        return self.fc(vectors)
    
# MLP model
class MLPModel(nn.Module):
    def __init__(self, dropout=0.02):
        super().__init__()
        self.fc1 = nn.Linear(784, 1024)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, images):
        vectors = torch.flatten(images, 1)
        out = self.fc1(vectors)
        out = self.relu(out)
        self.dropout = self.dropout(out)
        out = self.fc2(out)
        return out