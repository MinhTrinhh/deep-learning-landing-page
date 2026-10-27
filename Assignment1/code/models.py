import torch
from torch import nn
from config import (
    INPUT_DIM,
    OUTPUT_DIM,
)

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


class GRUModel(nn.Module):
    def __init__(self,dropout):
        super().__init__()

        self.gru = nn.GRU(
            input_size=28,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
        ) # (B, 28, 28) -> (B, 28, 128) (1, B, 128)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(128, OUTPUT_DIM) # (B, 128) -> (B, 10)

    def forward(self, images):
        sequences = images.squeeze(1) # (B, 1, 28, 28) -> (B, 28, 28)
        _, hidden = self.gru(sequences) # (B, 28, 128), (1, B, 128)

        final_hidden = self.dropout(hidden[-1, :, :]) # (B, 128)
        logits = self.classifier(final_hidden) # (B, 10)

        return logits


class ViTransformer(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        image_size = 28
        patch_size = 4
        embedding_dim = 128
        num_patches = (image_size // patch_size) ** 2

        self.patch_embedding = nn.Conv2d(
            in_channels=1,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
        ) #(B, 1, 28, 28) -> (B, 128, 7, 7)
        self.class_token = nn.Parameter(torch.zeros(1, 1, embedding_dim)) #(1, 1, 128) will be added to the input later
        self.position_embedding = nn.Parameter(
            torch.zeros(1, num_patches + 1, embedding_dim)
        ) # (B, 50, 128) gives information about position to model
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        
        self.embedding_dropout = nn.Dropout(p=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, # get in as 128 get out as 128
            nhead=4, # MHA
            dim_feedforward=embedding_dim * 4, # FFN
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True, # prenorm
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.norm = nn.LayerNorm(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, OUTPUT_DIM)

    def forward(self, images):
        # (B, 1, 28, 28) -> (B, 128, 7, 7) -> (B, 128, 49) -> (B, 49, 128)
        patches = self.patch_embedding(images).flatten(2).transpose(1, 2)
        class_tokens = self.class_token.expand(images.shape[0], -1, -1)  # (1, 1, 128) -> (B, 1, 128)
        tokens = torch.cat((class_tokens, patches), dim=1)  # (B, 50, 128)
        tokens = self.embedding_dropout(tokens + self.position_embedding)
        encoded = self.encoder(tokens)
        image_representation = self.norm(encoded[:, 0])
        return self.classifier(image_representation) # logits
