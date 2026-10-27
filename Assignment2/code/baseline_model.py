
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class BiLSTMSentimentClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 5,
        pad_idx: int = 0,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx,
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, lengths):
        """
        input_ids: (batch_size, sequence_length)
        lengths:   (batch_size,), number of non-padding tokens
        returns:   (batch_size, num_classes) raw logits
        """

        # (B, L) -> (B, L, embedding_dim)
        embedded = self.embedding(input_ids)

        # Ignore padding positions
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        # h_n shape:
        # (num_layers * 2, B, hidden_dim)
        _, (h_n, _) = self.lstm(packed)

        # Final layer's forward and backward hidden states.
        # (B, hidden_dim) each
        forward_hidden = h_n[-2] # The last hidden state of the forward layer
        backward_hidden = h_n[-1] # The last hidden state of the backward layer

        # (B, 2 * hidden_dim)
        representation = torch.cat(
            [forward_hidden, backward_hidden],
            dim=1,
        )

        representation = self.dropout(representation)

        # (B, num_classes)
        logits = self.classifier(representation)

        return logits
