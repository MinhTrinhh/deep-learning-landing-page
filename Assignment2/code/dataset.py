
# dataset.py

import re
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Constants
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

PAD_IDX = 0
UNK_IDX = 1

LABEL_NAMES = [
    "Extremely Negative",
    "Negative",
    "Neutral",
    "Positive",
    "Extremely Positive",
]

LABEL2ID = {
    label: idx for idx, label in enumerate(LABEL_NAMES)
}


# Tokenizer and vocabulary
def tokenize(text: str) -> list[str]:
    """
    Basic tokenizer for the BiLSTM baseline.
    Preserves URLs, mentions, hashtags, words, and punctuation.
    """
    # #
    # Regex explanation:
    # - https?://\S+ matches URLs
    # - @\w+ matches usernames
    # - #\w+ matches hashtags
    # - [\w']+ matches words with punctuations
    # - [^\w\s]" matches single characters that are not words nor whitespace
    # #
    return re.findall(
        r"https?://\S+|@\w+|#\w+|[\w']+|[^\w\s]",
        str(text).lower(),
    )


def build_vocab(texts, min_freq: int = 2) -> dict[str, int]:
    """
    Build vocabulary from training texts only.
    """
    counter = Counter()

    # Counts the number of occurences of each token
    for text in texts:
        counter.update(tokenize(text))

    vocab = {
        PAD_TOKEN: PAD_IDX,
        UNK_TOKEN: UNK_IDX,
    }

    for token, count in counter.items():
        if count >= min_freq and token not in vocab:
            vocab[token] = len(vocab) # Increment Token ID

    return vocab


def encode_text(
    text: str,
    vocab: dict[str, int],
    max_length: int,
) -> tuple[list[int], int]:
    """
    Convert text into token IDs, truncate if necessary,
    and pad to max_length.

    Returns:
        token_ids: list[int], fixed length
        length: number of actual tokens before padding
    """
    tokens = tokenize(text)[:max_length]

    token_ids = [
        vocab.get(token, UNK_IDX)
        for token in tokens
    ]

    # pack_padded_sequence requires lengths >= 1.
    if not token_ids:
        token_ids = [UNK_IDX]

    length = len(token_ids)

    token_ids += [PAD_IDX] * (max_length - length) # Padding to max length

    return token_ids, length


# Dataset
class CoronaNLPTweetDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        vocab: dict[str, int],
        max_length: int,
    ):
        self.texts = (
            dataframe["OriginalTweet"]
            .fillna("")
            .astype(str)
            .tolist()
        )

        # Convert string labels into integer class IDs.
        self.labels = (
            dataframe["Sentiment"]
            .map(LABEL2ID)
            .tolist()
        )

        if any(label is None or pd.isna(label) for label in self.labels):
            raise ValueError("Unknown or missing sentiment label found.")

        self.vocab = vocab
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int):
        token_ids, length = encode_text(
            self.texts[index],
            self.vocab,
            self.max_length,
        )

        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(length, dtype=torch.long),
            torch.tensor(self.labels[index], dtype=torch.long),
        )

# Dataloader
def get_dataloaders(
    train_path: str = "Corona_NLP_train.csv",
    test_path: str = "Corona_NLP_test.csv",
    batch_size: int = 64,
    eval_batch_size: int = 128,
    val_size: float = 0.10,
    random_state: int = 42,
    min_freq: int = 2,
    max_length: int | None = None,
    num_workers: int = 0,
):
    """
    Returns:
        train_loader
        val_loader
        test_loader
        vocab
        max_length

    The supplied test set is kept separate from model selection.
    Vocabulary and automatic max_length are derived from the
    training subset only.
    """

    # Load files.
    train_df = pd.read_csv(train_path, encoding="utf-8", encoding_errors="replace",)
    test_df = pd.read_csv(test_path, encoding="utf-8", encoding_errors="replace",)

    # Validate labels.
    unknown_train = set(train_df["Sentiment"].dropna()) - set(LABEL2ID)
    unknown_test = set(test_df["Sentiment"].dropna()) - set(LABEL2ID)

    if unknown_train or unknown_test:
        raise ValueError(
            f"Unknown labels. Train={unknown_train}, Test={unknown_test}"
        )

    # Stratified split preserves class proportions (90/10)
    train_part, val_part = train_test_split(
        train_df,
        test_size=val_size,
        random_state=random_state,
        stratify=train_df["Sentiment"],
    )

    # Build vocabulary using training texts only.
    vocab = build_vocab(
        train_part["OriginalTweet"],
        min_freq=min_freq,
    )

    # Derive max length from training data only if unspecified.
    if max_length is None:
        token_lengths = train_part["OriginalTweet"].map(
            lambda text: len(tokenize(text))
        )

        # Cover approximately 99% of training tweets.
        max_length = max(
            1,
            int(token_lengths.quantile(0.99)),
        )

    if max_length < 1:
        raise ValueError("max_length must be at least 1.")

    # Construct datasets.
    train_dataset = CoronaNLPTweetDataset(
        train_part, vocab, max_length
    )
    val_dataset = CoronaNLPTweetDataset(
        val_part, vocab, max_length
    )
    test_dataset = CoronaNLPTweetDataset(
        test_df, vocab, max_length
    )

    # Construct loaders.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    print(f"Training samples:   {len(train_dataset):,}")
    print(f"Validation samples: {len(val_dataset):,}")
    print(f"Test samples:       {len(test_dataset):,}")
    print(f"Vocabulary size:    {len(vocab):,}")
    print(f"Maximum length:     {max_length}")

    return train_loader, val_loader, test_loader, vocab, max_length
