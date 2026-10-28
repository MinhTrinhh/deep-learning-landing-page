import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, DataCollatorWithPadding

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

class CoronaNLPTweetDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer, max_length: int = 128):
        self.texts = dataframe["OriginalTweet"].fillna("").astype(str).tolist()
        self.labels = dataframe["Sentiment"].map(LABEL2ID).tolist()
        
        if any(label is None or pd.isna(label) for label in self.labels):
            raise ValueError("Unknown or missing sentiment label found.")
            
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int):
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            max_length=self.max_length,
            padding=False, # Padding is handled dynamically by DataCollatorWithPadding
        )
        encoded["labels"] = int(self.labels[index])
        return encoded

def get_dataloaders(
    tokenizer_name: str = "distilbert/distilbert-base-uncased",
    train_path: str = "dataset/Corona_NLP_train.csv",
    test_path: str = "dataset/Corona_NLP_test.csv",
    batch_size: int = 64,
    eval_batch_size: int = 128,
    val_size: float = 0.10,
    random_state: int = 42,
    max_length: int = 128,
    num_workers: int = 0,
):
    train_df = pd.read_csv(train_path, encoding="utf-8", encoding_errors="replace")
    test_df = pd.read_csv(test_path, encoding="utf-8", encoding_errors="replace")

    train_part, val_part = train_test_split(
        train_df, test_size=val_size, random_state=random_state, stratify=train_df["Sentiment"]
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataset = CoronaNLPTweetDataset(train_part, tokenizer, max_length)
    val_dataset = CoronaNLPTweetDataset(val_part, tokenizer, max_length)
    test_dataset = CoronaNLPTweetDataset(test_df, tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=collator, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=collator, num_workers=num_workers)

    return train_loader, val_loader, test_loader, tokenizer, max_length