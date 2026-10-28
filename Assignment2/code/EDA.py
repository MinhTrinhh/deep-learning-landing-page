
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. Configuration
# ============================================================

DATA_DIR = Path("dataset")
OUTPUT_DIR = Path("corona_eda")
OUTPUT_DIR.mkdir(exist_ok=True)

TRAIN_PATH = DATA_DIR / "Corona_NLP_train.csv"
TEST_PATH = DATA_DIR / "Corona_NLP_test.csv"

TEXT_COL = "OriginalTweet"
TARGET_COL = "Sentiment"
DATE_COL = "TweetAt"

EXPECTED_LABELS = [
    "Extremely Negative",
    "Negative",
    "Neutral",
    "Positive",
    "Extremely Positive",
]

sns.set_theme(style="whitegrid")


# ============================================================
# 2. Load files and inspect schemas
# ============================================================

def load_csv(path):
    if not path.exists():
        print(f"[MISSING FILE] {path.resolve()}")
        return None

    # utf-8-sig handles a possible UTF-8 BOM.
    # Fall back to a common alternative encoding if needed.
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")

    print(f"\n{'=' * 65}")
    print(f"FILE: {path.name}")
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print("\nColumns and dtypes:")
    print(df.dtypes)
    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))
    print("\nMissing values:")
    print(df.isna().sum().to_string())

    return df


train = load_csv(TRAIN_PATH)
test = load_csv(TEST_PATH)

if train is None or test is None:
    raise FileNotFoundError(
        "Place BOTH Corona_NLP_train.csv and Corona_NLP_test.csv "
        "in the script's folder, then run it again."
    )

required = {TEXT_COL, TARGET_COL, DATE_COL}
for name, df in [("train", train), ("test", test)]:
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"{name} is missing required columns: {sorted(missing_cols)}"
        )


# ============================================================
# 3. Dataset sizes, missingness, and duplicates
# ============================================================

def normalized_text(series):
    """Normalize text for duplicate comparison only; do not overwrite input."""
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def basic_report(df, name):
    report = {
        "split": name,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_text": int(df[TEXT_COL].isna().sum()),
        "missing_target": int(df[TARGET_COL].isna().sum()),
        "missing_location": (
            int(df["Location"].isna().sum())
            if "Location" in df.columns else None
        ),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "duplicate_text_rows": int(
            normalized_text(df[TEXT_COL]).duplicated(keep="first").sum()
        ),
        "unique_texts": int(normalized_text(df[TEXT_COL]).nunique()),
        "unique_labels": int(df[TARGET_COL].nunique(dropna=True)),
    }
    return report


basic = pd.DataFrame([
    basic_report(train, "train"),
    basic_report(test, "test"),
])

basic.to_csv(OUTPUT_DIR / "basic_summary.csv", index=False)

print("\nBASIC SUMMARY")
print(basic.to_string(index=False))

# Missingness per column, including percentages.
for name, df in [("train", train), ("test", test)]:
    missing = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percent": df.isna().mean() * 100,
    })
    missing.to_csv(OUTPUT_DIR / f"{name}_missing_values.csv")

# Duplicate text within each split.
for name, df in [("train", train), ("test", test)]:
    norm = normalized_text(df[TEXT_COL])
    duplicate_mask = norm.duplicated(keep=False) & norm.ne("")
    duplicate_examples = df.loc[
        duplicate_mask, [TEXT_COL, TARGET_COL]
    ].copy()
    duplicate_examples.to_csv(
        OUTPUT_DIR / f"{name}_duplicate_texts.csv", index=False
    )


# ============================================================
# 4. Sentiment class distribution
# ============================================================

for name, df in [("train", train), ("test", test)]:
    counts = df[TARGET_COL].value_counts(dropna=False)
    percentages = df[TARGET_COL].value_counts(
        normalize=True, dropna=False
    ) * 100

    distribution = pd.DataFrame({
        "count": counts,
        "percentage": percentages,
    })
    distribution.to_csv(
        OUTPUT_DIR / f"{name}_class_distribution.csv"
    )

    print(f"\n{name.upper()} SENTIMENT DISTRIBUTION")
    print(distribution.round(2).to_string())

# Compare class counts and proportions across both splits.
all_labels = sorted(
    set(train[TARGET_COL].dropna().unique())
    | set(test[TARGET_COL].dropna().unique())
)

comparison = pd.DataFrame(index=all_labels)
for name, df in [("train", train), ("test", test)]:
    comparison[f"{name}_count"] = (
        df[TARGET_COL].value_counts().reindex(all_labels, fill_value=0)
    )
    comparison[f"{name}_percent"] = (
        df[TARGET_COL].value_counts(normalize=True)
        .reindex(all_labels, fill_value=0) * 100
    )

comparison.index.name = "Sentiment"
comparison.to_csv(OUTPUT_DIR / "train_test_class_comparison.csv")
print("\nTRAIN VS TEST CLASS COMPARISON")
print(comparison.round(2).to_string())

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for ax, (name, df) in zip(axes, [("Train", train), ("Test", test)]):
    order = df[TARGET_COL].value_counts().index
    sns.countplot(data=df, x=TARGET_COL, order=order, ax=ax)
    ax.set_title(f"{name}: sentiment counts")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of tweets")
    ax.tick_params(axis="x", rotation=35)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "class_distribution.png", dpi=160)
plt.close()


# ============================================================
# 5. Text length and basic text characteristics
# ============================================================

def add_text_features(df):
    out = df.copy()
    text = out[TEXT_COL].fillna("").astype(str)

    out["char_count"] = text.str.len()
    out["word_count"] = text.str.split().str.len()
    out["url_count"] = text.str.count(r"https?://\S+|www\.\S+")
    out["hashtag_count"] = text.str.count(r"#\w+")
    out["mention_count"] = text.str.count(r"@\w+")
    out["exclamation_count"] = text.str.count("!")
    out["question_count"] = text.str.count(r"\?")
    return out


train_features = add_text_features(train)
test_features = add_text_features(test)

feature_cols = [
    "char_count",
    "word_count",
    "url_count",
    "hashtag_count",
    "mention_count",
    "exclamation_count",
    "question_count",
]

for name, df in [
    ("train", train_features),
    ("test", test_features),
]:
    stats = df[feature_cols].describe().T
    stats.to_csv(OUTPUT_DIR / f"{name}_text_statistics.csv")
    print(f"\n{name.upper()} TEXT STATISTICS")
    print(stats.round(2).to_string())

    # Save a per-sentiment comparison.
    grouped = df.groupby(TARGET_COL)[feature_cols].agg(
        ["mean", "median", "std"]
    )
    grouped.to_csv(OUTPUT_DIR / f"{name}_text_stats_by_sentiment.csv")

# Word count distribution.
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (name, df) in zip(
    axes,
    [("Train", train_features), ("Test", test_features)],
):
    sns.histplot(
        data=df, x="word_count", bins=40, kde=True, ax=ax
    )
    ax.set_title(f"{name}: words per tweet")
    ax.set_xlabel("Word count")
    ax.set_ylabel("Number of tweets")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "word_count_distribution.png", dpi=160)
plt.close()

# Text length by sentiment.
fig, ax = plt.subplots(figsize=(11, 6))
sns.boxplot(
    data=train_features,
    x=TARGET_COL,
    y="word_count",
    order=EXPECTED_LABELS,
    ax=ax,
)
ax.set_title("Training tweet length by sentiment")
ax.set_xlabel("Sentiment")
ax.set_ylabel("Words per tweet")
ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "word_count_by_sentiment.png", dpi=160)
plt.close()


# ============================================================
# 6. Date distribution and time coverage
# ============================================================

for name, df in [("train", train), ("test", test)]:
    # The supplied dataset commonly uses DD-MM-YYYY.
    parsed_dates = pd.to_datetime(
        df[DATE_COL], format="%d-%m-%Y", errors="coerce"
    )

    date_summary = {
        "rows": len(df),
        "unparseable_dates": int(parsed_dates.isna().sum()),
        "earliest_date": (
            str(parsed_dates.min().date())
            if parsed_dates.notna().any() else None
        ),
        "latest_date": (
            str(parsed_dates.max().date())
            if parsed_dates.notna().any() else None
        ),
        "unique_dates": int(parsed_dates.nunique()),
    }

    with open(
        OUTPUT_DIR / f"{name}_date_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(date_summary, f, indent=2)

    daily_counts = (
        parsed_dates.dropna()
        .value_counts()
        .sort_index()
        .rename_axis("date")
        .reset_index(name="tweet_count")
    )
    daily_counts.to_csv(
        OUTPUT_DIR / f"{name}_daily_counts.csv", index=False
    )

    print(f"\n{name.upper()} DATE SUMMARY")
    print(json.dumps(date_summary, indent=2))

    plt.figure(figsize=(12, 4))
    plt.plot(
        daily_counts["date"],
        daily_counts["tweet_count"],
        marker="o",
        markersize=3,
    )
    plt.title(f"{name.title()}: tweets per date")
    plt.xlabel("Date")
    plt.ylabel("Number of tweets")
    plt.xticks(rotation=35)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{name}_daily_counts.png", dpi=160)
    plt.close()


# ============================================================
# 7. Location examination (descriptive only)
# ============================================================

if "Location" in train.columns and "Location" in test.columns:
    for name, df in [("train", train), ("test", test)]:
        locations = (
            df["Location"]
            .fillna("[Missing]")
            .astype(str)
            .str.strip()
            .replace("", "[Blank]")
            .value_counts()
            .head(30)
            .rename_axis("location")
            .reset_index(name="count")
        )
        locations.to_csv(
            OUTPUT_DIR / f"{name}_top_locations.csv", index=False
        )


# ============================================================
# 8. Train/test leakage and consistency checks
# ============================================================

train_text = normalized_text(train[TEXT_COL])
test_text = normalized_text(test[TEXT_COL])

train_nonempty = set(train_text[train_text.ne("")])
test_nonempty = set(test_text[test_text.ne("")])

overlap = train_nonempty & test_nonempty
print("\nTRAIN/TEST TEXT OVERLAP")
print(f"Unique non-empty train texts: {len(train_nonempty):,}")
print(f"Unique non-empty test texts:  {len(test_nonempty):,}")
print(f"Exact normalized overlap:     {len(overlap):,}")
print(
    "Overlap as % of unique test texts: "
    f"{100 * len(overlap) / max(len(test_nonempty), 1):.2f}%"
)

# Save overlapping examples for manual inspection.
if overlap:
    overlap_df = pd.DataFrame({"normalized_text": sorted(overlap)})
    overlap_df.to_csv(
        OUTPUT_DIR / "train_test_text_overlap.csv", index=False
    )

# Check if the same normalized tweet has conflicting labels.
combined = pd.concat([
    pd.DataFrame({
        "text": train_text,
        "label": train[TARGET_COL].astype(str),
        "split": "train",
    }),
    pd.DataFrame({
        "text": test_text,
        "label": test[TARGET_COL].astype(str),
        "split": "test",
    }),
], ignore_index=True)

combined = combined[combined["text"].ne("")]

label_counts_per_text = combined.groupby("text")["label"].nunique()
conflicting_texts = label_counts_per_text[
    label_counts_per_text > 1
].index

print(
    "Unique normalized texts assigned multiple labels: "
    f"{len(conflicting_texts):,}"
)

if len(conflicting_texts):
    combined[combined["text"].isin(conflicting_texts)].sort_values(
        "text"
    ).to_csv(
        OUTPUT_DIR / "texts_with_conflicting_labels.csv",
        index=False,
    )

# Check ID overlap where both files contain these fields.
for col in ["UserName", "ScreenName"]:
    if col in train.columns and col in test.columns:
        a = set(train[col].dropna().astype(str))
        b = set(test[col].dropna().astype(str))
        print(f"{col} overlap: {len(a & b):,}")


# ============================================================
# 9. Save representative examples per class
# ============================================================

for name, df in [("train", train), ("test", test)]:
    samples = (
        df.dropna(subset=[TEXT_COL, TARGET_COL])
        .groupby(TARGET_COL, group_keys=False)
        .head(5)
        [[TARGET_COL, TEXT_COL, DATE_COL]]
    )
    samples.to_csv(
        OUTPUT_DIR / f"{name}_examples_by_sentiment.csv",
        index=False,
    )


# ============================================================
# 10. Machine-readable overview
# ============================================================

overview = {
    "dataset": "Corona_NLP",
    "task": "multiclass text sentiment classification",
    "input_column": TEXT_COL,
    "target_column": TARGET_COL,
    "train_rows": int(len(train)),
    "test_rows": int(len(test)),
    "train_columns": train.columns.tolist(),
    "test_columns": test.columns.tolist(),
    "train_label_counts": {
        str(k): int(v)
        for k, v in train[TARGET_COL].value_counts().items()
    },
    "test_label_counts": {
        str(k): int(v)
        for k, v in test[TARGET_COL].value_counts().items()
    },
    "normalized_text_overlap_count": int(len(overlap)),
    "normalized_text_overlap_percent_of_test_unique": round(
        100 * len(overlap) / max(len(test_nonempty), 1), 3
    ),
}

with open(
    OUTPUT_DIR / "overview.json", "w", encoding="utf-8"
) as f:
    json.dump(overview, f, indent=2, ensure_ascii=False)

print(f"\nEDA complete. Outputs saved to: {OUTPUT_DIR.resolve()}")
