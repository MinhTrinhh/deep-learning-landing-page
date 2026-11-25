"""Figures for the M2 report and landing page, from the files train.py / evaluate.py write.

  python code/plot_m2.py --runs RUNS_DIR --results outputs/metrics/m2

Writes to <results>/figures/ (png + pdf):
  learning_curves      val macro-F1 and train/val loss per epoch, mean over seeds (band = min..max)
  model_comparison     test macro-F1 ± std of the selected runs
  per_class_f1         test F1 per category for text, image and fusion; fusion minus text
  modality_agreement   support / harm / conflict counts on the test set, mean over seeds
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

KINDS = ["text", "image", "concat"]
LABEL = {"text": "Text (RoBERTa)", "image": "Image (ViT)", "concat": "Concat fusion"}
COLOR = {"text": "#2a78d6", "image": "#eb6834", "concat": "#1baf7a"}
INK, MUTED, GRID = "#21201c", "#898781", "#e4e3df"
GAIN, LOSS = "#2a78d6", "#d03b3b"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white", "legend.frameon": False,
})


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True, help="folder written by evaluate.py")
    return p.parse_args()


def save(fig, path):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def learning_curves(runs_dir, selected, path):
    fig, (ax_f1, ax_loss) = plt.subplots(1, 2, figsize=(12, 4.4))
    for kind in KINDS:
        hist = pd.concat([pd.read_csv(p) for p in sorted((runs_dir / selected[kind]).glob("seed*/history.csv"))])
        g = hist.groupby("epoch")
        c = COLOR[kind]
        f1 = g["val_macro_f1"].agg(["mean", "min", "max"]) * 100
        ax_f1.fill_between(f1.index, f1["min"], f1["max"], color=c, alpha=0.15, linewidth=0)
        ax_f1.plot(f1.index, f1["mean"], color=c, linewidth=2)
        best = f1["mean"].idxmax()
        ax_f1.plot(best, f1["mean"][best], "o", color=c, markersize=8, markeredgecolor="white", markeredgewidth=2)
        ax_f1.annotate(f"{LABEL[kind]}  {f1['mean'][best]:.1f}", (f1.index[-1], f1["mean"].iloc[-1]),
                       xytext=(6, 0), textcoords="offset points", va="center", fontsize=9)
        for col, style in [("train_loss", "--"), ("val_loss", "-")]:
            ax_loss.plot(g[col].mean().index, g[col].mean(), style, color=c, linewidth=2,
                         label=f"{LABEL[kind]} ({col.split('_')[0]})")
    ax_f1.set(title="Validation macro-F1 per epoch (dot = best epoch)", xlabel="epoch", ylabel="macro-F1 (%)")
    ax_f1.set_xlim(right=ax_f1.get_xlim()[1] * 1.35)
    ax_loss.set(title="Cross-entropy loss (dashed = train, solid = val)", xlabel="epoch", ylabel="loss")
    ax_loss.legend(fontsize=8, ncol=1, loc="upper right")
    fig.text(0.5, -0.02, "Mean of 3 seeds, shaded band = min..max across seeds. "
             "Each run stops 8 epochs after its best validation macro-F1.", ha="center", color=MUTED, fontsize=8)
    save(fig, path)


def model_comparison(results, selected, path):
    rows = results.set_index("run").loc[[selected[k] for k in KINDS]]
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    y = range(len(KINDS))[::-1]
    for yi, kind, (_, r) in zip(y, KINDS, rows.iterrows()):
        ax.barh(yi, r["test_macro_f1"] * 100, height=0.55, color=COLOR[kind])
        ax.errorbar(r["test_macro_f1"] * 100, yi, xerr=r["test_macro_f1_std"] * 100, color=INK, capsize=3, linewidth=1)
        ax.text(r["test_macro_f1"] * 100 + 1.5, yi, f"{100 * r['test_macro_f1']:.2f} ± {100 * r['test_macro_f1_std']:.2f}",
                va="center", fontsize=9)
    ax.set_yticks(list(y), [LABEL[k] for k in KINDS])
    ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False)
    ax.set(title="Test macro-F1 (mean ± std over 3 seeds)", xlabel="macro-F1 (%)")
    save(fig, path)


def per_class(per_class_f1, path):
    df = per_class_f1.copy() * 100
    df.columns = KINDS
    df = df.sort_values("text")
    fig, (ax, ax_d) = plt.subplots(1, 2, figsize=(12, 7.5), gridspec_kw={"width_ratios": [2.2, 1]}, sharey=True)
    for kind in KINDS:
        ax.plot(df[kind], range(len(df)), "o", color=COLOR[kind], markersize=7,
                markeredgecolor="white", markeredgewidth=1.5, label=LABEL[kind])
    ax.set_yticks(range(len(df)), df.index)
    ax.set(title="Test F1 per category (mean over 3 seeds)", xlabel="F1 (%)")
    ax.set_xlim(0, 100)
    ax.legend(loc="lower right", fontsize=9)
    delta = df["concat"] - df["text"]
    ax_d.barh(range(len(df)), delta, height=0.6, color=[GAIN if d > 0 else LOSS for d in delta])
    for i, d in enumerate(delta):
        ax_d.text(d + (0.15 if d >= 0 else -0.15), i, f"{d:+.1f}", va="center",
                  ha="left" if d >= 0 else "right", fontsize=8, color=INK)
    ax_d.axvline(0, color=MUTED, linewidth=1)
    ax_d.set_xlim(min(delta.min() * 1.5, -1), max(delta.max() * 1.5, 1))
    ax_d.set(title="Fusion minus text (F1 points)", xlabel="Δ F1")
    ax_d.grid(axis="y", visible=False)
    save(fig, path)


def modality_agreement(table, path):
    mean = table[table["seed"].astype(str) == "mean"].drop(columns=["seed", "n_test"]).iloc[0].astype(float)
    n_test = float(table["n_test"].iloc[0])
    mean = mean.drop("all correct").sort_values()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    colors = [GAIN if k.startswith("support") or k.endswith("fusion right") else
              LOSS if k.startswith("harm") or k.endswith("fusion wrong") else MUTED for k in mean.index]
    ax.barh(range(len(mean)), mean, height=0.6, color=colors)
    for i, v in enumerate(mean):
        ax.text(v + 15, i, f"{v:.0f}  ({100 * v / n_test:.1f}%)", va="center", fontsize=8)
    ax.set_yticks(range(len(mean)), mean.index, fontsize=9)
    ax.set_xlim(0, mean.max() * 1.25)
    ax.grid(axis="y", visible=False)
    ax.set(title=f"Text vs. image vs. fusion on the test set (mean of 3 seeds, n = {n_test:.0f}; "
                 f"all three correct: {table[table['seed'].astype(str) == 'mean']['all correct'].iloc[0]:.0f})",
           xlabel="articles")
    save(fig, path)


def main():
    args = parse_args()
    fig_dir = args.results / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(args.results / "results.csv")
    selected = {k: g.sort_values("val_macro_f1", ascending=False)["run"].iloc[0] for k, g in results.groupby("model")}

    learning_curves(args.runs, selected, fig_dir / "learning_curves")
    model_comparison(results, selected, fig_dir / "model_comparison")
    per_class(pd.read_csv(args.results / "per_class_f1.csv", index_col=0), fig_dir / "per_class_f1")
    modality_agreement(pd.read_csv(args.results / "modality_agreement.csv"), fig_dir / "modality_agreement")
    print(f"figures in {fig_dir.resolve()}")


if __name__ == "__main__":
    main()
