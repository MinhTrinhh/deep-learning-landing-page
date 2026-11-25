"""Aggregate the runs written by train.py into the M2 result tables.

  python code/evaluate.py --runs RUNS_DIR --out outputs/metrics/m2 --checkpoints checkpoints --data DATA_DIR
  python code/evaluate.py --runs RUNS_DIR --best text      # prints the text feature with the best val macro-F1

Writes to --out:
  results.csv / results.md   mean ± std over seeds (val macro-F1, test accuracy, test macro-F1)
  per_class_f1.csv           test F1 per category, mean over seeds
  modality_agreement.csv     text / image / fusion correctness on test per seed (support vs. conflict counts)
  fusion_cases.csv           test articles where fusion fixes or breaks the text prediction (best fusion seed)
  figures/confusion_<kind>.png/pdf   row-normalized test confusion matrix, mean over seeds
With --checkpoints, the best-val seed of each selected run is copied there.
"""
import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

KINDS = ["text", "image", "concat"]
ROLE = {"text": "Unimodal A (text)", "image": "Unimodal B (image)", "concat": "Simple fusion (concat)"}
CHECKPOINT_NAMES = {"text": "unimodal_a_baseline.pt", "image": "unimodal_b_baseline.pt", "concat": "simple_fusion_model.pt"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("m2_results"))
    p.add_argument("--checkpoints", type=Path, default=None)
    p.add_argument("--best", choices=KINDS, default=None, help="only print the selected feature for this model kind")
    p.add_argument("--data", type=Path, default=None, help="N24News folder, adds headlines to fusion_cases.csv")
    return p.parse_args()


def load_runs(runs_dir):
    runs = []
    for path in sorted(runs_dir.glob("*/seed*/metrics.json")):
        m = json.loads(path.read_text())
        m["dir"] = path.parent
        runs.append(m)
    if not runs:
        raise SystemExit(f"no metrics.json under {runs_dir}")
    return runs


def summarize(runs):
    per_seed = pd.DataFrame([{
        "model": m["model"], "run": m["run"], "seed": m["seed"], "parameters": m["parameters"],
        "val_macro_f1": m["val"]["macro_f1"],
        "test_accuracy": m["test"]["accuracy"], "test_macro_f1": m["test"]["macro_f1"],
    } for m in runs])
    metrics = ["val_macro_f1", "test_accuracy", "test_macro_f1"]
    summary = per_seed.groupby(["model", "run"]).agg(
        seeds=("seed", "nunique"), parameters=("parameters", "first"),
        **{c: (c, "mean") for c in metrics}, **{f"{c}_std": (c, "std") for c in metrics},
    ).fillna(0).reset_index()
    summary["order"] = summary["model"].map(KINDS.index)
    return summary.sort_values(["order", "val_macro_f1"], ascending=[True, False]).drop(columns="order")


def write_markdown(summary, selected, path):
    lines = ["| Model | Run (features) | Seeds | Val macro-F1 | Test accuracy | Test macro-F1 | Head params |",
             "|---|---|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        mark = " **(selected)**" if selected[r["model"]] == r["run"] else ""
        cells = [f"{100 * r[c]:.2f} ± {100 * r[c + '_std']:.2f}" for c in ["val_macro_f1", "test_accuracy", "test_macro_f1"]]
        lines.append(f"| {ROLE[r['model']]} | `{r['run']}`{mark} | {r['seeds']} | {' | '.join(cells)} | {r['parameters']:,} |")
    lines += ["", "Values in %, mean ± std over seeds. Feature choice per model is made on validation macro-F1 only."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_confusion(group, title, path):
    cm = np.mean([np.asarray(m["test_confusion_matrix"], dtype=float) for m in group], axis=0)
    cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    classes = group[0]["classes"]
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=8)
    ax.set_yticks(range(len(classes)), classes, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=200)
    plt.close(fig)


def modality_agreement(by_kind, labels_by_id):
    """Per seed: which of the text / image / fusion runs is right on each test article."""
    rows, cases = [], None
    best_seed = max(by_kind["concat"], key=lambda m: m["val"]["macro_f1"])["seed"]
    for seed in sorted({m["seed"] for m in by_kind["concat"]}):
        group = {k: next((m for m in by_kind[k] if m["seed"] == seed), None) for k in KINDS}
        if None in group.values():
            continue
        ids = np.load(group["text"]["dir"] / "test_image_ids.npy")
        pred = {k: np.load(group[k]["dir"] / "test_probs.npy").argmax(1) for k in KINDS}
        y = labels_by_id.loc[ids].to_numpy()
        t, i, f = (pred[k] == y for k in KINDS)
        rows.append({
            "seed": seed, "n_test": len(y),
            "all correct": int((t & i & f).sum()),
            "support: text wrong, fusion right": int((~t & f).sum()),
            "harm: text right, fusion wrong": int((t & ~f).sum()),
            "conflict: text right, image wrong -> fusion right": int((t & ~i & f).sum()),
            "conflict: text right, image wrong -> fusion wrong": int((t & ~i & ~f).sum()),
            "conflict: image right, text wrong -> fusion right": int((~t & i & f).sum()),
            "conflict: image right, text wrong -> fusion wrong": int((~t & i & ~f).sum()),
            "both unimodal wrong, fusion right": int((~t & ~i & f).sum()),
            "all wrong": int((~t & ~i & ~f).sum()),
        })
        if seed == best_seed:
            classes = np.asarray(group["concat"]["classes"])
            keep = t != f
            cases = pd.DataFrame({
                "image_id": ids[keep],
                "case": np.where(f[keep], "support (fusion fixes text)", "harm (fusion breaks text)"),
                "true": classes[y[keep]],
                **{f"{k}_pred": classes[pred[k][keep]] for k in KINDS},
            })
    table = pd.DataFrame(rows)
    if not table.empty:
        table.loc[len(table)] = {"seed": "mean", **table.drop(columns="seed").mean().round(1).to_dict()}
    return table, cases


def main():
    args = parse_args()
    runs = load_runs(args.runs)
    summary = summarize(runs)
    selected = summary.groupby("model")["run"].first().to_dict()  # summary is sorted by val macro-F1
    by_kind = {k: [m for m in runs if m["run"] == selected.get(k)] for k in KINDS}

    if args.best:
        m = by_kind[args.best][0]
        print({"text": m["text_feat"], "image": m["image_feat"], "concat": m["run"]}[args.best])
        return

    out = args.out
    (out / "figures").mkdir(parents=True, exist_ok=True)
    summary.round(4).to_csv(out / "results.csv", index=False)
    write_markdown(summary, selected, out / "results.md")
    pd.DataFrame({ROLE[k]: pd.DataFrame([m["test_per_class_f1"] for m in by_kind[k]]).mean()
                  for k in KINDS if by_kind[k]}).rename_axis("class").round(4).to_csv(out / "per_class_f1.csv")
    for k in KINDS:
        if by_kind[k]:
            title = f"{ROLE[k]}: {selected[k]} (test, row-normalized, mean of {len(by_kind[k])} seeds)"
            plot_confusion(by_kind[k], title, out / "figures" / f"confusion_{k}")

    if all(by_kind.values()):
        index = pd.read_csv(Path(runs[0]["hyperparameters"]["features"]) / "index.csv", dtype={"image_id": str})
        agreement, cases = modality_agreement(by_kind, index.set_index("image_id")["y"])
        agreement.to_csv(out / "modality_agreement.csv", index=False, float_format="%g")
        if cases is not None:
            if args.data:
                from dataset import load_records
                headlines = load_records(args.data).set_index("image_id")["headline"]
                cases.insert(1, "headline", headlines.loc[cases["image_id"]].to_numpy())
            cases.to_csv(out / "fusion_cases.csv", index=False)

    if args.checkpoints:
        args.checkpoints.mkdir(parents=True, exist_ok=True)
        for k in KINDS:
            if by_kind[k]:
                best = max(by_kind[k], key=lambda m: m["val"]["macro_f1"])
                shutil.copy(best["dir"] / "model.pt", args.checkpoints / CHECKPOINT_NAMES[k])
                print(f"{CHECKPOINT_NAMES[k]} <- {best['run']} seed{best['seed']}")

    print((out / "results.md").read_text(encoding="utf-8"))
    print(f"selected per model kind (val macro-F1): {selected}")
    print(f"tables and figures in {out.resolve()}")


if __name__ == "__main__":
    main()
