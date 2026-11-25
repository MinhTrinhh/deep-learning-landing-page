"""Aggregate the runs written by train.py into the M2 result tables.

  python code/evaluate.py --runs RUNS_DIR --out outputs/metrics/m2 --checkpoints checkpoints
  python code/evaluate.py --runs RUNS_DIR --best image      # prints the image feature with the best val macro-F1

Writes to --out:
  results.csv / results.md     mean ± std over seeds (val + test accuracy, macro-F1)
  per_class_f1.csv             test F1 per category, mean over seeds
  modality_agreement.csv       text vs image vs fusion correctness on test (support / conflict counts)
  fusion_cases.csv             test articles fusion fixes or breaks relative to text (seed of best fusion run)
  figures/confusion_<run>.png/pdf   row-normalized test confusion matrix, averaged over seeds
and, with --checkpoints, copies the best-val seed of the selected text / image / concat
runs to unimodal_a_baseline.pt, unimodal_b_baseline.pt, simple_fusion_model.pt.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

CHECKPOINT_NAMES = {"text": "unimodal_a_baseline.pt", "image": "unimodal_b_baseline.pt", "concat": "simple_fusion_model.pt"}
ROLE = {"text": "Unimodal A (text)", "image": "Unimodal B (image)", "concat": "Simple fusion (concat)"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--checkpoints", type=Path, default=None)
    p.add_argument("--best", choices=["text", "image", "concat"], default=None,
                   help="only print the best feature/run name for this model kind and exit")
    p.add_argument("--data", type=Path, default=None, help="optional N24News folder, adds headlines to fusion_cases.csv")
    p.add_argument("--include-debug", action="store_true")
    return p.parse_args()


def load_runs(runs_dir: Path, include_debug: bool) -> list[dict]:
    runs = []
    for path in sorted(runs_dir.glob("*/seed*/metrics.json")):
        m = json.loads(path.read_text())
        if m.get("debug") and not include_debug:
            continue
        m["dir"] = path.parent
        runs.append(m)
    if not runs:
        raise SystemExit(f"no metrics.json under {runs_dir}")
    return runs


def summarize(runs: list[dict]) -> pd.DataFrame:
    rows = []
    for m in runs:
        rows.append({"run": m["run"], "model": m["model"], "seed": m["seed"],
                     "val_accuracy": m["val"]["accuracy"], "val_macro_f1": m["val"]["macro_f1"],
                     "test_accuracy": m["test"]["accuracy"], "test_macro_f1": m["test"]["macro_f1"],
                     "parameters": m["parameters"], "train_seconds": m["train_seconds"], "best_epoch": m["best_epoch"]})
    per_seed = pd.DataFrame(rows)
    agg = per_seed.groupby(["model", "run"]).agg(
        seeds=("seed", "nunique"),
        val_macro_f1=("val_macro_f1", "mean"), val_macro_f1_std=("val_macro_f1", "std"),
        test_accuracy=("test_accuracy", "mean"), test_accuracy_std=("test_accuracy", "std"),
        test_macro_f1=("test_macro_f1", "mean"), test_macro_f1_std=("test_macro_f1", "std"),
        parameters=("parameters", "first"), train_seconds=("train_seconds", "mean"),
    ).reset_index().fillna({"val_macro_f1_std": 0, "test_accuracy_std": 0, "test_macro_f1_std": 0})
    order = {"text": 0, "image": 1, "concat": 2}
    return agg.sort_values(["model", "val_macro_f1"], key=lambda s: s.map(order) if s.name == "model" else -s)


def best_runs(summary: pd.DataFrame) -> dict[str, str]:
    """Per model kind, the run (feature choice) with the highest mean val macro-F1."""
    return {kind: g.sort_values("val_macro_f1", ascending=False)["run"].iloc[0] for kind, g in summary.groupby("model")}


def write_markdown(summary: pd.DataFrame, selected: dict[str, str], path: Path) -> None:
    lines = ["| Model | Run (features) | Seeds | Val macro-F1 | Test accuracy | Test macro-F1 | Head params |",
             "|---|---|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        mark = " **(selected)**" if selected.get(r["model"]) == r["run"] else ""
        lines.append(
            f"| {ROLE[r['model']]} | `{r['run']}`{mark} | {r['seeds']} "
            f"| {100 * r['val_macro_f1']:.2f} ± {100 * r['val_macro_f1_std']:.2f} "
            f"| {100 * r['test_accuracy']:.2f} ± {100 * r['test_accuracy_std']:.2f} "
            f"| {100 * r['test_macro_f1']:.2f} ± {100 * r['test_macro_f1_std']:.2f} | {r['parameters']:,} |")
    lines += ["", "Values in %, mean ± std over seeds. Feature choice per model is made on validation macro-F1 only."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def per_class_table(runs: list[dict], selected: dict[str, str]) -> pd.DataFrame:
    table = {}
    for kind, name in selected.items():
        rows = [m["test_per_class_f1"] for m in runs if m["run"] == name]
        table[ROLE[kind]] = pd.DataFrame(rows).mean()
    df = pd.DataFrame(table)
    df.index.name = "class"
    return df.round(4)


def plot_confusions(runs: list[dict], selected: dict[str, str], fig_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    for kind, name in selected.items():
        group = [m for m in runs if m["run"] == name]
        cm = np.mean([np.asarray(m["test_confusion_matrix"], dtype=float) for m in group], axis=0)
        cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        classes = group[0]["classes"]
        fig, ax = plt.subplots(figsize=(10, 8.5))
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=8)
        ax.set_yticks(range(len(classes)), classes, fontsize=8)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{ROLE[kind]}: {name} (test, row-normalized, mean of {len(group)} seeds)")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(fig_dir / f"confusion_{kind}.{ext}", dpi=200)
        plt.close(fig)


def load_predictions(m: dict) -> tuple[np.ndarray, np.ndarray]:
    return np.load(m["dir"] / "test_probs.npy").astype(np.float32).argmax(1), np.load(m["dir"] / "test_image_ids.npy")


def modality_agreement(runs, selected, labels_by_id) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """For each seed with all three selected runs: which model is right on each test article."""
    if not all(k in selected for k in ("text", "image", "concat")):
        return pd.DataFrame(), None
    by_seed = {}
    for m in runs:
        for kind, name in selected.items():
            if m["run"] == name:
                by_seed.setdefault(m["seed"], {})[kind] = m
    rows, cases = [], None
    best_concat_seed = max((m for m in runs if m["run"] == selected["concat"]), key=lambda m: m["val"]["macro_f1"])["seed"]
    for seed, group in sorted(by_seed.items()):
        if len(group) < 3:
            continue
        preds = {k: load_predictions(group[k]) for k in group}
        ids = preds["text"][1]
        assert all((preds[k][1] == ids).all() for k in preds)
        y = labels_by_id.loc[ids].to_numpy()
        t, i, f = (preds[k][0] == y for k in ("text", "image", "concat"))
        rows.append({
            "seed": seed, "n_test": len(y),
            "all_correct": int((t & i & f).sum()),
            "text_wrong_fusion_right (support)": int((~t & f).sum()),
            "text_right_fusion_wrong (harm)": int((t & ~f).sum()),
            "text_vs_image_disagree": int((preds["text"][0] != preds["image"][0]).sum()),
            "conflict: text right, image wrong -> fusion right": int((t & ~i & f).sum()),
            "conflict: text right, image wrong -> fusion wrong": int((t & ~i & ~f).sum()),
            "conflict: image right, text wrong -> fusion right": int((~t & i & f).sum()),
            "conflict: image right, text wrong -> fusion wrong": int((~t & i & ~f).sum()),
            "both unimodal wrong -> fusion right": int((~t & ~i & f).sum()),
            "all wrong": int((~t & ~i & ~f).sum()),
        })
        if seed == best_concat_seed:
            classes = group["concat"]["classes"]
            keep = (~t & f) | (t & ~f)
            cases = pd.DataFrame({
                "image_id": ids[keep],
                "case": np.where((~t & f)[keep], "support (fusion fixes text)", "harm (fusion breaks text)"),
                "true": [classes[k] for k in y[keep]],
                "text_pred": [classes[k] for k in preds["text"][0][keep]],
                "image_pred": [classes[k] for k in preds["image"][0][keep]],
                "fusion_pred": [classes[k] for k in preds["concat"][0][keep]],
            })
    table = pd.DataFrame(rows)
    if not table.empty:
        mean = table.drop(columns="seed").mean().round(1)
        table = pd.concat([table, pd.DataFrame([{"seed": "mean", **mean.to_dict()}])], ignore_index=True)
    return table, cases


def main() -> None:
    args = parse_args()
    runs = load_runs(args.runs, args.include_debug)
    summary = summarize(runs)
    selected = best_runs(summary)

    if args.best:
        name = selected[args.best]
        m = next(r for r in runs if r["run"] == name)
        print(m["text_feat"] if args.best == "text" else m["image_feat"] if args.best == "image" else name)
        return

    out = args.out or Path("m2_results")
    out.mkdir(parents=True, exist_ok=True)
    summary.round(4).to_csv(out / "results.csv", index=False)
    write_markdown(summary, selected, out / "results.md")
    per_class_table(runs, selected).to_csv(out / "per_class_f1.csv")
    plot_confusions(runs, selected, out / "figures")

    labels_by_id = None
    index_path = Path(runs[0]["hyperparameters"]["features"]) / "index.csv"
    if index_path.exists():
        index = pd.read_csv(index_path, dtype={"image_id": str})
        labels_by_id = index.set_index("image_id")["y"]
    if labels_by_id is not None:
        agreement, cases = modality_agreement(runs, selected, labels_by_id)
        if not agreement.empty:
            agreement.to_csv(out / "modality_agreement.csv", index=False)
        if cases is not None:
            if args.data is not None:
                from dataset import load_records
                records = load_records(args.data).set_index("image_id")
                cases.insert(1, "headline", records.loc[cases["image_id"], "headline"].to_numpy())
            cases.to_csv(out / "fusion_cases.csv", index=False)
    else:
        print(f"skipped modality agreement: {index_path} not found")

    if args.checkpoints:
        args.checkpoints.mkdir(parents=True, exist_ok=True)
        for kind, name in selected.items():
            best = max((m for m in runs if m["run"] == name), key=lambda m: m["val"]["macro_f1"])
            shutil.copy(best["dir"] / "model.pt", args.checkpoints / CHECKPOINT_NAMES[kind])
            print(f"{CHECKPOINT_NAMES[kind]} <- {name} seed{best['seed']}")

    print((out / "results.md").read_text(encoding="utf-8"))
    print(f"selected per model kind (val macro-F1): {selected}")
    print(f"tables and figures in {out.resolve()}")


if __name__ == "__main__":
    main()
