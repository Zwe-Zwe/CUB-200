"""Evaluate every finished run on the test set and analyse the best one.

    python -m birdcls.evaluate                      # all complete runs in outputs/runs/
    python -m birdcls.evaluate --runs convnext_tiny__A_standard convnext_tiny__B_finegrained
    python -m birdcls.evaluate --analyse convnext_tiny__tl_L6_hires__full   # error analysis only, any run

Each run's best checkpoint is evaluated with and without TTA (horizontal-flip averaging, no retraining).
The best entry is selected by VALIDATION accuracy only; test numbers are reported, never used to choose.

Outputs (outputs/):
    results.csv / results.md     one row per run (+TTA): train/val accuracy, overfitting gap, test metrics
                                 (merged: --runs only replaces the rows of the runs it evaluates)
    figures/val_curves.png       validation accuracy per epoch, all runs
    runs/<run>/curves.png        loss / accuracy curves (re-drawn)
    error_analysis/              per-class accuracy, worst classes, top confusions, example images
"""
import argparse
from collections import Counter

import numpy as np
import pandas as pd
import torch

from . import config as C
from .data import make_loader, load_data
from .engine import evaluate, predict_probs
from .metrics import compute_metrics
from .models import build_model, count_params, normalization
from .plots import (plot_confused_pairs, plot_history, plot_misclassified, plot_per_class_hist,
                    plot_val_curves)
from .utils import get_device, load_json, save_json


def load_run(name, num_classes, device):
    """Rebuild the run's architecture and load its best checkpoint."""
    ckpt = torch.load(C.run_dir(name) / "best.pt", map_location="cpu")
    model = build_model(ckpt["model"], num_classes, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device), ckpt


def evaluate_runs(names, data, device):
    rows, preds = [], {}
    for name in names:
        hist = load_json(C.run_dir(name) / "history.json")
        model, ckpt = load_run(name, data.num_classes, device)
        size = ckpt["img_size"]                 # evaluate at the resolution of the best epoch
        # input statistics the model was trained with (older checkpoints: look them up)
        norm = (ckpt["mean"], ckpt["std"]) if "mean" in ckpt else normalization(ckpt["model"])
        loaders = {split: make_loader(df, size, C.EVAL_BATCH_SIZE, train=False,
                                      pin_memory=device == "cuda", norm=norm)
                   for split, df in [("train", data.train), ("val", data.val), ("test", data.test)]}
        full = bool(hist.get("full_train", False))   # retrained on train + val: no honest validation score
        if full:
            loaders["train"] = make_loader(data.full_train, size, C.EVAL_BATCH_SIZE, train=False,
                                           pin_memory=device == "cuda", norm=norm)
        # Clean training accuracy (eval preprocessing, no augmentation / CutMix): comparable across runs
        _, train_acc = evaluate(model, loaders["train"], device)

        for tta in (False, True):
            label = name + (" +TTA" if tta else "")
            # full-train runs saw the validation images in training, so their val score is meaningless (NaN);
            # NaN also keeps them out of "best by validation" selection
            val_acc = float("nan") if full else evaluate(model, loaders["val"], device, tta)[1]
            probs, y = predict_probs(model, loaders["test"], device, tta)
            yt, yp = y.numpy(), probs.argmax(1).numpy()
            top1, avg_cls, per_class = compute_metrics(yt, yp, data.num_classes)
            preds[label] = (yt, yp, per_class)
            rows.append({"run": label, "model": ckpt["model"], "config": ckpt["config"], "tta": tta,
                         "params_M": round(count_params(model), 1), "eval_size": size,
                         "best_epoch": ckpt["epoch"] + 1, "train_acc": train_acc, "val_top1": val_acc,
                         "gap": train_acc - val_acc, "test_top1": top1, "test_avg_per_class": avg_cls,
                         "train_minutes": hist["minutes"]})
            print(f"{label:55s} val {val_acc:.4f} | test top-1 {top1:.4f} | test avg/class {avg_cls:.4f}")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    return pd.DataFrame(rows).set_index("run"), preds


def to_markdown(df):
    """DataFrame -> Markdown table (avoids an extra dependency on `tabulate`)."""
    cols = [df.index.name or ""] + list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in [idx, *row.values]) + " |")
    return "\n".join(lines) + "\n"


def error_analysis(label, preds, data, out_dir):
    """Per-class accuracy, worst classes and most common confusions of one run."""
    out_dir.mkdir(parents=True, exist_ok=True)
    yt, yp, per_class = preds[label]
    names = data.class_names

    per_class_df = (pd.DataFrame({"class": names, "test_acc": per_class,
                                  "n_test": pd.Series(yt).value_counts().reindex(range(len(names))).values})
                    .sort_values("test_acc"))
    per_class_df.to_csv(out_dir / "per_class_accuracy.csv", index_label="label")

    pairs = Counter((t, p) for t, p in zip(yt, yp) if t != p)
    conf_df = pd.DataFrame([{"count": n, "true": names[t], "predicted": names[p]}
                            for (t, p), n in pairs.most_common(20)])
    conf_df.to_csv(out_dir / "top_confusions.csv", index=False)

    plot_per_class_hist(per_class, f"Per-class test accuracy ({label})", out_dir / "per_class_hist.png")
    top_pairs = [tp for tp, _ in pairs.most_common(4)]
    if top_pairs:
        examples = data.full_train.groupby("label").path.first().to_dict()
        plot_confused_pairs(top_pairs, examples, names, out_dir / "confused_pairs.png")

    print(f"\nWorst 10 classes ({label}):")
    print(per_class_df.head(10).to_string(index=False))
    print("\nMost common confusions:")
    print(conf_df.head(10).to_string(index=False))


def bird_group(name):
    """Coarse bird group = last word of the species name ("Ring billed Gull" -> "Gull")."""
    return name.split()[-1]


def analyse_run(name, data, device):
    """Detailed test-set error analysis of one run, written to outputs/error_analysis/<run>/ (results.csv untouched):
    per-class accuracy, top confusions, Top-5 accuracy, share of errors within the same bird group, per-image
    predictions, and figures (per-class histogram, misclassified examples)."""
    out = C.OUTPUT_DIR / "error_analysis" / name
    out.mkdir(parents=True, exist_ok=True)
    model, ckpt = load_run(name, data.num_classes, device)
    norm = (ckpt["mean"], ckpt["std"]) if "mean" in ckpt else normalization(ckpt["model"])
    loader = make_loader(data.test, ckpt["img_size"], C.EVAL_BATCH_SIZE, train=False,
                         pin_memory=device == "cuda", norm=norm)
    probs, y = predict_probs(model, loader, device)          # no TTA: the final models do not use it
    yt, yp = y.numpy(), probs.argmax(1).numpy()
    top1, avg_cls, per_class = compute_metrics(yt, yp, data.num_classes)
    top5 = float((probs.topk(5, dim=1).indices == y[:, None]).any(1).float().mean())
    names = data.class_names

    error_analysis(name, {name: (yt, yp, per_class)}, data, out)

    wrong = yt != yp
    same_group = sum(bird_group(names[t]) == bird_group(names[p]) for t, p in zip(yt[wrong], yp[wrong]))
    conf = probs.max(1).values.numpy()
    pred_df = pd.DataFrame({"file": data.test.file.values, "true": [names[i] for i in yt],
                            "pred": [names[i] for i in yp], "confidence": conf, "correct": ~wrong})
    pred_df.to_csv(out / "test_predictions.csv", index=False)

    # misclassified examples for the 4 most common confusions: the most confident wrong test image of each pair
    pairs = Counter((t, p) for t, p in zip(yt, yp) if t != p).most_common(4)
    train_example = data.full_train.groupby("label").path.first().to_dict()
    items = []
    for (t, p), _ in pairs:
        idx = [i for i in range(len(yt)) if yt[i] == t and yp[i] == p]
        i = max(idx, key=lambda j: conf[j])
        items.append({"test_path": data.test.path.values[i], "true": names[t], "pred": names[p],
                      "conf": float(conf[i]), "pred_example_path": train_example[p]})
    if items:
        plot_misclassified(items, out / "misclassified_examples.png")

    summary = {"run": name, "test_top1": top1, "test_avg_per_class": avg_cls, "test_top5": top5,
               "n_test": int(len(yt)), "n_errors": int(wrong.sum()),
               "errors_within_same_bird_group": int(same_group),
               "share_errors_within_same_bird_group": float(same_group / max(1, wrong.sum())),
               "classes_100pct": int((per_class == 1).sum()), "classes_below_50pct": int((per_class < 0.5).sum()),
               "classes_0pct": int((per_class == 0).sum()), "median_class_acc": float(np.nanmedian(per_class)),
               "mean_conf_correct": float(conf[~wrong].mean()), "mean_conf_wrong": float(conf[wrong].mean())}
    save_json(summary, out / "summary.json")
    print("\n" + "\n".join(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}" for k, v in summary.items()))
    del model
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", help="run names (default: every complete run)")
    ap.add_argument("--analyse", nargs="+", help="only write a detailed error analysis for these runs")
    args = ap.parse_args()

    if args.analyse:
        device, data = get_device(), load_data()
        for name in args.analyse:
            analyse_run(name, data, device)
        return

    names = args.runs or [n for n in C.completed_runs() if not n.endswith("__smoke")]
    assert names, f"no complete runs found in {C.RUNS_DIR}"
    device = get_device()
    data = load_data()
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    histories = {n: load_json(C.run_dir(n) / "history.json") for n in names}
    for n, h in histories.items():
        plot_history(h, n, C.run_dir(n) / "curves.png")
    # the combined figure always shows every complete run, not only the ones evaluated now
    all_hist = {n: load_json(C.run_dir(n) / "history.json")
                for n in C.completed_runs() if not n.endswith("__smoke")}
    plot_val_curves(all_hist, C.FIGURES_DIR / "val_curves.png")

    results, preds = evaluate_runs(names, data, device)
    results = results.round(4)
    print("\n" + results.drop(columns=["model", "config"]).to_string())

    # Merge into the existing table: re-evaluated runs replace their old rows, other runs are kept
    csv = C.OUTPUT_DIR / "results.csv"
    if csv.exists():
        old = pd.read_csv(csv, index_col="run")
        old = old[~old.index.str.replace(" +TTA", "", regex=False).isin(names)]
        results = pd.concat([old, results])
    results.to_csv(csv)
    (C.OUTPUT_DIR / "results.md").write_text(to_markdown(results))

    best = results.val_top1.idxmax()            # selection by validation accuracy only
    print(f"\nBest by validation accuracy (all runs in results.csv): {best}")
    if best in preds:                           # error analysis needs this session's predictions
        error_analysis(best, preds, data, C.OUTPUT_DIR / "error_analysis")
    else:
        print("best run was not re-evaluated in this call, so outputs/error_analysis/ is left unchanged")


if __name__ == "__main__":
    main()
