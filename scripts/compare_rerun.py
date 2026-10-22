"""Compare the full re-run of the ladder (outputs_rerun/) with the original runs (outputs/).

    python scripts/compare_rerun.py

For every run: validation / test accuracy of both versions, and how many test images get the same prediction.
Writes outputs_rerun/comparison.csv and prints a summary.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from birdcls import config as C                     # noqa: E402
from birdcls.data import load_data, make_loader     # noqa: E402
from birdcls.engine import predict_probs            # noqa: E402
from birdcls.models import build_model, normalization  # noqa: E402

ORIG, RERUN = ROOT / "outputs", ROOT / "outputs_rerun"


def test_preds(run_dir, data, device):
    """Test-set predictions of the checkpoint in run_dir (no TTA)."""
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu")
    model = build_model(ckpt["model"], data.num_classes, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    norm = (ckpt["mean"], ckpt["std"]) if "mean" in ckpt else normalization(ckpt["model"])
    loader = make_loader(data.test, ckpt["img_size"], C.EVAL_BATCH_SIZE, train=False,
                         pin_memory=device == "cuda", norm=norm)
    probs, _ = predict_probs(model, loader, device)
    del model
    torch.cuda.empty_cache()
    return probs.argmax(1).numpy()


def main():
    """For each re-run: compare accuracies with the original run and count identical test predictions."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_data()
    o = pd.read_csv(ORIG / "results.csv", index_col="run")
    r = pd.read_csv(RERUN / "results.csv", index_col="run")
    o, r = o[~o.tta], r[~r.tta]
    rows = []
    for run in r.index:
        if run not in o.index:
            continue
        same = (test_preds(ORIG / "runs" / run, data, device) == test_preds(RERUN / "runs" / run, data, device))
        rows.append({"run": run,
                     "val_orig": 100 * o.loc[run, "val_top1"], "val_rerun": 100 * r.loc[run, "val_top1"],
                     "test_orig": 100 * o.loc[run, "test_top1"], "test_rerun": 100 * r.loc[run, "test_top1"],
                     "avg_orig": 100 * o.loc[run, "test_avg_per_class"],
                     "avg_rerun": 100 * r.loc[run, "test_avg_per_class"],
                     "best_epoch_orig": o.loc[run, "best_epoch"], "best_epoch_rerun": r.loc[run, "best_epoch"],
                     "same_test_predictions": int(same.sum())})
        print(f"{run:50s} test {rows[-1]['test_orig']:.2f} -> {rows[-1]['test_rerun']:.2f}  "
              f"same predictions {same.sum()}/{len(same)}")
    df = pd.DataFrame(rows)
    df["test_diff"] = df.test_rerun - df.test_orig
    df.to_csv(RERUN / "comparison.csv", index=False)
    print(f"\n{len(df)} runs compared; identical test predictions in "
          f"{(df.same_test_predictions == len(data.test)).sum()} runs; "
          f"max |test difference| {df.test_diff.abs().max():.2f} pp, mean {df.test_diff.abs().mean():.2f} pp")


if __name__ == "__main__":
    main()
