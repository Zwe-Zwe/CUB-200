"""Evidence for the report appendix: per-image predictions of every Track 2 run, McNemar tests for each ladder
decision (validation and test), and bootstrap confidence intervals for the final models.

    python report/appendix_stats.py

Writes outputs/predictions/<run>.npz (cached; val / test labels and predictions) and report/data/*.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from birdcls import config as C                                   # noqa: E402
from birdcls.data import load_data, make_loader                   # noqa: E402
from birdcls.engine import predict_probs                          # noqa: E402
from birdcls.evaluate import load_run                             # noqa: E402
from birdcls.models import normalization                          # noqa: E402

PRED = C.OUTPUT_DIR / "predictions"
DATA = ROOT / "report" / "data"
PRED.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

R, X = "resnet50.tv2_in1k__", "convnext_tiny__"
RUNS = ([R + c for c in ["tl_L0_probe", "tl_L1_finetune", "tl_L2_discriminative", "tl_L3_regularise",
                         "tl_L4_strongaug", "tl_L5_longer_strongaug", "tl_L6_hires_strongaug",
                         "tl_L8_sgd_1e-3", "tl_L8_sgd_3e-3", "tl_L8_sgd_1e-2", "tl_L6_hires_strongaug__full"]]
        + [X + c for c in ["tl_L0_probe", "tl_L1_finetune", "tl_L2_discriminative", "tl_L3_regularise",
                           "tl_L4_strongaug", "tl_L5_longer", "tl_L6_hires", "tl_L8_llrd_080", "tl_L8_llrd_065",
                           "tl_L6_hires__full"]])


def predictions(name, data, device):
    """Validation and test predictions of one run (no TTA), cached on disk."""
    f = PRED / f"{name}.npz"
    if f.exists():
        return dict(np.load(f))
    model, ckpt = load_run(name, data.num_classes, device)
    norm = (ckpt["mean"], ckpt["std"]) if "mean" in ckpt else normalization(ckpt["model"])
    out = {}
    for split, df in [("val", data.val), ("test", data.test)]:
        loader = make_loader(df, ckpt["img_size"], C.EVAL_BATCH_SIZE, train=False, pin_memory=device == "cuda",
                             norm=norm)
        probs, y = predict_probs(model, loader, device)
        out[f"{split}_y"], out[f"{split}_pred"] = y.numpy(), probs.argmax(1).numpy()
    np.savez(f, **out)
    del model
    torch.cuda.empty_cache()
    print("predicted", name)
    return out


def mcnemar(y, a, b):
    """Exact McNemar test: only the images where exactly one of the two models is right matter."""
    a_only = int(((a == y) & (b != y)).sum())
    b_only = int(((a != y) & (b == y)).sum())
    p = binomtest(a_only, a_only + b_only, 0.5).pvalue if a_only + b_only else 1.0
    return a_only, b_only, p


def main():
    """Compute every appendix statistic from the saved checkpoints and write them to report/data/."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_data()
    P = {n: predictions(n, data, device) for n in RUNS}

    # ---- every ladder decision: new step vs the step it was compared with
    pairs = [("ResNet-50", R, [("L1 vs L0", "tl_L1_finetune", "tl_L0_probe"),
                               ("L2 vs L0", "tl_L2_discriminative", "tl_L0_probe"),
                               ("L3 vs L2", "tl_L3_regularise", "tl_L2_discriminative"),
                               ("L4 vs L3", "tl_L4_strongaug", "tl_L3_regularise"),
                               ("L5 vs L4", "tl_L5_longer_strongaug", "tl_L4_strongaug"),
                               ("L6 vs L5", "tl_L6_hires_strongaug", "tl_L5_longer_strongaug"),
                               ("L8 (SGD 1e-2) vs L6", "tl_L8_sgd_1e-2", "tl_L6_hires_strongaug")]),
             ("ConvNeXt-Tiny", X, [("L1 vs L0", "tl_L1_finetune", "tl_L0_probe"),
                                   ("L2 vs L0", "tl_L2_discriminative", "tl_L0_probe"),
                                   ("L3 vs L2", "tl_L3_regularise", "tl_L2_discriminative"),
                                   ("L4 vs L3", "tl_L4_strongaug", "tl_L3_regularise"),
                                   ("L5 vs L3", "tl_L5_longer", "tl_L3_regularise"),
                                   ("L6 vs L5", "tl_L6_hires", "tl_L5_longer"),
                                   ("L8 (LLRD 0.65) vs L6", "tl_L8_llrd_065", "tl_L6_hires")])]
    rows = []
    for model, pre, comps in pairs:
        for label, new, old in comps:
            row = {"model": model, "comparison": label}
            for split in ("val", "test"):
                a, b = P[pre + new], P[pre + old]
                y = a[f"{split}_y"]
                assert (y == b[f"{split}_y"]).all()
                n_new, n_old, p = mcnemar(y, a[f"{split}_pred"], b[f"{split}_pred"])
                row.update({f"{split}_acc_new": 100 * (a[f"{split}_pred"] == y).mean(),
                            f"{split}_acc_old": 100 * (b[f"{split}_pred"] == y).mean(),
                            f"{split}_only_new": n_new, f"{split}_only_old": n_old, f"{split}_p": p})
            rows.append(row)
    dec = pd.DataFrame(rows)
    dec.to_csv(DATA / "mcnemar_decisions.csv", index=False)
    print(dec[["model", "comparison", "val_acc_old", "val_acc_new", "val_only_new", "val_only_old", "val_p",
               "test_p"]].round(4).to_string(index=False))

    # ---- final models: ConvNeXt-Tiny vs ResNet-50 on the test set + bootstrap CIs
    fc, fr = P[X + "tl_L6_hires__full"], P[R + "tl_L6_hires_strongaug__full"]
    y = fc["test_y"]
    c_only, r_only, p = mcnemar(y, fc["test_pred"], fr["test_pred"])
    rng = np.random.default_rng(0)
    rows = []
    for name, pr in [("ConvNeXt-Tiny final", fc["test_pred"]), ("ResNet-50 final", fr["test_pred"])]:
        top1, ave = [], []
        for _ in range(2000):             # resample test images with replacement
            i = rng.integers(0, len(y), len(y))
            yt, yp = y[i], pr[i]
            top1.append((yt == yp).mean())
            ave.append(np.mean([(yp[yt == c] == c).mean() for c in np.unique(yt)]))
        rows.append({"model": name, "top1": 100 * (y == pr).mean(),
                     "top1_lo": 100 * np.percentile(top1, 2.5), "top1_hi": 100 * np.percentile(top1, 97.5),
                     "ave_lo": 100 * np.percentile(ave, 2.5), "ave_hi": 100 * np.percentile(ave, 97.5)})
    ci = pd.DataFrame(rows)
    ci.to_csv(DATA / "bootstrap_ci.csv", index=False)
    pd.DataFrame([{"convnext_only": c_only, "resnet_only": r_only, "p": p}]).to_csv(DATA / "final_mcnemar.csv",
                                                                                   index=False)
    print(ci.round(2).to_string(index=False))
    print(f"final: ConvNeXt only right {c_only}, ResNet only right {r_only}, p = {p:.2e}")


if __name__ == "__main__":
    main()
