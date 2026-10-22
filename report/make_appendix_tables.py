"""Write the appendix tables as Typst rows (report/data/tables.typ) straight from the result files, so no number
in the appendix is typed by hand. Run after report/appendix_stats.py.

    python report/make_appendix_tables.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "report" / "data"
EA = ROOT / "outputs" / "error_analysis"
R, X = "resnet50.tv2_in1k__", "convnext_tiny__"


# ---- small helpers that turn Python values into Typst table cells
def cell(x):
    """One table cell in Typst markup; config names are shown as code, special characters are escaped."""
    x = str(x)
    if x.startswith("tl_"):                       # config names: monospace, underscores kept literally
        return f'raw("{x}")'
    return "[" + x.replace("#", "\\#").replace("_", "\\_").replace("*", "\\*") + "]"


def row(*xs):
    """One table row: the cells separated by commas, ready to be spread into a Typst table()."""
    return "  " + ", ".join(cell(x) for x in xs) + ",\n"


def pfmt(p):
    """Format a p-value for a table: very small values as "< 0.0001", others with 2 to 4 decimals."""
    return "< 0.0001" if p < 1e-4 else f"{p:.4f}" if p < 0.01 else f"{p:.2f}"


out = []

# ---- A1: every run of the ladder
res = pd.read_csv(ROOT / "outputs" / "results.csv", index_col="run")
res = res[~res.tta]
steps = [("L0", "tl_L0_probe", "tl_L0_probe"), ("L1", "tl_L1_finetune", "tl_L1_finetune"),
         ("L2", "tl_L2_discriminative", "tl_L2_discriminative"), ("L3", "tl_L3_regularise", "tl_L3_regularise"),
         ("L4", "tl_L4_strongaug", "tl_L4_strongaug"), ("L5", "tl_L5_longer_strongaug", "tl_L5_longer"),
         ("L6", "tl_L6_hires_strongaug", "tl_L6_hires"),
         ("L7 (final)", "tl_L6_hires_strongaug__full", "tl_L6_hires__full"),
         ("L8 SGD 1e-3", "tl_L8_sgd_1e-3", None), ("L8 SGD 3e-3", "tl_L8_sgd_3e-3", None),
         ("L8 SGD 1e-2", "tl_L8_sgd_1e-2", None), ("L8 LLRD 0.8", None, "tl_L8_llrd_080"),
         ("L8 LLRD 0.65", None, "tl_L8_llrd_065")]
s = "#let all-runs = (\n"
for step, rc, cc in steps:
    for model, pre, cfg in (("ResNet-50", R, rc), ("ConvNeXt-Tiny", X, cc)):
        if cfg is None:
            continue
        r = res.loc[pre + cfg]
        val = "--" if np.isnan(r.val_top1) else f"{100 * r.val_top1:.2f}"
        s += row(step, model, f"{cfg}", f"{100 * r.train_acc:.1f}", val, f"{100 * r.test_top1:.2f}",
                 f"{100 * r.test_avg_per_class:.2f}", int(r.best_epoch))
out.append(s + ")\n")

# ---- A2: McNemar for every decision (+ cumulative L2 -> L6)
dec = pd.read_csv(DATA / "mcnemar_decisions.csv")
P = lambda n: np.load(ROOT / "outputs" / "predictions" / f"{n}.npz")
extra = []
for model, pre, l6 in (("ResNet-50", R, "tl_L6_hires_strongaug"), ("ConvNeXt-Tiny", X, "tl_L6_hires")):
    a, b = P(pre + l6), P(pre + "tl_L2_discriminative")
    rowd = {"model": model, "comparison": "L6 vs L2 (L3 to L6 together)"}
    for sp in ("val", "test"):
        y = a[f"{sp}_y"]
        n_new = int(((a[f"{sp}_pred"] == y) & (b[f"{sp}_pred"] != y)).sum())
        n_old = int(((a[f"{sp}_pred"] != y) & (b[f"{sp}_pred"] == y)).sum())
        rowd.update({f"{sp}_acc_old": 100 * (b[f"{sp}_pred"] == y).mean(),
                     f"{sp}_acc_new": 100 * (a[f"{sp}_pred"] == y).mean(), f"{sp}_only_new": n_new,
                     f"{sp}_only_old": n_old, f"{sp}_p": binomtest(n_new, n_new + n_old).pvalue})
    extra.append(rowd)
dec = pd.concat([dec, pd.DataFrame(extra)], ignore_index=True)
s = "#let decisions = (\n"
for _, d in dec.iterrows():
    s += row(d.model, d.comparison, f"{d.val_acc_old:.1f} → {d.val_acc_new:.1f}",
             f"{int(d.val_only_new)} / {int(d.val_only_old)}", pfmt(d.val_p),
             f"{d.test_acc_old:.1f} → {d.test_acc_new:.1f}", pfmt(d.test_p))
out.append(s + ")\n")

# ---- A3: bootstrap confidence intervals + final McNemar
ci = pd.read_csv(DATA / "bootstrap_ci.csv")
s = "#let cis = (\n"
for _, c in ci.iterrows():
    s += row(c.model, f"{c.top1:.2f}", f"{c.top1_lo:.1f} to {c.top1_hi:.1f}", f"{c.ave_lo:.1f} to {c.ave_hi:.1f}")
out.append(s + ")\n")
fm = pd.read_csv(DATA / "final_mcnemar.csv").iloc[0]
out.append(f"#let final-mcnemar = (c: {int(fm.convnext_only)}, r: {int(fm.resnet_only)}, p: \"{fm.p:.1e}\")\n")

# ---- A4: worst classes (both final models) and top confusions (ConvNeXt-Tiny)
wc = pd.read_csv(EA / "convnext_tiny__tl_L6_hires__full" / "per_class_accuracy.csv").set_index("class")
wr = pd.read_csv(EA / "resnet50.tv2_in1k__tl_L6_hires_strongaug__full" / "per_class_accuracy.csv").set_index("class")
s = "#let worst = (\n"
for (cn, c), (rn, r) in zip(wc.head(10).iterrows(), wr.head(10).iterrows()):
    s += row(cn, f"{100 * c.test_acc:.0f} ({int(c.n_test)})", rn, f"{100 * r.test_acc:.0f} ({int(r.n_test)})")
out.append(s + ")\n")
tc = pd.read_csv(EA / "convnext_tiny__tl_L6_hires__full" / "top_confusions.csv").head(10)
s = "#let confusions = (\n"
for _, t in tc.iterrows():
    s += row(t["true"], t.predicted, int(t["count"]))
out.append(s + ")\n")

# ---- A.x: test-time augmentation (flip) for every ladder run: decided on validation
res_all = pd.read_csv(ROOT / "outputs" / "results.csv", index_col="run")
pct = lambda v: "--" if np.isnan(v) else f"{100 * v:.2f}"
s = "#let tta = (\n"
for step, rc, cc in steps:
    for model, pre, cfg in (("ResNet-50", R, rc), ("ConvNeXt-Tiny", X, cc)):
        if cfg is None:
            continue
        a, b = res_all.loc[pre + cfg], res_all.loc[pre + cfg + " +TTA"]
        s += row(step, model, pct(a.val_top1), pct(b.val_top1), pct(a.test_top1), pct(b.test_top1))
out.append(s + ")\n")

# ---- A.x: early exploration before the ladder (two fixed recipes on four pretrained models)
early = [("ResNet-50", "ImageNet-1k", "resnet50.tv2_in1k"), ("ConvNeXt-Tiny", "ImageNet-1k", "convnext_tiny"),
         ("BiT ResNet-50x1", "ImageNet-21k", "resnetv2_50x1_bit.goog_in21k"),
         ("ConvNeXt-Base", "ImageNet-22k", "convnext_base.fb_in22k")]
s = "#let early = (\n"
for name, data_, m in early:
    for cfg, label in (("A_standard", "A: 224 px"), ("B_finegrained", "B: 224→448 px + CutMix")):
        r_ = res[res.index == f"{m}__{cfg}"].iloc[0]
        s += row(name, data_, f"{r_.params_M:.1f} M", label, pct(r_.val_top1), pct(r_.test_top1),
                 pct(r_.test_avg_per_class))
out.append(s + ")\n")

(DATA / "tables.typ").write_text("// generated by report/make_appendix_tables.py, do not edit\n" + "".join(out))
print("wrote", DATA / "tables.typ")
