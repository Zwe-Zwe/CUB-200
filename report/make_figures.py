"""Figures for the report (report/figures/), drawn from outputs/ (black and white, print-friendly).

    python report/make_figures.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

res = pd.read_csv(ROOT / "outputs" / "results.csv", index_col="run")
res = res[~res.tta]
test = lambda run: 100 * res.loc[run, "test_top1"]

# Colour pair validated for colour-blind readers; models also differ by line style and marker (readable in print)
BLUE, ORANGE = "#2a78d6", "#eb6834"                  # ResNet-50, ConvNeXt-Tiny
K, G, LG = "#000000", "#7a7a7a", "#d9d9d9"          # text, muted elements, grid
CNX = {"color": ORANGE, "linestyle": "-", "marker": "o"}
RES = {"color": BLUE, "linestyle": "--", "marker": "s"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": "#444444",
                     "axes.labelcolor": K, "xtick.color": "#444444", "ytick.color": "#444444",
                     "axes.spines.top": False, "axes.spines.right": False})


def history(run):
    return json.load(open(ROOT / "outputs" / "runs" / run / "history.json"))


# ---------------------------------------------------------------- Figure 1: the tuning ladder
# Test Top-1 accuracy after every ladder step L0..L7, one line per model (report Figure 1, slide 4).
# ConvNeXt-Tiny's L1 collapse (4.4%) is far below the y-axis, so it is drawn as a dotted dip to the axis edge.
steps = ["L0\nprobe", "L1\nFT 1e-3", "L2\nLR policy", "L3\nregul.", "L4\nstrong aug", "L5\n40 ep",
         "L6\n448 px", "L7\n+val data"]
resnet = ["tl_L0_probe", "tl_L1_finetune", "tl_L2_discriminative", "tl_L3_regularise", "tl_L4_strongaug",
          "tl_L5_longer_strongaug", "tl_L6_hires_strongaug", "tl_L6_hires_strongaug__full"]
convnext = ["tl_L0_probe", "tl_L1_finetune", "tl_L2_discriminative", "tl_L3_regularise", "tl_L4_strongaug",
            "tl_L5_longer", "tl_L6_hires", "tl_L6_hires__full"]
r = [test(f"resnet50.tv2_in1k__{x}") for x in resnet]
c = [test(f"convnext_tiny__{x}") for x in convnext]

fig, ax = plt.subplots(figsize=(5.6, 2.45))
xs = list(range(len(steps)))
ax.axhline(93.69, color=G, linestyle=":", linewidth=0.9)
ax.text(0, 94.2, "ConvNeXt-Base, ImageNet-22k reference (untuned): 93.7", fontsize=6, color="#444444", va="bottom")
ax.plot(xs, r, linewidth=1.6, markersize=4.2, label="ResNet-50 (ImageNet-1k)", zorder=3, **RES)
cx = list(range(2, len(xs)))                         # ConvNeXt L1 (4.4 %) is far below the axis
ax.plot(cx, [c[i] for i in cx], linewidth=1.8, markersize=4.5, label="ConvNeXt-Tiny (ImageNet-1k)", zorder=3,
        **CNX)
ax.plot([0], [c[0]], color=ORANGE, marker="o", markersize=4.5, zorder=3)
ax.plot([0, 1, 2], [c[0], 55, c[2]], color=ORANGE, linewidth=0.8, linestyle=(0, (1, 2)), zorder=2, clip_on=False)
ax.text(1.1, 55.5, f"ConvNeXt L1 = {c[1]:.1f} (collapse)", ha="left", va="bottom", fontsize=6)
ax.plot([4], [c[4]], marker="o", markersize=6.5, markerfacecolor="white", markeredgecolor=ORANGE, zorder=4)
ax.text(4, c[4] + 1.0, "reverted", ha="center", va="bottom", fontsize=6, color="#444444")
for i in (0, 2, 6, 7):
    ax.text(i, c[i] + 1.0, f"{c[i]:.1f}", ha="center", va="bottom", fontsize=6.5)
    if i == 2:
        ax.text(i + 0.1, r[i] - 0.8, f"{r[i]:.1f}", ha="left", va="top", fontsize=6.5)
    else:
        ax.text(i, r[i] - 1.2, f"{r[i]:.1f}", ha="center", va="top", fontsize=6.5)
ax.text(0.88, r[1], f"{r[1]:.1f}", ha="right", va="center", fontsize=6.5)
ax.set_xticks(xs, steps, fontsize=6)
ax.set_ylim(55, 97); ax.set_ylabel("test Top-1 (%)")
ax.grid(axis="y", color=LG, linewidth=0.6); ax.set_axisbelow(True)
ax.legend(loc="lower right", fontsize=6.5, frameon=False)
fig.tight_layout(pad=0.4)
fig.savefig(OUT / "ladder.png", dpi=300)
plt.close(fig)

# ---------------------------------------------------------------- Figure 2: learning curves (overfitting)
# Training accuracy (on augmented batches, grey dashed) vs validation accuracy (model colour) per epoch for three
# runs, to show overfitting with and without regularisation; the dotted line marks the switch to 448 px.
curves = [("(a) ConvNeXt-Tiny, L2 (no regularisation)", "convnext_tiny__tl_L2_discriminative"),
          ("(b) ConvNeXt-Tiny, L6 (final recipe)", "convnext_tiny__tl_L6_hires"),
          ("(c) ResNet-50, L6 (final recipe)", "resnet50.tv2_in1k__tl_L6_hires_strongaug")]
fig, axes = plt.subplots(1, 3, figsize=(7.1, 1.85))
for ax, (title, run) in zip(axes, curves):
    h = history(run)
    ep = range(1, len(h["train_acc"]) + 1)
    col = BLUE if run.startswith("resnet") else ORANGE     # validation in the model's colour, train in grey
    ax.plot(ep, [100 * v for v in h["train_acc"]], color=G, linestyle="--", linewidth=1.4,
            label="train (augmented batches)")
    ax.plot(ep, [100 * v for v in h["val_acc"]], color=col, linewidth=1.6, label="validation")
    for s in h["phase_starts"][1:]:
        ax.axvline(s + 0.5, color=G, linestyle=":", linewidth=0.9)
        ax.text(s + 1.2, 35, "448 px", fontsize=6, color="#444444")
    v = 100 * h["val_acc"][-1]
    ax.text(len(ep) * 0.55, v - 5, f"final val {v:.1f}", ha="center", va="top", fontsize=6.3)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_title(title, fontsize=7.5, loc="left")
    ax.set_ylim(30, 103); ax.set_xlabel("epoch", fontsize=7)
    ax.grid(axis="y", color=LG, linewidth=0.6); ax.set_axisbelow(True)
    ax.tick_params(labelsize=6.5)
axes[0].set_ylabel("accuracy (%)", fontsize=7)
axes[0].legend(loc="lower right", fontsize=6.3, frameon=False)
fig.tight_layout(pad=0.4, w_pad=0.8)
fig.savefig(OUT / "curves.png", dpi=300)
plt.close(fig)

# ---------------------------------------------------------------- Appendix: per-class accuracy histogram
# How many of the 200 species fall in each 10-point accuracy band, for both final models.
EA = ROOT / "outputs" / "error_analysis"
pc_c = pd.read_csv(EA / "convnext_tiny__tl_L6_hires__full" / "per_class_accuracy.csv").test_acc * 100
pc_r = pd.read_csv(EA / "resnet50.tv2_in1k__tl_L6_hires_strongaug__full" / "per_class_accuracy.csv").test_acc * 100
bins = np.arange(0, 101, 10)
hc, _ = np.histogram(pc_c, bins=bins); hr, _ = np.histogram(pc_r, bins=bins)
fig, ax = plt.subplots(figsize=(5.2, 2.0))
w = 3.8
ax.bar(bins[:-1] + 5 - w / 2 - 0.2, hr, width=w, color=BLUE, label="ResNet-50 final")
ax.bar(bins[:-1] + 5 + w / 2 + 0.2, hc, width=w, color=ORANGE, label="ConvNeXt-Tiny final")
ax.set_xticks(bins[:-1] + 5, [f"{a}-{a + 10}" for a in bins[:-1]], fontsize=6.5)
ax.set_xlabel("test accuracy of the species (%)"); ax.set_ylabel("number of species")
ax.grid(axis="y", color=LG, linewidth=0.6); ax.set_axisbelow(True)
ax.legend(loc="upper left", fontsize=6.5, frameon=False)
fig.tight_layout(pad=0.4)
fig.savefig(OUT / "per_class_hist.png", dpi=300)
plt.close(fig)

# ---------------------------------------------------------------- Appendix: learning-rate schedule actually used
# The backbone LR logged after every epoch of the final ConvNeXt-Tiny recipe (warm-up + cosine, then phase 2).
h = history("convnext_tiny__tl_L6_hires")
fig, ax = plt.subplots(figsize=(4.2, 1.7))
ax.plot(range(1, len(h["lr_backbone"]) + 1), h["lr_backbone"], color=ORANGE, marker="o", markersize=2.5, linewidth=1.2)
ax.axvline(30.5, color=G, linestyle=":", linewidth=0.9)
ax.text(31.2, 8e-5, "448 px\nphase", fontsize=6, color="#444444")
ax.set_xlabel("epoch"); ax.set_ylabel("backbone LR")
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax.grid(axis="y", color=LG, linewidth=0.6); ax.set_axisbelow(True)
fig.tight_layout(pad=0.4)
fig.savefig(OUT / "lr_schedule.png", dpi=300)
plt.close(fig)
print("wrote figures to", OUT)
