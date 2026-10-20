"""Figures for the report. Every function saves a PNG and closes the figure (no display needed)."""
import matplotlib

matplotlib.use("Agg")                     # render to files, works without a screen
import matplotlib.pyplot as plt
from PIL import Image


def plot_history(hist, title, path):
    """Loss and accuracy curves; dashed lines mark a new phase (resolution change)."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    epochs = range(1, len(hist["val_acc"]) + 1)
    for a, key in zip(ax, ["loss", "acc"]):
        a.plot(epochs, hist[f"train_{key}"], label="train")
        a.plot(epochs, hist[f"val_{key}"], label="val")
        for s in hist["phase_starts"][1:]:
            a.axvline(s + 0.5, ls="--", c="grey", lw=1)
        a.set_title(f"{title}: {key}")
        a.set_xlabel("epoch")
        a.legend()
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_val_curves(histories, path):
    """Validation accuracy per epoch for every run on one chart."""
    fig = plt.figure(figsize=(8, 4.5))
    for name, h in histories.items():
        plt.plot(range(1, len(h["val_acc"]) + 1), h["val_acc"], label=name)
    plt.xlabel("epoch")
    plt.ylabel("validation accuracy")
    plt.title("Validation accuracy per epoch")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_run_comparison(histories, path, labels=None):
    """Side-by-side comparison of runs (e.g. ladder steps): train vs val accuracy, and val loss.

    Solid line = validation, dashed = training (same colour per run)."""
    labels = labels or list(histories)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for (name, h), label in zip(histories.items(), labels):
        epochs = range(1, len(h["val_acc"]) + 1)
        line, = ax[0].plot(epochs, h["val_acc"], label=f"{label} val")
        ax[0].plot(epochs, h["train_acc"], ls="--", c=line.get_color(), alpha=0.7, label=f"{label} train")
        ax[1].plot(epochs, h["val_loss"], c=line.get_color(), label=label)
    ax[0].set_title("accuracy (dashed = train, solid = val)")
    ax[1].set_title("validation loss")
    for a in ax:
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_ladder(steps, series, path, title="Test Top-1 accuracy per ladder step", reference=None, zoom=(60, 95)):
    """Accuracy per tuning step for one or more models (the report's "ladder" figure).

    steps: list of step labels; series: {model label: list of accuracies (None = not run)};
    reference: optional (label, value) drawn as a dashed horizontal line.
    Two panels: the full 0-100% range (shows collapses) and a zoom on `zoom` (shows the small steps)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1, 1.4]})
    x = list(range(len(steps)))
    for ax, (lo, hi) in zip(axes, [(0, 100), zoom]):
        for k, (label, ys) in enumerate(series.items()):
            pts = [(i, 100 * y) for i, y in zip(x, ys) if y is not None]
            line, = ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=label)
            if (lo, hi) == zoom:                 # value labels on the zoomed panel only
                for i, y in pts:
                    if lo <= y <= hi:
                        ax.annotate(f"{y:.1f}", (i, y), textcoords="offset points",
                                    xytext=(0, 8 if k % 2 else -14), ha="center", fontsize=8,
                                    color=line.get_color())
        if reference:
            ax.axhline(100 * reference[1], ls="--", c="grey", lw=1)
            ax.text(len(steps) - 1, 100 * reference[1] - 1.8, reference[0], ha="right", fontsize=8, color="grey")
        ax.set_ylim(lo, hi)
        ax.set_xticks(x)
        ax.set_xticklabels(steps, fontsize=7.5)
        ax.set_ylabel("test Top-1 accuracy (%)")
        ax.grid(alpha=0.3)
    axes[0].set_title("full range")
    axes[1].set_title(f"zoom {zoom[0]}-{zoom[1]}%")
    axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_misclassified(items, path, title=None):
    """Misclassified test images (top row) next to a training image of the species they were mistaken for
    (bottom row). items: list of dicts with test_path, true, pred, conf, pred_example_path."""
    n = len(items)
    fig, ax = plt.subplots(2, n, figsize=(2.6 * n, 5.6), squeeze=False)
    for c, it in enumerate(items):
        ax[0, c].imshow(Image.open(it["test_path"]).convert("RGB"))
        ax[0, c].set_title(f"test image\ntrue: {it['true']}", fontsize=8)
        ax[1, c].imshow(Image.open(it["pred_example_path"]).convert("RGB"))
        ax[1, c].set_title(f"predicted ({it['conf']:.0%}):\n{it['pred']}\n(training example)", fontsize=8)
        for r in (0, 1):
            ax[r, c].axis("off")
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_per_class_hist(per_class, title, path):
    """Distribution of per-class test accuracy: are errors spread out or concentrated?"""
    fig = plt.figure(figsize=(6, 3))
    plt.hist(per_class, bins=20)
    plt.xlabel("test accuracy of a class")
    plt.ylabel("number of classes")
    plt.title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confused_pairs(pairs, example_paths, class_names, path):
    """One training image of the true and the predicted species for each confused pair."""
    fig, ax = plt.subplots(len(pairs), 2, figsize=(6, 3 * len(pairs)), squeeze=False)
    for r, (t, p) in enumerate(pairs):
        for c, (cls, tag) in enumerate([(t, "true"), (p, "predicted")]):
            ax[r, c].imshow(Image.open(example_paths[cls]).convert("RGB"))
            ax[r, c].set_title(f"{tag}: {class_names[cls]}", fontsize=9)
            ax[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
