"""Evaluation metrics required by the brief."""
import numpy as np


def compute_metrics(y_true, y_pred, num_classes):
    """Returns (Top-1 accuracy, average accuracy per class, per-class accuracy array).

    Top-1 = (1/N) * sum 1{argmax(y) == ground truth}
    Ave   = (1/C) * sum_i T_i, where T_i is the accuracy on the test images of class i
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    top1 = float((y_true == y_pred).mean())          # share of all test images classified correctly
    per_class = np.full(num_classes, np.nan)         # NaN = class has no test image (not the case here)
    for c in range(num_classes):
        mask = y_true == c                           # the test images whose true class is c
        if mask.any():
            per_class[c] = (y_pred[mask] == c).mean()   # T_i: accuracy on class c alone
    # Ave: every class counts equally, so a few large easy classes cannot hide weak small ones
    return top1, float(np.nanmean(per_class)), per_class
