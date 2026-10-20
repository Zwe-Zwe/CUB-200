"""Batch-level augmentation."""
import math

import numpy as np
import torch


def cutmix(imgs, labels, alpha=1.0):
    """CutMix (Yun et al., 2019): paste a random box from another image of the batch into each image.

    The loss is later mixed in proportion to the visible area, so the model cannot rely on a single
    small region of the bird (a regulariser against overfitting).

    Returns (mixed images, original labels, pasted labels, lam = fraction of the original image kept).
    """
    lam = np.random.beta(alpha, alpha)                     # target mixing ratio
    perm = torch.randperm(imgs.size(0), device=imgs.device)
    H, W = imgs.shape[2], imgs.shape[3]
    cut_h, cut_w = int(H * math.sqrt(1 - lam)), int(W * math.sqrt(1 - lam))
    cy, cx = np.random.randint(H), np.random.randint(W)    # box centre
    y1, y2 = int(np.clip(cy - cut_h // 2, 0, H)), int(np.clip(cy + cut_h // 2, 0, H))
    x1, x2 = int(np.clip(cx - cut_w // 2, 0, W)), int(np.clip(cx + cut_w // 2, 0, W))
    imgs = imgs.clone()
    imgs[:, :, y1:y2, x1:x2] = imgs[perm, :, y1:y2, x1:x2]
    lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)              # exact ratio after clipping to the image
    return imgs, labels, labels[perm], lam
