# EXP-09: Track 1, S5: + weight decay

**Date:** 2026-10-15 (trained 17:07 to 17:35) · **Run:** [scratch_vgg_gap__scratch_wd](runs/scratch_vgg_gap__scratch_wd.md)
· **Command:** `python -m birdcls.train --model scratch_vgg_gap --config scratch_wd`
· **Comparison figure:** [`outputs/figures/track1_S4_vs_S5.png`](../outputs/figures/track1_S4_vs_S5.png)

## Question and hypothesis (written before training)

**One change vs S4:** `weight_decay` 0 → **5e-4** (`configs/scratch_wd.yaml`). Same model (`scratch_vgg_gap`, 2.41 M
parameters), same SGD momentum 0.9, peak LR 0.01 with 2-epoch warm-up + cosine decay over up to 150 epochs, early
stopping patience 20, basic augmentation, no label smoothing / dropout, same seed.

What weight decay does: at every SGD step each weight is also pulled towards zero,
`w ← w − lr · (∂L/∂w + λ·w)` with λ = 5e-4, which is equivalent to adding `λ/2 · ‖w‖²` to the loss (L2 regularisation).
Weights can only stay large if they keep reducing the loss across many training images, which favours smoother,
simpler functions that memorise less. 5e-4 is the standard value for SGD-trained CNNs (VGG, ResNet recipes). It is
applied to all parameters here (convolutions, BatchNorm γ/β and the linear head).

Why now: S4 fits the training set almost perfectly (clean train 99.8% vs val 46.9%, gap 52.9 pp) with **no explicit
regularisation at all**, but its val loss never rose, so it can afford more regularisation without under-fitting.

Hypothesis:
- **Smaller gap** than S4 (52.9 pp): clean train accuracy noticeably below 99.8%.
- **Higher val/test accuracy** than S4 (46.3% test), by a few pp (size uncertain).
- Possible extra effect via BatchNorm: weight decay keeps conv weights small, so the *effective* learning rate stays
  higher late in training, which can help the network keep improving in the last epochs.
- Early stopping may trigger later or not at all.

## Results

| | S4 (weight decay 0) | **S5 (weight decay 5e-4)** | Change |
|---|---|---|---|
| Epochs trained | 150 (no early stop) | 150 (no early stop) | |
| Best val Top-1 | 46.89% (ep 143) | **48.03%** (ep 139) | +1.1 pp |
| Mean val Top-1, last 10 epochs | 45.91% | **47.04%** | +1.1 pp (less sensitive to the single best epoch) |
| **Test Top-1** | 46.26% | **48.09%** | **+1.8 pp** |
| **Test avg per class** | 45.31% | **47.42%** | +2.1 pp |
| With TTA: val / test | 46.58 / 47.34% | 48.03 / 47.92% | TTA off (ladder rule) |
| Clean train accuracy | 99.82% | 99.92% | unchanged |
| **Overfitting gap** | 52.9 pp | **51.9 pp** | −1.0 pp, essentially unchanged |
| Min val loss (epoch) | 2.68 (148) | **2.39 (125)** | **−0.29** |
| Final-layer weight norm ‖W‖ | 30.32 | 26.43 | −13% |
| Test mean confidence (correct / wrong predictions) | 0.789 / 0.508 | 0.763 / **0.463** | less confident when wrong |
| Training time | 26.2 min | 27.8 min | |

### Paired significance test (McNemar, same images, exact binomial)

| Split | Both right | Only S4 right | Only S5 right | Both wrong | p-value |
|---|---|---|---|---|---|
| Validation (966) | 376 | 77 | 88 | 425 | 0.44 |
| Test (1,204) | 473 | 84 | **106** | 541 | **0.13** |

S5 fixes 106 test images that S4 gets wrong, but breaks 84 that S4 gets right. The net +22 images (+1.8 pp) points in
S5's favour on both splits, but **p = 0.13 is not significant** at the usual 0.05 level with a single seed.

| Epoch | 1 | 25 | 50 | 75 | 100 | 110 | 125 | 139 | 150 |
|---|---|---|---|---|---|---|---|---|---|
| S4 val acc | .011 | .168 | .294 | .385 | .450 | .443 | .451 | .453 | .453 |
| S5 val acc | .012 | .149 | .310 | .376 | .442 | .459 | .464 | **.480** | .471 |
| S4 val loss | 5.25 | 3.82 | 3.35 | 2.88 | 2.76 | 2.75 | 2.69 | 2.70 | 2.69 |
| S5 val loss | 5.25 | 3.68 | 2.99 | 2.85 | 2.58 | 2.49 | **2.39** | 2.43 | 2.42 |
| S4 train acc | .004 | .221 | .523 | .792 | .923 | .943 | .957 | .968 | .958 |
| S5 train acc | .004 | .222 | .498 | .796 | .930 | .952 | .971 | .975 | .968 |

## Observations

1. **Hypothesis mostly refuted: the gap did not shrink.** Train accuracy is the same (99.9%) and the gap is
   essentially unchanged (51.9 vs 52.9 pp). At this strength, weight decay did not stop memorisation.
2. **Accuracy improved slightly and consistently, but not significantly:** +1.1 pp val (best and last-10 mean),
   +1.8 pp test Top-1, +2.1 pp per-class average. The same direction on every measure, but McNemar p = 0.13 (test), 0.44 (val).
   Report as a *small, likely real but unconfirmed* gain. More seeds would be needed to confirm it.
3. **The clear effect is on calibration:** val loss drops from 2.68 to 2.39 (−11%), and the gap between the curves opens up
   after epoch 60. The final layer's weights are 13% smaller, so its logits are smaller and the network is less confident,
   especially on its mistakes (mean confidence on wrong test predictions 0.51 → 0.46). Being less confidently wrong
   lowers the loss a lot while changing few top-1 decisions.
4. **Why weight decay does little against memorisation here: BatchNorm.** Every convolution is followed by BN, which
   makes the network's output invariant to the scale of the conv weights. Shrinking them does not reduce what the network
   can represent; it mainly raises their effective learning rate. Only the final linear layer (after global average
   pooling, no BN) is truly regularised by the decay, which matches observation 3. This is a known interaction
   (van Laarhoven, 2017, "L2 regularization versus batch and weight normalization").
5. Curves for the first ~60 epochs are almost identical. The effect builds up as the LR decays and the network starts to fit
   the training data.

## Conclusion

Weight decay 5e-4 gives a small, consistent gain (+1.8 pp test, 48.1%) and clearly better calibration, but **does not
reduce overfitting**, because BatchNorm makes the conv layers scale-invariant. The ladder so far:
4.9 → 14.4 → 25.4 → 30.8 → 46.3 → **48.1%**. To cut the 52 pp gap we need regularisers that act on the *data or targets*
rather than on weight size: **stronger augmentation / Mixup / CutMix** (augmentation gave +11 pp in S2) and **label
smoothing**. Keep weight decay 5e-4 in all later steps (standard, and it helps calibration).
