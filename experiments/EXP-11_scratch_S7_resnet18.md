# EXP-11: Track 1, S7: residual network (ResNet-18 from scratch)

**Date:** 2026-10-15 (first attempt 18:29, interrupted; re-run 18:54 to 19:14) · **Run:** [scratch_resnet18__scratch_strongaug](runs/scratch_resnet18__scratch_strongaug.md)
· **Command:** `python -m birdcls.train --model scratch_resnet18 --config scratch_strongaug`
· **Comparison figure:** [`outputs/figures/track1_S6_vs_S7.png`](../outputs/figures/track1_S6_vs_S7.png)

## Question and hypothesis (written before training)

**Change vs S6: the architecture only.** Same training recipe (`scratch_strongaug`: strong augmentation, SGD momentum
0.9, peak LR 0.01, 2-epoch warm-up + cosine over up to 150 epochs, early stopping patience 20, weight decay 5e-4, no label
smoothing / dropout, same seed). Model `scratch_vgg_gap` → **`scratch_resnet18`** (`birdcls/scratch.py`, written from
scratch; its parameter count matches torchvision's ResNet-18 exactly, 11.28 M with a 200-class head).

This step changes the architecture as a package (residual connections + depth + width), so it tests the hypothesis from
EXP-10 that **capacity is now the bottleneck**. It does not isolate the effect of the skip connections alone.

| | S6 `scratch_vgg_gap` | **S7 `scratch_resnet18`** |
|---|---|---|
| Stem | none (first conv at full 224 × 224) | conv 7×7 stride 2 + BN + ReLU + maxpool, so 224 → 56 |
| Body | 5 VGG blocks (2 × conv3×3-BN-ReLU + maxpool) | 4 stages × 2 **residual BasicBlocks** (conv3×3-BN-ReLU-conv3×3-BN, plus skip) |
| Conv layers | 10 | 20 (17 main path + 3 1×1 shortcut projections) |
| Channels | 32-64-128-256-256 | 64-128-256-512 |
| Head | GAP → linear 256→200 | GAP → linear 512→200 |
| Parameters | 2.41 M | **11.28 M** (4.7×) |
| Training speed (RTX 4060, AMP, bs 32) | 2.19 ms/img | **1.01 ms/img** (early downsampling) |
| Peak memory (bs 32) | 1.04 GiB | 0.52 GiB |
| Init details | Kaiming | Kaiming + **zero-init residual** (last BN γ = 0 in each block, so every block starts as identity) |

Why residual connections: each block outputs `ReLU(F(x) + x)`, so it only learns a correction to its input, and the
identity path passes signal and gradient through unchanged. This is what lets ResNets be much deeper than VGG-style
networks without degrading (He et al., 2016).

Hypothesis:
- With strong augmentation keeping memorisation in check (S6 aug train acc 72%), the extra capacity should turn into
  **higher val/test accuracy** than S6 (48.3% test), possibly by several pp.
- Training accuracy (augmented) will be higher than S6's 72% (more capacity to fit the hard training data).
- Risk: 11.3 M parameters on 3,863 images could overfit again, i.e. the train/val gap grows back.

## Results

**Run history:** the first attempt was stopped by the user after epoch 61 (moving location). It was re-run from scratch
with the same command and seed; its log is kept as `outputs/runs/scratch_resnet18__scratch_strongaug/train_attempt1_interrupted.log`.
**Reproducibility check: all 61 logged epochs of the interrupted attempt are identical** (train/val loss and accuracy to
3 decimals) to the re-run, so seeded training is deterministic.

| | S6 VGG-GAP | **S7 ResNet-18** | Change |
|---|---|---|---|
| Parameters | 2.41 M | 11.28 M | 4.7× |
| Compute per 224 px image | **2.82 GMACs** | 1.81 GMACs | −36% |
| Compute at ≥ 112 px resolution | **1.20 GMACs** (blocks 1 and 2) | 0.12 GMACs (stem only) | 10× less |
| Epochs trained | 150 (no early stop) | 150 (no early stop) | |
| Best val Top-1 | **49.90%** (ep 135) | 41.61% (ep 149) | **−8.3 pp** |
| Mean val Top-1, last 10 epochs | **48.85%** | 40.77% | −8.1 pp |
| **Test Top-1** | **48.26%** | 41.36% | **−6.9 pp** |
| **Test avg per class** | **46.90%** | 40.64% | −6.3 pp |
| With TTA: val / test | 49.28 / 49.42% | 40.89 / 41.61% | TTA off (ladder rule) |
| Augmented train acc, epoch 150 | **71.8%** | 63.4% | ResNet fits the training data *less* |
| Train loss, epoch 150 | **1.27** | 1.65 | |
| Clean train accuracy | 96.71% | 93.17% | |
| Overfitting gap | 46.8 pp | 51.6 pp | |
| Min val loss (epoch) | **2.14** (148) | 2.58 (148) | |
| Training time | 26.2 min | **19.8 min** | faster |

### Paired significance test (McNemar, exact)

| Split | Both right | Only S6 right | Only S7 right | Both wrong | p-value |
|---|---|---|---|---|---|
| Validation (966) | 346 | 136 | 56 | 428 | **7.3e-9** |
| Test (1,204) | 425 | 156 | 73 | 550 | **4.3e-8** |

S7 is **significantly worse** than S6 on both splits.

| Epoch | 1 | 25 | 50 | 75 | 100 | 110 | 125 | 135 | 149 | 150 |
|---|---|---|---|---|---|---|---|---|---|---|
| S6 val acc | .012 | .133 | .243 | .344 | .386 | .453 | .468 | .499 | .492 | .489 |
| S7 val acc | .011 | .080 | .175 | .242 | .331 | .365 | .384 | .402 | **.416** | .405 |
| S6 train acc (aug) | .004 | .127 | .266 | .415 | .564 | .620 | .683 | .692 | .699 | .718 |
| S7 train acc (aug) | .008 | .089 | .189 | .321 | .445 | .517 | .567 | .619 | .610 | .634 |
| S6 val loss | 5.26 | 4.04 | 3.26 | 2.77 | 2.55 | 2.29 | 2.21 | 2.16 | 2.14 | 2.15 |
| S7 val loss | 5.27 | 4.47 | 3.75 | 3.38 | 2.93 | 2.88 | 2.70 | 2.62 | 2.58 | 2.58 |

## Observations

1. **Hypothesis refuted:** 4.7× more parameters and residual connections made the network **worse** (−6.9 pp test,
   p = 4e-8), not better.
2. **More parameters did not mean more fitting power here:** ResNet-18 fits the (augmented) training data *less* than the
   small VGG (63% vs 72% train accuracy, higher train loss) and is behind at every epoch. So EXP-10's diagnosis
   "capacity-limited" was too simple: what matters is *how* the capacity is spent, not the parameter count.
3. **Likely main reason: early downsampling loses fine detail.** The ResNet stem (7×7 conv stride 2 + max-pool) reduces
   224 px to 56 px before any residual block, and spends only 0.12 GMACs above 56 px resolution. The VGG-GAP network
   keeps full 224 px and 112 px resolution for its first 4 convolutions (1.20 GMACs, 10× more) and has 1.56× more compute
   overall. For fine-grained birds, distinguishing features (eye rings, wing bars, bill shape) are small, so processing
   at high resolution plausibly matters more than depth or width. (This is a hypothesis, not tested in isolation; a ResNet
   with a gentler stem would test it.)
4. **Still improving at the end:** best val at epoch 149 and val loss lowest at 148, so the bigger network is also
   under-trained at 150 epochs with peak LR 0.01. It would likely need more epochs or a higher LR. With zero-init
   residual blocks it also starts slowly (val 8.0% at epoch 25 vs 13.3% for S6).
5. Residual connections were not the limiting factor: training is stable and smooth (no divergence); the problem is
   accuracy, not trainability.

## Conclusion

Swapping in a standard ImageNet-style ResNet-18 (11.3 M params) **lowered** accuracy to 41.4% test. On this small
fine-grained dataset, at this training budget, a small network that keeps high-resolution feature maps beats a larger one
that downsamples aggressively. **S6 (VGG-GAP + strong augmentation, 48.26% test) remains the best from-scratch model.**
This is a useful negative result for the report: architecture choices must fit the task (fine detail) and the data budget,
and "bigger / standard" is not automatically better.
