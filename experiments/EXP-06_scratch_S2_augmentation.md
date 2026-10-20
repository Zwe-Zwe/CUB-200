# EXP-06: Track 1, S2: + data augmentation

**Date:** 2026-10-13 (trained 23:39 to 23:48) · **Run:** [scratch_vgg_bn__scratch_aug](runs/scratch_vgg_bn__scratch_aug.md)
· **Command:** `python -m birdcls.train --model scratch_vgg_bn --config scratch_aug`
· **Comparison figure:** [`outputs/figures/track1_S0_S1_S2.png`](../outputs/figures/track1_S0_S1_S2.png)

## Question and hypothesis (written before training)

**One change vs S1:** training-image augmentation `none` → `basic` (`configs/scratch_aug.yaml`). Same model
(`scratch_vgg_bn`, 8,879,528 parameters, architecture in EXP-05), same SGD momentum 0.9 / constant LR 0.01, 50 epochs,
no regularisation, same seed.

`basic` augmentation (applied freshly every time an image is loaded, training set only):

| Transform | Setting | What it teaches the network |
|---|---|---|
| RandomResizedCrop | scale 0.6 to 1.0 of the image, output 224 px | birds appear at different sizes and positions |
| RandomHorizontalFlip | p 0.5 | a bird facing left is the same species facing right |
| RandomRotation | ±15° | small camera tilt does not change the species |
| ColorJitter | brightness / contrast / saturation ±0.2 | lighting changes; rely on pattern, not exact colour level |
| RandomErasing | p 0.25 | parts of the bird can be hidden (branches, leaves) |

Hypothesis:
- **Overfitting drops sharply:** the network never sees exactly the same image twice, so memorising the ~19 images per
  class becomes much harder. Expect clean train accuracy well below 100% within 50 epochs and a much smaller gap.
- **Validation accuracy rises substantially** (expected to be the biggest single gain of the ladder).
- **Slower convergence:** the task is harder, so val may still be rising at epoch 50 (that is the target of S5,
  longer training with an LR schedule).
- The FC head (6.5 M parameters) is still there, so some overfitting will remain (target of S3).

## Results

| | S0 (plain) | S1 (+ BN) | **S2 (+ augmentation)** | S2 vs S1 |
|---|---|---|---|---|
| Best val Top-1 | 5.69% (ep 43) | 15.11% (ep 38) | **25.88%** (ep 44) | **+10.8 pp** |
| **Test Top-1** | 4.90% | 14.37% | **25.42%** | **+11.1 pp (1.8×)** |
| **Test avg per class** | 4.78% | 13.81% | **24.62%** | +10.8 pp |
| With TTA: val / test | 5.90 / 4.98% | 14.39 / 16.28% | 25.67 / 26.16% | TTA lowers val (−0.2), so off |
| Clean train accuracy | 98.58% | 99.97% | **77.12%** | no longer memorised |
| Augmented train acc, epoch 50 | n/a | n/a | 67.0% | |
| **Overfitting gap** (clean train − val) | 92.9 pp | 84.9 pp | **51.2 pp** | **−33.7 pp** |
| Min val loss (epoch) | 4.96 (7) | 4.74 (7) | **3.75 (34)** | overfitting onset delayed by 27 epochs |
| Val loss at epoch 50 | 13.79 | 8.87 | 4.56 | |
| Training time | 8.1 min | 9.1 min | 9.1 min | same cost |

Test standard error at about 25% is about ±1.25 pp, so +11 pp is far beyond noise.

| Epoch | 1 | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 | 50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S2 train acc (augmented) | .008 | .021 | .050 | .105 | .172 | .238 | .318 | .405 | .500 | .588 | .670 |
| S2 val acc | .008 | .012 | .062 | .089 | .137 | .137 | .197 | .196 | .210 | .228 | .243 |
| S2 val loss | 5.22 | 4.96 | 4.57 | 4.21 | 3.99 | 4.09 | 3.78 | 4.17 | 3.92 | 4.14 | 4.56 |
| S1 val acc (reference) | .010 | .048 | .079 | .095 | .127 | .136 | .148 | .144 | .144 | .142 | .148 |
| S1 val loss (reference) | 5.18 | 4.85 | 5.08 | 7.21 | 8.15 | 8.44 | 8.70 | 8.74 | 8.78 | 8.81 | 8.87 |

## Observations

1. **Hypothesis confirmed: the biggest gain so far.** +11.1 pp test (14.4% → 25.4%), larger than BatchNorm's +9.5 pp, at
   zero extra training cost.
2. **Memorisation is broken:** clean train accuracy is 77% instead of ~100%, and the gap shrinks from 85 to 51 pp. For the
   first 25 epochs val accuracy even matches or beats training accuracy (training images are harder because they are
   augmented).
3. **Overfitting is delayed, not removed:** val loss keeps falling until epoch 34 (S0/S1: epoch 7), then rises again
   (3.75 → 4.56 at epoch 50) while train accuracy keeps climbing. With more epochs memorisation would slowly return.
   The 6.5 M-parameter FC head is the remaining capacity to memorise (target of S3).
4. **Slower, noisier convergence, as predicted:** val accuracy oscillates between 22 and 26% over the last 10 epochs.
   A constant LR of 0.01 keeps the weights bouncing around; a decaying LR schedule should let them settle (target of S5).
5. **TTA:** val −0.2, test +0.7 pp, so off by the validation rule (noise level).

## Conclusion

Data augmentation is the most effective single anti-overfitting step for this tiny dataset (+11 pp, gap −34 pp). Two
problems remain and are visible in the curves: the oversized FC head (late re-emerging overfitting) and the constant
learning rate (noisy plateau). Next rung: **S3, global average pooling + dropout instead of the FC head.**
