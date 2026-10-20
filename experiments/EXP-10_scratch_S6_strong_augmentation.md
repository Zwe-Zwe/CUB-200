# EXP-10: Track 1, S6: stronger data augmentation

**Date:** 2026-10-15 (trained 17:49 to 18:16) · **Run:** [scratch_vgg_gap__scratch_strongaug](runs/scratch_vgg_gap__scratch_strongaug.md)
· **Command:** `python -m birdcls.train --model scratch_vgg_gap --config scratch_strongaug`
· **Comparison figure:** [`outputs/figures/track1_S5_vs_S6.png`](../outputs/figures/track1_S5_vs_S6.png)

## Question and hypothesis (written before training)

**One change vs S5:** the augmentation policy `basic` → **`strong`** (`configs/scratch_strongaug.yaml`). Same model
(`scratch_vgg_gap`, 2.41 M parameters), same SGD momentum 0.9, peak LR 0.01, 2-epoch warm-up + cosine over up to 150
epochs, early stopping patience 20, weight decay 5e-4, no label smoothing / dropout, same seed.

| Transform | `basic` (S2 to S5) | **`strong` (S6)** |
|---|---|---|
| RandomResizedCrop | 60 to 100% of the image area | **25 to 100%** (more scale / position variety) |
| Horizontal flip | p 0.5 | p 0.5 |
| Rotation ±15° + colour jitter 0.2 | yes | replaced by ↓ |
| **TrivialAugmentWide** | no | **one random op per image at a random strength**: identity, shear x/y, translate x/y, rotate, brightness, colour, contrast, sharpness, posterize, solarize, auto-contrast, equalize |
| Random erasing | p 0.25 | p 0.25 |

TrivialAugment (Müller & Hutter, 2021) performs as well as tuned policies such as AutoAugment / RandAugment but has no
hyperparameters, which matters with only 966 validation images to tune on. The crop minimum is 25% rather than the
ImageNet default of 8%, because in fine-grained bird images an 8% crop often contains no bird at all.

Data-loading check: `strong` runs at about 565 img/s with 2 workers (basic: 500 img/s), so it does not slow training.

Why: S5 still memorises the training set (clean train 99.9% vs val 48.0%, gap 51.9 pp). Weight decay could not prevent
this because BatchNorm makes conv weights scale-invariant (EXP-09), so the regulariser has to act on the **data**. The
last data regulariser (S2, basic augmentation) was the largest anti-overfitting step so far (+11 pp, gap −34 pp).

Hypothesis:
- **The gap shrinks clearly:** clean train accuracy well below 99.9% at the end of training.
- **Val/test accuracy rises** above S5 (48.1% test), by a few pp or more.
- **Slower learning early on** (harder training images); possibly still improving at epoch 150, which would mean more
  epochs could help.
- Risk: too much distortion for 224 px fine-grained birds (e.g. colour ops changing plumage cues), which would show up
  as lower train *and* val accuracy (under-fitting).

## Results

| | S5 (basic aug) | **S6 (strong aug)** | Change |
|---|---|---|---|
| Epochs trained | 150 (no early stop) | 150 (no early stop) | |
| Best val Top-1 | 48.03% (ep 139) | **49.90%** (ep 135) | +1.9 pp |
| Mean val Top-1, last 10 epochs | 47.04% | **48.85%** | +1.8 pp |
| **Test Top-1** | 48.09% | **48.26%** | **+0.2 pp (tie)** |
| **Test avg per class** | 47.42% | 46.90% | −0.5 pp |
| With TTA: val / test | 48.03 / 47.92% | 49.28 / 49.42% | TTA off (ladder rule) |
| **Augmented** train acc, epoch 150 | 96.8% | **71.8%** | memorisation of augmented images strongly reduced |
| Clean train accuracy (eval preprocessing) | 99.92% | 96.71% | |
| **Overfitting gap** (clean train − val) | 51.9 pp | **46.8 pp** | −5.1 pp |
| Min val loss (epoch) | 2.39 (125) | **2.14 (148)** | −0.26, still falling at the end |
| Test mean confidence (correct / wrong) | 0.763 / 0.463 | 0.696 / **0.367** | much less confident when wrong |
| Training time | 27.8 min | 26.2 min | |

### Paired significance test (McNemar, exact)

| Split | Both right | Only S5 right | Only S6 right | Both wrong | p-value |
|---|---|---|---|---|---|
| Validation (966) | 383 | 81 | 99 | 403 | 0.21 |
| Test (1,204) | 462 | 117 | 119 | 506 | **0.95** |

The two models get very different images wrong (236 test images differ), but the totals cancel out on test.

| Epoch | 1 | 25 | 50 | 75 | 100 | 110 | 125 | 135 | 150 |
|---|---|---|---|---|---|---|---|---|---|
| S5 val acc | .012 | .149 | .310 | .376 | .442 | .459 | .464 | .464 | .471 |
| S6 val acc | .012 | .133 | .243 | .344 | .386 | .453 | .468 | **.499** | .489 |
| S5 train acc (aug) | .004 | .222 | .498 | .796 | .930 | .952 | .971 | .963 | .968 |
| S6 train acc (aug) | .004 | .127 | .266 | .415 | .564 | .620 | .683 | .692 | .718 |
| S5 val loss | 5.25 | 3.68 | 2.99 | 2.85 | 2.58 | 2.49 | 2.39 | 2.44 | 2.42 |
| S6 val loss | 5.26 | 4.04 | 3.26 | 2.77 | 2.55 | 2.29 | 2.21 | 2.16 | 2.15 |

## Observations

1. **Hypothesis partly confirmed: memorisation is strongly reduced.** On augmented images S6 reaches only 72% train
   accuracy (S5: 97%), and its val loss is the lowest of the ladder (2.14) and still falling at epoch 148. The clean-image
   gap shrinks by 5 pp (51.9 → 46.8), less than the augmented curves suggest, because the network still fits the plain,
   centre-cropped versions of the training images well (96.7%).
2. **Accuracy: no significant change.** Val is +1.9 pp (best epoch) and +1.8 pp (last-10 mean), but test is a tie (+0.2 pp,
   McNemar p = 0.95) and the per-class average is −0.5 pp. Val and test disagree, so by the rules this is **not** an
   improvement; S5 and S6 are equally accurate.
3. **Slower learning, then catching up:** S6 is 7 pp behind S5 at epoch 50 and 6 pp behind at epoch 100, catches up during the
   low-LR phase (epochs 110 to 125) and is ahead on val only in the last 25 epochs. Combined with observation 1 (train 72%,
   val loss still falling), **the model is now under-fitting the harder training distribution**: 2.4 M parameters and 150
   epochs are not enough to exploit the stronger augmentation.
4. **Better calibration again:** confidence on wrong test predictions falls from 0.46 to 0.37, consistent with the lower val loss.
5. **TTA:** −0.6 val / +1.2 test, inconsistent as before, so off.

## Conclusion

Strong augmentation does what a regulariser should (much less memorisation, lowest val loss, better calibration), but
with this small network and schedule it **does not raise test accuracy** (48.3% vs 48.1%, a tie). The bottleneck has moved
from overfitting to **capacity / training length**. This is the classic regularisation-capacity trade-off: regularisation
only pays off if the model is big enough (or trained long enough) to use the harder data. Ladder:
4.9 → 14.4 → 25.4 → 30.8 → 46.3 → 48.1 → **48.3%** (S6 ≈ S5).

Next options: **more capacity** (a residual network / wider channels, now that memorisation is under control) or **longer
training** (e.g. 300 epochs) with the strong augmentation kept. Strong augmentation should stay in the recipe because it is
what makes a larger model safe to train.
