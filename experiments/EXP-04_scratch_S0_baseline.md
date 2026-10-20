# EXP-04: Track 1, S0: naive CNN trained from scratch

**Date:** 2026-10-08 · **Code:** `birdcls` (`birdcls/scratch.py`, `configs/scratch_base.yaml`)
· **Run:** [scratch_vgg__scratch_base](runs/scratch_vgg__scratch_base.md)

## Question and hypothesis

How well does a plain CNN trained from random initialisation do with about 19 training images per class and no
tricks? This is the first rung of the Track 1 "ablation ladder", where each later step changes one thing.
Hypothesis: severe overfitting. The large fully connected head can memorise the training set.

## Model: `scratch_vgg` (VGG-style, 8,878,056 parameters)

| Stage | Layers | Output (C × H × W) | Params |
|---|---|---|---|
| Input | RGB image, normalised | 3 × 224 × 224 | 0 |
| Block 1 | conv3×3(3→32)-ReLU, conv3×3(32→32)-ReLU, maxpool 2×2 | 32 × 112 × 112 | 10,144 |
| Block 2 | conv3×3(32→64)-ReLU, conv3×3(64→64)-ReLU, maxpool | 64 × 56 × 56 | 55,424 |
| Block 3 | conv3×3(64→128)-ReLU, conv3×3(128→128)-ReLU, maxpool | 128 × 28 × 28 | 221,440 |
| Block 4 | conv3×3(128→256)-ReLU, conv3×3(256→256)-ReLU, maxpool | 256 × 14 × 14 | 885,248 |
| Block 5 | conv3×3(256→256)-ReLU, conv3×3(256→256)-ReLU, maxpool | 256 × 7 × 7 | 1,180,160 |
| Classifier | flatten (12,544) → FC 512 → ReLU → (dropout 0) → FC 200 | 200 logits | **6,525,640** |

- 10 convolutional layers (3×3, padding 1) + 2 FC layers. **74% of the parameters are in the FC head.**
- He (Kaiming) initialisation (needed for a deep ReLU network without normalisation); FC weights ~ N(0, 0.01).
- An adaptive average pool fixes the feature map at 7×7, so the FC input does not depend on image size.

## Training setup (`scratch_base`)

| Setting | Value | Why |
|---|---|---|
| Optimiser | SGD, momentum 0.9 | classic baseline |
| Learning rate | 0.01, **constant** (no schedule, no warm-up) | baseline |
| Epochs | 50, **no early stopping** | show the full overfitting curve |
| Batch | 32 at 224 px | |
| Augmentation | **none** (resize 256 + centre crop 224 for train too) | baseline |
| Regularisation | **none** (weight decay 0, label smoothing 0, dropout 0) | baseline |
| Command | `python -m birdcls.train --model scratch_vgg --config scratch_base` | |

Pre-check (2-epoch probe): loss still at chance after 2 epochs (5.29 vs ln 200 = 5.30). About 10 s per epoch.

## Results

| | Value |
|---|---|
| Best val Top-1 | **5.69%** (epoch 43) |
| Test Top-1 / avg per class | **4.90% / 4.78%** (chance = 0.5%) |
| With TTA | val 5.90%, test 4.98% / 4.86% (noise) |
| Clean train accuracy | 98.58% |
| **Overfitting gap** | **92.9 pp** |
| Training time | 8.1 min |

| Epoch | 1 | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 | 50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Train acc | .005 | .024 | .235 | .680 | .873 | .939 | .961 | .993 | .972 | .996 | .999 |
| Val acc | .004 | .022 | .035 | .039 | .046 | .055 | .036 | .043 | .033 | .052 | .053 |
| Val loss | 5.30 | 5.04 | 6.25 | 7.90 | 10.29 | 12.13 | 12.44 | 14.29 | 11.84 | 11.69 | 13.79 |

Curves: [`outputs/runs/scratch_vgg__scratch_base/curves.png`](../outputs/runs/scratch_vgg__scratch_base/curves.png)

## Observations

1. **Slow start:** at chance for about 4 epochs; a deep plain network without BatchNorm needs time before the gradients
   become useful (the target of S1).
2. **Memorisation:** train accuracy 0.5% → 23% (epoch 10) → 87% (20) → 99.9% (50).
3. **No generalisation:** val reaches about 4% by epoch 11, then stays in a noisy 3 to 6% band for 40 epochs.
4. **Val loss is lowest at epoch 7 (4.96), then explodes to about 14:** the network becomes confidently wrong on unseen
   birds, the clearest overfitting signal.
5. Test 4.90% is about 10× chance, so a little real signal is learned, but almost all capacity goes into memorising.

## Conclusion

A textbook overfitting baseline. Next rungs: S1 BatchNorm (trainability), then augmentation and global average
pooling (overfitting), which should give the largest gains.
