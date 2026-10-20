# EXP-05: Track 1, S1: + BatchNorm

**Date:** trained by the user 2026-10-08 21:47 to 21:56, evaluated 2026-10-12 · **Run:** [scratch_vgg_bn__scratch_base](runs/scratch_vgg_bn__scratch_base.md)
· **Command:** `python -m birdcls.train --model scratch_vgg_bn --config scratch_base`
· **Comparison figure:** [`outputs/figures/track1_S0_vs_S1.png`](../outputs/figures/track1_S0_vs_S1.png)

## Question and hypothesis

**One change vs S0:** BatchNorm after every convolution. Everything else (optimiser, LR, epochs, no augmentation, no
regularisation, seed) is identical, so any difference is caused by BatchNorm.

Hypothesis:
- **Trainability improves:** BN normalises each channel over the batch, keeping activations in a healthy range, so
  gradients flow from the first epoch (no 4-epoch stall as in S0) and training converges faster.
- **Overfitting is not solved:** BN adds slight regularisation (batch noise), but the 6.5 M-parameter FC head and the
  lack of augmentation remain, so a large train/val gap is expected. Those are addressed in S2 (augmentation) and
  S3 (global average pooling + dropout).

## Model: `scratch_vgg_bn` (8,879,528 parameters)

Same as S0, with each `conv → ReLU` replaced by `conv (no bias) → BatchNorm → ReLU`:

| Stage | Layers | Output (C × H × W) | Params |
|---|---|---|---|
| Input | RGB image, normalised | 3 × 224 × 224 | 0 |
| Block 1 | conv3×3(3→32)-**BN**-ReLU, conv3×3(32→32)-**BN**-ReLU, maxpool 2×2 | 32 × 112 × 112 | 10,208 |
| Block 2 | conv3×3(32→64)-BN-ReLU, conv3×3(64→64)-BN-ReLU, maxpool | 64 × 56 × 56 | 55,552 |
| Block 3 | conv3×3(64→128)-BN-ReLU, conv3×3(128→128)-BN-ReLU, maxpool | 128 × 28 × 28 | 221,696 |
| Block 4 | conv3×3(128→256)-BN-ReLU, conv3×3(256→256)-BN-ReLU, maxpool | 256 × 14 × 14 | 885,760 |
| Block 5 | conv3×3(256→256)-BN-ReLU, conv3×3(256→256)-BN-ReLU, maxpool | 256 × 7 × 7 | 1,180,672 |
| Classifier | flatten (12,544) → FC 512 → ReLU → (dropout 0) → FC 200 | 200 logits | 6,525,640 |

- The conv bias is removed (BN's own shift β does the same job). Each BN layer adds a learnable scale γ and shift β per
  channel, plus running mean/variance used at evaluation time.
- Parameter count differs from S0 by only +1,472 (BN γ/β minus conv biases).

## Results

| | S0 (no BN) | **S1 (+ BatchNorm)** | Change |
|---|---|---|---|
| Best val Top-1 | 5.69% (epoch 43) | **15.11%** (epoch 38) | **+9.4 pp (2.7×)** |
| **Test Top-1** | 4.90% | **14.37%** | **+9.5 pp (2.9×)** |
| **Test avg per class** | 4.78% | **13.81%** | +9.0 pp |
| With TTA: val / test | 5.90% / 4.98% | 14.39% / 16.28% | TTA lowers val (−0.7), so off by the rule |
| Clean train accuracy | 98.58% | 99.97% | |
| Overfitting gap | 92.9 pp | **84.9 pp** | −8 pp, still huge |
| Train acc ≥ 99% first reached | epoch 35 | epoch 22 | memorises faster |
| Min val loss (epoch) | 4.96 (7) | 4.74 (7) | |
| Val loss at epoch 50 | 13.79 | 8.87 | far less over-confident |
| Training time | 8.1 min | 9.1 min | |

Test standard error at about 14% is about ±1.0 pp, so the +9.5 pp gain is far beyond noise.

| Epoch | 1 | 2 | 3 | 5 | 10 | 15 | 20 | 25 | 30 | 40 | 50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S0 train acc | .005 | .005 | .009 | .024 | .235 | .680 | .873 | .939 | .961 | .972 | .999 |
| S1 train acc | .008 | .016 | .030 | .057 | .244 | .670 | .960 | 1.000 | .999 | .999 | 1.000 |
| S0 val acc | .004 | .007 | .015 | .022 | .035 | .039 | .046 | .055 | .036 | .033 | .053 |
| S1 val acc | .010 | .031 | .035 | .048 | .079 | .095 | .127 | .136 | .148 | .144 | .148 |
| S0 val loss | 5.30 | 5.28 | 5.13 | 5.04 | 6.25 | 7.90 | 10.29 | 12.13 | 12.44 | 11.84 | 13.79 |
| S1 val loss | 5.18 | 4.99 | 4.93 | 4.85 | 5.08 | 7.21 | 8.15 | 8.44 | 8.70 | 8.78 | 8.87 |

## Observations

1. **Hypothesis partly confirmed: faster start.** S1 learns from the first epochs (val 3.1% at epoch 2 vs 0.7% for S0;
   train 5.7% vs 2.4% at epoch 5). S0's stall is gone, though the overall shape of the training curve is similar, and S1
   reaches full memorisation (≥99% train) 13 epochs earlier.
2. **Unexpected: BN also improves generalisation about 3× (5.7% → 15.1% val, 4.9% → 14.4% test).** It was expected to
   help optimisation only. Plausible reasons: (a) the noise from per-batch statistics acts as a mild regulariser;
   (b) normalised activations let the convolutional layers learn useful features instead of the FC head doing most of
   the fitting; (c) better-conditioned optimisation tends to find flatter solutions. These are hypotheses; this
   experiment does not separate them.
3. **Much more stable validation:** S0's val loss oscillates wildly (10 to 14.7) while S1's rises smoothly and plateaus near
   8.8. S1's val accuracy also plateaus cleanly at about 14 to 15% from epoch 25.
4. **Overfitting is still severe:** 100% train vs 15% val (gap 85 pp). Val loss is lowest at epoch 7 for both models, then
   rises. BatchNorm fixes trainability, not memorisation. The 6.5 M-parameter FC head and the lack of augmentation
   remain, which is what S2 (augmentation) and S3 (global average pooling + dropout) target.
5. **TTA is inconsistent** (val −0.7, test +1.9 pp), so it stays off by the validation rule. At this accuracy level a few
   images swing the numbers.

## Conclusion

BatchNorm is the single most valuable architectural fix so far (about 3× accuracy) and makes training stable, but the
network still memorises the training set. Next rung: **S2, data augmentation**, the classic defence against
overfitting with tiny data.
