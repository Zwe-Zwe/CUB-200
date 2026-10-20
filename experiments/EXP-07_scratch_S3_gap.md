# EXP-07: Track 1, S3: global average pooling instead of the FC head

**Date:** 2026-10-15 (trained 16:11 to 16:20) · **Run:** [scratch_vgg_gap__scratch_aug](runs/scratch_vgg_gap__scratch_aug.md)
· **Command:** `python -m birdcls.train --model scratch_vgg_gap --config scratch_aug`
· **Comparison figure:** [`outputs/figures/track1_S1_S2_S3.png`](../outputs/figures/track1_S1_S2_S3.png)

## Question and hypothesis (written before training)

**One change vs S2:** the classifier head. Same config (`scratch_aug`: augmentation, SGD momentum 0.9, constant LR 0.01,
50 epochs, no regularisation, dropout 0), same convolutional layers, same seed. Only the model changes,
`scratch_vgg_bn` → `scratch_vgg_gap`.

| | S2 head (FC) | **S3 head (GAP)** |
|---|---|---|
| After block 5 | 256 × 7 × 7 feature map | 256 × 7 × 7 feature map |
| Pooling | none (flatten all 12,544 values) | **global average pooling**: each channel → its mean (256 values) |
| Layers | FC 12,544→512, ReLU, FC 512→200 | FC 256→200 |
| Head parameters | 6,525,640 | **51,400** (127× fewer) |
| Total parameters | 8,879,528 | **2,405,288** (−73%) |

Convolutional feature extractor: unchanged, 2,353,888 parameters (architecture in EXP-05).

Why GAP should help (Lin et al., 2014, "Network in Network"; used by ResNet, ConvNeXt and most modern CNNs):
- **Far less capacity to memorise:** S2's val loss started rising again at epoch 34 while train accuracy kept climbing,
  and the 6.5 M-weight FC head is the obvious place where memorisation happens.
- **Translation robustness:** averaging over all 7 × 7 positions makes the prediction depend on *whether* a pattern is
  present, not *where*, which suits birds appearing anywhere in the frame.
- **Forces better features:** each of the 256 channels must act as a detector for a class-relevant pattern, since the
  head can no longer combine individual positions.

Hypothesis:
- Smaller train/val gap than S2 (51.2 pp) and val loss rising later or not at all within 50 epochs.
- Val/test accuracy higher than S2 (25.4% test), but possibly a smaller gain than S1/S2, and possibly slower early
  learning (the head has less capacity, so all the work moves into the convolutional layers).
- Risk: a 256-dimensional feature vector may be a capacity bottleneck for 200 fine-grained classes.

## Results

| | S1 (+ BN) | S2 (+ augmentation) | **S3 (GAP head)** | S3 vs S2 |
|---|---|---|---|---|
| Parameters | 8.88 M | 8.88 M | **2.41 M** | −73% |
| Best val Top-1 | 15.11% (ep 38) | 25.88% (ep 44) | **30.23%** (ep 45) | **+4.4 pp** |
| **Test Top-1** | 14.37% | 25.42% | **30.81%** | **+5.4 pp** |
| **Test avg per class** | 13.81% | 24.62% | **30.04%** | +5.4 pp |
| With TTA: val / test | 14.39 / 16.28% | 25.67 / 26.16% | 30.85 / 30.48% | +0.6 val / −0.3 test: noise |
| Clean train accuracy | 99.97% | 77.12% | **59.18%** | |
| Augmented train acc, epoch 50 | n/a | 67.0% | 51.8% | |
| **Overfitting gap** | 84.9 pp | 51.2 pp | **29.0 pp** | **−22.2 pp** |
| Min val loss (epoch) | 4.74 (7) | 3.75 (34) | **3.12 (45)** | **still falling near the end** |
| Val loss at epoch 50 | 8.87 | 4.56 | 3.38 | |
| Training time | 9.1 min | 9.1 min | 8.7 min | |

Test standard error at about 30% is about ±1.3 pp, so +5.4 pp is about 4 standard errors: a real gain.

| Epoch | 1 | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 | 50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S3 train acc (augmented) | .008 | .038 | .080 | .125 | .173 | .236 | .268 | .340 | .412 | .464 | .518 |
| S3 val acc | .021 | .051 | .063 | .089 | .145 | .157 | .183 | .216 | .267 | **.302** | .284 |
| S3 val loss | 5.13 | 4.58 | 4.39 | 4.14 | 3.83 | 3.89 | 3.72 | 3.60 | 3.37 | **3.12** | 3.38 |
| S2 val acc (reference) | .008 | .012 | .062 | .089 | .137 | .137 | .197 | .196 | .210 | .228 | .243 |
| S2 val loss (reference) | 5.22 | 4.96 | 4.57 | 4.21 | 3.99 | 4.09 | 3.78 | 4.17 | 3.92 | 4.14 | 4.56 |

## Observations

1. **Hypothesis confirmed: fewer parameters, better accuracy.** Removing 6.47 M head parameters (−73% of the model)
   raised test accuracy by +5.4 pp. The extra capacity of the FC head was being used to memorise, not to generalise.
2. **Overfitting is pushed out of the 50-epoch window:** gap 51 → 29 pp, and val loss keeps falling until epoch 45
   (S2 turned upward at epoch 34). Train and val curves stay much closer together.
3. **Hypothesis wrong on early learning:** I expected GAP to learn more slowly at first; it learned *faster*
   (val 5.1% at epoch 5 vs 1.2% for S2). With a tiny head, gradients go straight into the convolutional features.
4. **Under-trained, not capacity-limited:** clean train accuracy is only 59% and val loss is still decreasing, so the
   network has not finished learning. More epochs with a decaying LR (S5) should pay off.
5. **Very noisy validation:** val accuracy swings between 21.6% and 30.2% over the last 10 epochs (constant LR 0.01).
   The best epoch (45) is partly a lucky peak, although test (30.81%) agrees with val (30.23%) here. A cosine LR schedule
   should remove most of this noise (S5).
6. **TTA:** +0.6 val, −0.3 test, i.e. noise. Ladder steps are compared without TTA (see conventions in `README.md`).

## Conclusion

GAP is a clear win: a smaller, faster model with +5.4 pp test accuracy and the gap down to 29 pp. The ladder so far:
4.9% → 14.4% → 25.4% → **30.8%** test. The main limitation is now training dynamics (noisy constant LR, too few epochs).
Remaining planned rungs: dropout / label smoothing / weight decay (regularisation), residual connections, and
LR schedule + longer training (S5, which the curves suggest is now the most promising next step).
