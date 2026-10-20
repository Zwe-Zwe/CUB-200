# EXP-08: Track 1, S4: cosine LR schedule + longer training with early stopping

**Date:** 2026-10-15 (trained 16:24 to 16:51) · **Run:** [scratch_vgg_gap__scratch_cosine](runs/scratch_vgg_gap__scratch_cosine.md)
· **Command:** `python -m birdcls.train --model scratch_vgg_gap --config scratch_cosine`
· **Comparison figure:** [`outputs/figures/track1_S2_S3_S4.png`](../outputs/figures/track1_S2_S3_S4.png)

## Question and hypothesis (written before training)

**Change vs S3: the training schedule** (model `scratch_vgg_gap` and everything else unchanged). Unlike earlier rungs this
step deliberately bundles three schedule settings into one run (user decision: one 150-epoch run with early stopping
instead of separate 50- and 150-epoch runs), so the effect of the schedule and of the extra epochs is not separated.

| Setting | S3 (`scratch_aug`) | **S4 (`scratch_cosine`)** |
|---|---|---|
| LR schedule | constant 0.01 | **2-epoch linear warm-up to 0.01, then cosine decay towards 0** |
| Epochs | 50 | **up to 150** (cosine stretched over 150) |
| Early stopping | off | **on, patience 20** (stop after 20 epochs without a new best val accuracy) |

Why:
- S3 was **under-trained**: val loss still falling at epoch 45, clean train accuracy only 59%.
- S3 was **noisy**: val accuracy swung 21.6 to 30.2% over its last 10 epochs; a constant LR keeps SGD bouncing around the
  minimum. A decaying LR takes smaller and smaller steps, letting the weights settle (standard in ResNet/ConvNeXt
  training recipes).
- **Warm-up** avoids large updates in the first epochs while the random weights and BatchNorm statistics settle.
- **Patience 20** rather than 6: epoch-to-epoch val noise was about ±4 pp, so a short patience could stop on a random dip.
  Note that if early stopping triggers, the cosine schedule has not fully decayed at the stopping point.

Hypothesis:
- Clearly higher accuracy than S3 (30.8% test); much smoother val curve in the second half as the LR decays.
- The model trains longer before overfitting sets in; train accuracy rises well above S3's 59%, and the gap may grow
  again (no regularisation yet: no dropout, weight decay or label smoothing). That would motivate a regularisation rung.
- Early stopping is likely to trigger before 150 if overfitting returns.

## Results

| | S2 (+ aug) | S3 (GAP head) | **S4 (cosine, 150 epochs)** | S4 vs S3 |
|---|---|---|---|---|
| Epochs trained | 50 | 50 | **150** (early stopping did not trigger) | |
| Best val Top-1 | 25.88% (ep 44) | 30.23% (ep 45) | **46.89%** (ep 143) | **+16.7 pp** |
| **Test Top-1** | 25.42% | 30.81% | **46.26%** | **+15.5 pp (1.5×)** |
| **Test avg per class** | 24.62% | 30.04% | **45.31%** | +15.3 pp |
| With TTA: val / test | 25.67 / 26.16% | 30.85 / 30.48% | 46.58 / 47.34% | −0.3 val / +1.1 test: TTA off (ladder rule) |
| Clean train accuracy | 77.12% | 59.18% | **99.82%** | memorisation is back |
| **Overfitting gap** | 51.2 pp | 29.0 pp | **52.9 pp** | **+23.9 pp** |
| Min val loss (epoch) | 3.75 (34) | 3.12 (45) | **2.68 (148)** | val loss never turned upward |
| Val acc spread, last 10 epochs (sd) | 0.011 | 0.025 | **0.005** | noise largely gone |
| Training time | 9.1 min | 8.7 min | 26.2 min | 3× epochs |

Test standard error at about 46% is about ±1.4 pp, so +15.5 pp is more than 10 standard errors.

| Epoch | 1 | 10 | 20 | 30 | 40 | 50 | 60 | 75 | 90 | 100 | 110 | 125 | 140 | 143 | 150 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Backbone LR | 5.0e-3 | 9.9e-3 | 9.6e-3 | 9.1e-3 | 8.5e-3 | 7.6e-3 | 6.7e-3 | 5.1e-3 | 3.5e-3 | 2.6e-3 | 1.7e-3 | 6.9e-4 | 1.1e-4 | 5.5e-5 | 0 |
| Train acc (augmented) | .004 | .071 | .161 | .266 | .407 | .523 | .650 | .792 | .888 | .923 | .943 | .957 | .964 | .966 | .958 |
| Val acc | .011 | .054 | .148 | .200 | .239 | .294 | .346 | .385 | .413 | .450 | .443 | .451 | .454 | **.469** | .453 |
| Val loss | 5.25 | 4.43 | 3.82 | 3.76 | 3.78 | 3.35 | 3.05 | 2.88 | 2.87 | 2.76 | 2.75 | 2.69 | 2.69 | 2.70 | 2.69 |

Val accuracy reached 30% at epoch 51 and 40% at epoch 69.

## Observations

1. **Hypothesis confirmed, with the largest jump of the ladder: +15.5 pp test (30.8% → 46.3%).** S3 was limited by
   its training schedule, not by its architecture.
2. **The first 50 epochs replicate S3** (the cosine LR is still 0.0076 to 0.01 there, close to S3's constant 0.01), and S4's
   val accuracy at epoch 50 (29.4%) matches S3's best (30.2%). Almost all the gain comes from epochs 50 to 150, while the
   LR decays. Because the schedule and the epoch count changed together, this run cannot say how much each contributed.
3. **The decaying LR removes the noise:** the spread (sd) of val accuracy over 10 epochs drops from 0.026 (epochs 41 to 50)
   to 0.014 (91 to 100) to 0.005 (141 to 150). The best epoch is no longer a lucky spike.
4. **Early stopping never triggered:** val accuracy kept setting new bests until epoch 143 and val loss until epoch 148.
   Patience 20 was safe; the full 150 epochs were useful.
5. **Overfitting is back in the accuracy gap:** clean train 99.8% vs val 46.9% (gap 52.9 pp, up from 29.0). Yet val loss
   kept falling, so the model memorised the training images *and* kept improving on new ones. There is still no explicit
   regularisation (weight decay 0, label smoothing 0, dropout 0), so this is now the obvious lever.
6. **TTA:** −0.3 val / +1.1 test, inconsistent, so it stays off by the ladder rule.

## Conclusion

Training long enough with a decaying LR is the most valuable step of the ladder (+15.5 pp), taking the from-scratch CNN to
**46.3% test**: 4.9% → 14.4% → 25.4% → 30.8% → **46.3%**. With the model now fitting the training set almost perfectly
(gap 53 pp), the next rung should be **regularisation**: weight decay (standard 5e-4 for SGD-trained CNNs) and/or label
smoothing and dropout, then residual connections.
