# EXP-12: Track 1, S8: S6 retrained on all training data (no validation split)

**Date:** 2026-10-15 (re-run 23:08 to 23:36) · **Run:** [scratch_vgg_gap__scratch_strongaug__full](runs/scratch_vgg_gap__scratch_strongaug__full.md)
· **Command:** `python -m birdcls.train --model scratch_vgg_gap --config scratch_strongaug --full-train`
· **Comparison figure:** [`outputs/figures/track1_S6_vs_S8.png`](../outputs/figures/track1_S6_vs_S8.png) (S8 has no val curve by design)

## Question and hypothesis (written before training)

**One change vs S6: the training data.** S6's recipe and model are reused unchanged (`scratch_vgg_gap`,
`configs/scratch_strongaug.yaml`: strong augmentation, SGD momentum 0.9, peak LR 0.01, 2-epoch warm-up + cosine over 150
epochs, weight decay 5e-4, same seed). The only difference is that the 966 validation images are added to the training set.

| | S6 | **S8 (full-train)** |
|---|---|---|
| Training images | 3,863 (80% of train.txt) | **4,829 (all of train.txt)** |
| Images per class | about 19 | **about 24** (+25%) |
| Validation | 966 held-out images | **none** (they are training images now) |
| Early stopping | on (patience 20; never triggered) | **off** |
| Checkpoint | best epoch by val accuracy (135) | **final epoch (150)** |

Why this is legitimate (and how it avoids test-set leakage): every decision was already made on the validation split
in EXP-04 to EXP-11: the architecture (S3/S7), the recipe (S6), the number of epochs (150; S6 never early-stopped and its
cosine schedule ends at LR 0), and no TTA. S8 only re-trains that fixed recipe on more data, and the **test set is
evaluated once** at the end. No validation score is reported for S8 (it would be inflated, since those images were
trained on); `birdcls.evaluate` reports it as NaN and excludes the run from "best by validation" selection.

Code added for this step: `--full-train` option in `birdcls.train` (and full-train handling in `engine.fit`,
`evaluate`, `report`); smoke test passed (4,829 training images, val NaN, final epoch kept).

Fairness note: S6's test score (48.26%) comes from its best-epoch checkpoint selected on validation (epoch 135). S8 has no
validation and must use the final epoch, which is slightly conservative. S6's final-epoch val accuracy (48.86%) was close to
its best (49.90%), so the difference should be small. A measured S8 gain is, if anything, an underestimate.

Hypothesis:
- **Higher test accuracy than S6 (48.26%)**: 25% more images per class is a lot for a from-scratch model; size uncertain.
- Training curves similar in shape to S6 (same schedule), with slightly lower training accuracy at a given epoch
  (more images to fit).
- Time about 25% longer per epoch (4,829 vs 3,863 images), partly offset by skipping validation: about 30 min.

## Results

**Run history:** the first attempt (19:27) stopped after epoch 35 of 150 when the agent session ended; it was re-run from
scratch with the same command and seed (log of the first attempt kept as `train_attempt1_interrupted.log` in the run folder).
The re-run reproduced the first attempt exactly (e.g. epoch 25: train loss 3.891, acc 0.137 in both). The re-run was started
detached from the agent session (`setsid nohup`) so that a session end cannot stop it.

The test set was evaluated **once** for S8 (no TTA, as fixed in advance).

| | S6 | **S8 (full-train)** | Change |
|---|---|---|---|
| Training images | 3,863 | 4,829 | +25% |
| Optimiser updates (150 epochs × updates/epoch) | 18,000 (120/epoch) | 22,500 (150/epoch) | +25% (see caveat) |
| Checkpoint | best val epoch (135) | final epoch (150) | |
| **Test Top-1** | 48.26% | **58.64%** | **+10.4 pp** |
| **Test avg per class** | 46.90% | **57.60%** | **+10.7 pp** |
| Classes with 0% test accuracy | 14 | **2** | |
| Test with TTA (reported only) | 49.42% | 59.14% | TTA stays off (decided before) |
| Augmented train acc, epoch 150 | 71.8% | 74.9% | |
| Clean train accuracy | 96.71% | 98.63% (on all 4,829 images) | |
| Test mean confidence (correct / wrong) | 0.696 / 0.367 | 0.752 / 0.417 | |
| Training time | 26.2 min | 27.4 min | |

### Paired significance test (McNemar, exact, test set)

| Both right | Only S6 right | Only S8 right | Both wrong | p-value |
|---|---|---|---|---|
| 519 | 62 | **187** | 436 | **9.4e-16** |

S8 fixes 187 test images that S6 gets wrong and breaks only 62, a highly significant improvement.

| Epoch | 1 | 25 | 50 | 75 | 100 | 125 | 135 | 150 |
|---|---|---|---|---|---|---|---|---|
| S6 train acc (aug) | .004 | .127 | .266 | .415 | .564 | .683 | .692 | .718 |
| S8 train acc (aug) | .007 | .137 | .310 | .468 | .609 | .732 | .731 | .749 |
| S6 train loss | 5.32 | 4.02 | 3.17 | 2.47 | 1.80 | 1.41 | 1.36 | 1.27 |
| S8 train loss | 5.31 | 3.89 | 2.93 | 2.20 | 1.64 | 1.22 | 1.16 | 1.12 |

## Observations

1. **Hypothesis confirmed, with a large effect: +10.4 pp test Top-1 (48.3% → 58.6%), p = 9e-16**, despite the
   conservative final-epoch checkpoint. The per-class average rises even more (+10.7 pp), and classes the model gets
   completely wrong drop from 14 to 2.
2. **Data is the strongest lever for a from-scratch model.** 25% more images (about 5 more per class) gave more than any
   single regulariser (weight decay +1.8, strong augmentation +0.2). At ~19 to 24 images per class the model is clearly on
   the steep part of its learning curve.
3. **Caveat (confound): more data also meant more updates.** The epoch count was fixed, so S8 made 25% more SGD updates
   (22,500 vs 18,000), and S6 was still improving at epoch 150 (val loss lowest at epoch 148). Part of the gain may come
   from the longer optimisation rather than the extra images. A control run, S6 for 188 epochs (≈ 22,500 updates), would
   separate the two; S4 suggests longer training helps but would not explain a gain this size on its own.
4. **Fits training data better too** (aug train 74.9% vs 71.8%, lower train loss at every epoch), the opposite of
   memorising more: with more examples per class the features generalise better.
5. Because the validation images were used for training, no validation score exists for S8 (shown as NaN by design).

## Conclusion

Retraining the selected recipe on all of `train.txt` is the best from-scratch result: **58.64% test Top-1, 57.60% average
per-class accuracy** (2.4 M-parameter VGG-style CNN with GAP head, strong augmentation, cosine LR, weight decay, trained from
random initialisation). Final Track 1 ladder (test Top-1):
S0 4.9 → S1 14.4 → S2 25.4 → S3 30.8 → S4 46.3 → S5 48.1 → S6 48.3 → (S7 ResNet-18: 41.4) → **S8 58.6%**.

Report wording: "Hyperparameters were selected on a held-out 20% validation split; the final model was retrained on all
4,829 training images with the selected recipe and fixed number of epochs, and evaluated once on the test set."

