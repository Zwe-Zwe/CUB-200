# EXP-20: Track 2, L7: final models (retrained on train + val, test evaluated once)

**Date:** 2026-10-17 (trained 14:53 to 15:32) · **Runs:** [resnet50.tv2_in1k__tl_L6_hires_strongaug__full](runs/resnet50.tv2_in1k__tl_L6_hires_strongaug__full.md),
[convnext_tiny__tl_L6_hires__full](runs/convnext_tiny__tl_L6_hires__full.md) · **Ladder figure:** [`outputs/figures/track2_ladder.png`](../outputs/figures/track2_ladder.png)
· **Commands:** `python -m birdcls.train --model resnet50.tv2_in1k --config tl_L6_hires_strongaug --full-train` and
`python -m birdcls.train --model convnext_tiny --config tl_L6_hires --full-train`

## Decisions fixed before training (all made on validation)

| Decision | ResNet-50 | ConvNeXt-Tiny | Evidence |
|---|---|---|---|
| Recipe | **L6**: L2 LR policy + L3 regularisation + **strong** augmentation + 40 epochs (30 at 224 px + 10 at 448 px) | **L6**: L2 LR policy + L3 regularisation + **basic** augmentation + 40 epochs (30 at 224 px + 10 at 448 px) | best val accuracy of every ladder step (EXP-13 to EXP-19) |
| TTA | **off** | **off** | val accuracy with and without TTA is identical for both L6 models (84.58 / 89.44%), so no evidence it helps; default off |
| Epochs / checkpoint | fixed 40-epoch schedule, **final epoch** | same | L6 best epochs were 40 and 38 of 40 (final-epoch val 84.58% / 89.03%) |

## Question and hypothesis (written before training)

**One change vs L6: the training data**, train (3,863) → **train + val (4,829 images, +25%)**, via `--full-train` (as Track 1
S8, EXP-12): no validation, no early stopping, final epoch kept, test evaluated once.

Hypothesis:
- **Higher test accuracy than L6** for both (82.97% / 88.04%). In Track 1 the same step gave +10 pp, but pretrained models
  depend much less on the amount of bird data, so a smaller gain is expected (perhaps +1 to 2 pp).
- Caveats as in EXP-12: 25% more images also means 25% more updates; the final epoch is used instead of a val-selected one.

## Results

The test set was evaluated **once** for each final model, with the decisions fixed above (no TTA).

| | ResNet-50 L6 (train only) | **ResNet-50 final (L7)** | ConvNeXt-Tiny L6 (train only) | **ConvNeXt-Tiny final (L7)** |
|---|---|---|---|---|
| Training images | 3,863 | 4,829 | 3,863 | 4,829 |
| **Test Top-1** | 82.97% | **86.46%** (+3.5) | 88.04% | **90.20%** (+2.2) |
| **Test avg per class** | 82.40% | **86.12%** | 87.57% | **89.94%** |
| Classes with 0% test accuracy | 0 | 0 | 1 | 0 |
| Test with TTA (reported only) | 83.31% | 86.63% | 88.46% | 90.20% |
| Clean train accuracy | 99.95% | 99.92% (all 4,829) | 99.87% | 99.90% (all 4,829) |
| Training time | 15.7 min | 18.0 min | 16.9 min | 19.7 min |

**Paired tests (McNemar, exact, test set):**

| Comparison | Only A right | Only B right | p-value |
|---|---|---|---|
| ResNet-50: L6 (A) vs final (B) | 28 | **70** | **2.6e-5** |
| ConvNeXt-Tiny: L6 (A) vs final (B) | 22 | **48** | **0.0026** |
| Final ResNet-50 (A) vs final ConvNeXt-Tiny (B) | 30 | **75** | **1.3e-5** |
| Final ConvNeXt-Tiny (A) vs ConvNeXt-Base-22k B (EXP-02, reference) (B) | 27 | **69** | 2.2e-5 |

### The complete ladder (test Top-1, %)

| Step | Change | ResNet-50 | ConvNeXt-Tiny |
|---|---|---|---|
| L0 | linear probe (frozen backbone) | 66.20 | 74.00 |
| L1 | full fine-tuning, single LR 1e-3 | 59.80 | 4.40 (collapse) |
| L2 | backbone LR 1e-4 / head 1e-3 + warm-up + cosine | 77.49 | 86.13 |
| L3 | + label smoothing, dropout, stochastic depth, weight decay 0.05 | 80.48 | 86.13 |
| L4 | + strong augmentation | 80.73 (kept) | 83.72 (not kept) |
| L5 | 40 epochs, patience 10 | 81.73 | 85.96 |
| L6 | progressive 224 → 448 px | 82.97 | 88.04 |
| **L7** | **retrain on train + val (final)** | **86.46** | **90.20** |
| | **total gain over L0** | **+20.3 pp** | **+16.2 pp** |

## Observations

1. **Hypothesis confirmed, with larger gains than expected:** +3.5 pp (ResNet-50, p = 3e-5) and +2.2 pp (ConvNeXt-Tiny,
   p = 0.003) from 25% more training images, even with the conservative final-epoch checkpoint. Pretrained models still
   benefit clearly from more bird images at ~19 to 24 images per class. The caveats from EXP-12 apply (25% more updates).
2. **Final results: ConvNeXt-Tiny 90.20% Top-1 / 89.94% average per class; ResNet-50 86.46% / 86.12%.** The architecture gap
   is 3.7 pp (p = 1e-5), smaller than at L0 (7.8 pp): tuning helped the classic ResNet-50 more (+20.3 vs +16.2 pp).
3. **Per-class accuracy tracks Top-1 closely** (within 0.3 to 0.4 pp) and no class has 0% accuracy: the gains are spread over
   classes, not concentrated in easy ones.
4. **The ConvNeXt-Base-22k reference (93.69%) is still 3.5 pp higher** (p = 2e-5) with almost no tuning. Larger model and 11×
   more pretraining data remain worth more than the recipe; the tuned ImageNet-1k ConvNeXt-Tiny closes about half of the
   gap between its untuned L2/L3 level (86.1%) and that reference.
5. TTA would have added +0.2 pp (ResNet-50) and 0 pp (ConvNeXt-Tiny) on test; it stayed off, as decided on validation.

## Conclusion

The two-model ladder reaches **90.2% (ConvNeXt-Tiny) and 86.5% (ResNet-50) test Top-1 accuracy** from 74.0% / 66.2% linear-probe
baselines, with every step justified, measured on validation and recorded. Report wording for the final step: "Hyperparameters
and the number of epochs were selected on a held-out 20% validation split; the final models were retrained on all 4,829 training
images with the selected recipe and evaluated once on the test set."
