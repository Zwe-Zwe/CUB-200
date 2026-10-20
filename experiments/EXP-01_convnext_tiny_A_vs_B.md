# EXP-01: ConvNeXt-Tiny (ImageNet-1k), standard vs fine-grained fine-tuning

**Date:** 2026-10-06 · **Code:** original notebook (`archive/notebook_convnext_tiny_run.ipynb`), same logic as `birdcls`
· **Runs:** [convnext_tiny__A_standard](runs/convnext_tiny__A_standard.md), [convnext_tiny__B_finegrained](runs/convnext_tiny__B_finegrained.md)

## Question and hypothesis

Does a recipe aimed at fine-grained recognition (higher resolution, CutMix) beat a standard fine-tune?
Hypothesis: fine details (beak shape, wing bars, eye rings) are lost at 224 px, so 448 px should help, and CutMix
should reduce overfitting on about 19 training images per class.

## Setup

Transfer learning by full fine-tuning: ImageNet-1k weights (torchvision), the 1000-class layer replaced by
Dropout(0.3) → Linear(768, 200). Both configs share AdamW (weight decay 0.05), label smoothing 0.1, stochastic depth 0.1,
basic augmentation, two LR groups (backbone 1e-4, new head 1e-3), 1 warm-up epoch then cosine, early stopping (patience 6).

| | Config A (`A_standard`) | Config B (`B_finegrained`) |
|---|---|---|
| Resolution | 224 px, 20 epochs | 15 epochs at 224 px, then 5 at 448 px (from the best 224 px weights) |
| LR at 448 px | none | backbone 2e-5, head 2e-4, fresh cosine |
| CutMix | none | p = 0.5 |
| Batch | 32 | 32, then 16 |

## Results

| Run | Best epoch | Train acc (clean) | Val Top-1 | Gap | Test Top-1 | Test avg/class | Time |
|---|---|---|---|---|---|---|---|
| A | 17/20 | 99.90% | 86.85% | 13.0 pp | 86.13% | 85.71% | 6.9 min |
| A + TTA | | | 86.65% | | 86.88% | 86.68% | |
| **B** (selected) | 19/20 | 99.84% | **87.16%** | 12.7 pp | 86.63% | 86.39% | 10.2 min |
| B + TTA | | | 86.44% | | 87.04% | 86.86% | |

## Observations

1. **A and B are statistically tied:** +0.3 pp val, +0.5 pp test (6 of 1,204 images), within the ±1 pp standard error.
2. **The 448 px phase helped within B:** 85.3% val at the end of the 224 px + CutMix phase (A was at 86.2% at the same
   epoch), then 87.2% after 5 epochs at 448 px (+1.9 pp). CutMix slows convergence; resolution makes up for it.
3. **Overfitting:** train about 99.9% vs val about 87%, a 13 pp gap. CutMix barely changed it (12.7 vs 13.0 pp).
4. **Calibration:** A's val loss is lowest at epoch 7 (0.741) and rises to 0.839 while accuracy still creeps up
   (growing over-confidence). B's val loss falls during the 448 px phase (0.758).
5. **TTA inconclusive:** raises test (+0.75 A, +0.4 B) but lowers val (−0.2, −0.7), so by the validation rule it is off.
6. Early stopping never triggered; B's best epoch was 19 of 20.
7. **Error analysis (B):** worst classes are gulls, terns, crows, grackles (Ring-billed Gull 0.20 = 1 of 5 correct).
   Top confusion: Boat-tailed Grackle → Brewer's Blackbird (4×). Errors are between near-identical sister species.

## Conclusion

Transfer learning reaches about 86 to 87% with no tuning. The fine-grained recipe gives a small, consistent but not
significant gain, mostly from the 448 px phase. Next: a stronger backbone (EXP-02).
