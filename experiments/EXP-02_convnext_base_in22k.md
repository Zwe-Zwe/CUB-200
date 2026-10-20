# EXP-02: ConvNeXt-Base pretrained on ImageNet-22k

**Date:** 2026-10-06 · **Code:** `birdcls` · **Runs:** [convnext_base.fb_in22k__A_standard](runs/convnext_base.fb_in22k__A_standard.md),
[convnext_base.fb_in22k__B_finegrained](runs/convnext_base.fb_in22k__B_finegrained.md)

## Question and hypothesis

How much does a bigger backbone with much bigger pretraining data help? ImageNet-22k (~14 M images, 21,841 classes)
contains many fine-grained categories, including hundreds of bird species, so its features should transfer better
than ImageNet-1k (1.28 M images, 1,000 classes).

## Setup

- Model: `timm` `convnext_base.fb_in22k`, 87.8 M parameters (vs 28.0 M for Tiny); head = Dropout(0.3) → Linear(1024, 200).
- **Same configs A and B as EXP-01**, unchanged hyperparameters.
- Memory: at 448 px batch 16 is OOM on 8 GB (batch 8 = 5.0 GiB peak), so 2 × 8 gradient accumulation. This is exactly
  equivalent because ConvNeXt uses LayerNorm.
- A ConvNeXt-Base ImageNet-1k run was started and then cancelled before training in favour of 22k, so model size
  and pretraining data change together (confound, see below).
- Commands: `python -m birdcls.train --model convnext_base.fb_in22k --config A_standard B_finegrained`

## Results

| Run | Best epoch | Train acc | Val Top-1 | Gap | Test Top-1 | Test avg/class | Time |
|---|---|---|---|---|---|---|---|
| A | 11 (early stop at 17) | 99.92% | 93.17% | 6.75 pp | 93.19% | 92.96% | 10.7 min |
| A + TTA | | | 93.37% | | 93.02% | 92.88% | |
| **B (best overall)** | 18/20 | 99.90% | **94.62%** | **5.28 pp** | **93.69%** | **93.55%** | 19.6 min |
| B + TTA | | | 94.41% | | 93.60% | 93.47% | |

Validation accuracy per epoch:

| Epoch | 1 | 2 | 3 | 5 | 7 | 9 | 11 | 13 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A (224) | .719 | .889 | .896 | .909 | .925 | .927 | **.933** | .930 | .929 | .931 | .929 | stop | | |
| B | .672 | .859 | .907 | .917 | .933 | .932 | .935 | .936 | .936 | .935† | .934† | **.946†** | .941† | .941† |

† = 448 px phase. B's val loss is lowest in the 448 px phase (0.360 vs 0.374 at 224 px).

## Observations

1. **Base-22k vs Tiny-1k: +7.1 pp test Top-1 (86.63% → 93.69%); test errors 161 → 76 images (−53%).**
2. **The overfitting gap halves** (12.7 → 5.3 pp) at the same train accuracy (~99.9%): better pretrained features
   generalise better. This reduced overfitting more than any regulariser.
3. **Confound:** model size (28 M → 88 M) and pretraining data (1k → 22k) changed together. EXP-03 isolates the data
   effect at fixed size for ResNet-50.
4. **B vs A: +1.45 pp val, +0.50 pp test.** The test difference alone is within noise (SE about ±0.7 pp), but B ≥ A on val
   and test for both ConvNeXts, and the 448 px phase adds +0.9 pp val within B.
5. **TTA: no reliable effect** (±0.2 pp), so it stays off.
6. **Validation optimism, a live example:** B's val (94.62%) is about 0.9 pp above its test (93.69%) because the best of
   20 epochs was chosen on val. A's val and test agree (93.17 vs 93.19). This is why the test set is never used to choose.
7. **Error analysis (B):** no confused pair occurs more than 2×. Remaining errors: American Crow → Common Raven,
   Western → California Gull, Whip-poor-will → Chuck-will's-widow, Common → Arctic Tern, Warbling → Philadelphia Vireo,
   Louisiana → Northern Waterthrush. Tiny's worst confusion (grackle → blackbird, 4×) is mostly fixed.

## Conclusion

Pretraining source and model capacity dominate: +7 pp and half the overfitting, without tuning. ConvNeXt-Base-22k B is
the best model so far and the reference ceiling for the CNN-tuning story (Track 2).
