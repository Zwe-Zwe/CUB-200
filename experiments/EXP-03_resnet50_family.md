# EXP-03: ResNet-50 (ImageNet-1k) vs BiT-ResNet-50 (ImageNet-21k)

**Dates:** prepared 2026-10-06 20:50 to 20:57, trained by the user 2026-10-06/07, evaluated 2026-10-08 · **Code:** `birdcls`
· **Runs:** [resnet50.tv2_in1k__A_standard](runs/resnet50.tv2_in1k__A_standard.md), [resnet50.tv2_in1k__B_finegrained](runs/resnet50.tv2_in1k__B_finegrained.md),
[resnetv2_50x1_bit.goog_in21k__A_standard](runs/resnetv2_50x1_bit.goog_in21k__A_standard.md), [resnetv2_50x1_bit.goog_in21k__B_finegrained](runs/resnetv2_50x1_bit.goog_in21k__B_finegrained.md)

## Question and hypothesis

ConvNeXt was designed by "modernising" ResNet-50 (Liu et al., 2022). Two questions:
1. **Architecture:** at similar size and the same ImageNet-1k data, is ConvNeXt-Tiny better than ResNet-50?
2. **Pretraining data:** at the same size, does ImageNet-21k beat ImageNet-1k for a classic ResNet?
Together with EXP-01/02 this gives a 2 × 2 (architecture × data) comparison.

## Setup

| | ResNet-50 | BiT-M ResNetV2-50x1 |
|---|---|---|
| timm name | `resnet50.tv2_in1k` (torchvision V2 weights) | `resnetv2_50x1_bit.goog_in21k` |
| Params (200 classes) | 23.9 M | 23.9 M |
| Pretraining | ImageNet-1k | ImageNet-21k |
| Normalisation layers | **BatchNorm** | GroupNorm + weight standardisation |
| Input mean/std | ImageNet | **0.5 / 0.5** |
| Memory 224 px, bs 32 | 1.7 GiB | 3.3 GiB |
| Memory 448 px, bs 16 | 3.0 GiB, fits, no accumulation | 6.2 GiB, too tight, so 2 × 8 (exact for GroupNorm) |

- **Same configs A and B, identical hyperparameters** to EXP-01/02 (a fair but ConvNeXt-oriented recipe).
- Code changes made for this experiment: per-model normalisation; BatchNorm guard (no gradient accumulation for BatchNorm models).
- Pre-check: BiT features are about 6× larger (std 2.27 vs 0.19 to 0.39), so its new head starts with a higher loss (6.0 vs ln 200 = 5.3).
  A 2-epoch check showed it learns fine (val 79.0% after 2 epochs; ResNet-50: 60.3%), so no model-specific changes were made.
- Smoke tests passed for both models (Config B, both phases).

## Results

| Run | Best epoch | Train acc | Val Top-1 | Gap | Test Top-1 | Test avg/class | Time |
|---|---|---|---|---|---|---|---|
| ResNet-50-1k A | 20/20 | 99.90% | 81.37% | 18.5 pp | 80.48% | 79.87% | 4.8 min |
| ResNet-50-1k A + TTA | | | **82.30%** | | **81.64%** | 81.22% | |
| ResNet-50-1k B | 20/20 | 99.15% | 81.06% | 18.1 pp | 79.90% | 79.24% | 8.0 min |
| ResNet-50-1k B + TTA | | | 80.85% | | 80.40% | 79.76% | |
| BiT-21k A | 17/20 | 99.87% | 86.23% | 13.6 pp | 85.71% | 85.41% | 7.4 min |
| BiT-21k A + TTA | | | 85.92% | | 85.55% | 85.29% | |
| BiT-21k B | 19/20 | 99.66% | 87.89% | 11.8 pp | 86.71% | 86.49% | 15.6 min |
| BiT-21k B + TTA | | | **88.20%** | | **86.30%** | 86.11% | |

### 2 × 2 summary (test Top-1, no TTA, Config A / Config B)

| | ImageNet-1k | ImageNet-21k/22k | Data effect |
|---|---|---|---|
| **ResNet-50** (24 M) | 80.48 / 79.90 | BiT: 85.71 / 86.71 | **+5.2 / +6.8** |
| **ConvNeXt** | Tiny (28 M): 86.13 / 86.63 | Base (88 M): 93.19 / 93.69 | +7.1 / +7.1 (size also changes) |
| **Architecture effect** | **+5.7 / +6.7** | | |

## Observations

1. **Architecture matters:** ConvNeXt-Tiny beats ResNet-50 by about 6 to 7 pp at similar size and the same data, and ResNet-50
   overfits most (gap about 18 vs 13 pp). Well beyond noise (SE about ±1.1 pp).
2. **Pretraining data matters as much:** BiT-21k beats ResNet-50-1k by about 5 to 7 pp at the same size. Caveat: BiT is ResNetV2
   (pre-activation, GroupNorm + WS), so not a perfectly pure data comparison.
3. **A classic CNN with 21k data matches a modern CNN with 1k data:** BiT-21k B 86.71% vs ConvNeXt-Tiny-1k B 86.63%.
4. **Config B helps every model except ResNet-50-1k** (BiT +1.0, ConvNeXts +0.5; ResNet-50 −0.6 test). ResNet-50's best epoch
   is the last one (20/20) in both configs and val is still rising, so it is **under-trained**. B gives it only 15 epochs
   at 224 px with CutMix (clean train acc 99.15% vs 99.90% in A), and BatchNorm must adapt to the resolution change.
   **Fairness caveat:** the recipe (AdamW, backbone LR 1e-4, 20 epochs) suits ConvNeXt; ResNets are usually fine-tuned
   longer and/or with SGD at a higher LR. This motivates Track 2.
5. **TTA helps only the weakest model:** ResNet-50-1k A +0.93 val / +1.16 test (the first consistent TTA gain); noise elsewhere.
6. Validation slightly optimistic for every run (val above test by 0.4 to 1.2 pp).

## Conclusion

Both architecture and pretraining data are worth about 6 pp each. ResNet-50-1k (80.5% test, under-trained) is the
natural starting point for **Track 2: tuning a pretrained CNN**.
