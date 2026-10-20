# EXP-17: Track 2, L4: strong data augmentation

**Date:** 2026-10-17 (trained 13:46 to 13:56) · **Runs:** [resnet50.tv2_in1k__tl_L4_strongaug](runs/resnet50.tv2_in1k__tl_L4_strongaug.md),
[convnext_tiny__tl_L4_strongaug](runs/convnext_tiny__tl_L4_strongaug.md) · **Figure:** [`outputs/figures/track2_L3_vs_L4.png`](../outputs/figures/track2_L3_vs_L4.png)
· **Commands:** `python -m birdcls.train --model <model> --config tl_L4_strongaug`

## Question and hypothesis (written before training)

**One change vs L3:** augmentation `basic` → **`strong`** (`configs/tl_L4_strongaug.yaml`; verified: the only setting that
differs from L3). Same preset as Track 1 S6 (EXP-10):

| Transform | `basic` (L0 to L3) | **`strong` (L4)** |
|---|---|---|
| RandomResizedCrop | 60 to 100% of the image | **25 to 100%** |
| Horizontal flip | p 0.5 | p 0.5 |
| Rotation ±15° + colour jitter 0.2 | yes | replaced by ↓ |
| **TrivialAugmentWide** | no | **one random op per image at a random strength** (shear, translate, rotate, colour, contrast, sharpness, posterize, solarize, equalize, ...) |
| Random erasing | p 0.25 | p 0.25 |

Everything else as L3: AdamW, backbone LR 1e-4 / head 1e-3, 1-epoch warm-up + cosine, label smoothing 0.1, dropout 0.3,
stochastic depth 0.1, weight decay 0.05, 224 px, batch 32, 20 epochs, patience 6, same seed.

Why: after L3 both models still fit the training set almost perfectly (clean train ≈ 99.9%; gaps 18.5 / 13.0 pp). The L3
regularisers mainly reduced confidence, not memorisation. Strong augmentation acts on the data, so the network never sees
the same image twice.

What Track 1 suggests: in S6, strong augmentation cut memorisation sharply but did **not** raise test accuracy within the
same number of epochs, because the harder data needs more training. Here the models are pretrained, so they may adapt faster.

Hypothesis:
- **Lower training accuracy and a smaller gap** for both models.
- **Accuracy:** a gain for ResNet-50 (the model that overfits more and benefited from L3); little or no change for
  ConvNeXt-Tiny. With only 20 epochs, the harder data could also leave a model **under-trained** (best epoch = 20), which would
  then call for longer training (L5b).

## Results

| | ResNet-50 L3 | **ResNet-50 L4** | ConvNeXt-Tiny L3 | **ConvNeXt-Tiny L4** |
|---|---|---|---|---|
| Best val Top-1 (epoch) | 81.37% (20 of 20) | **82.51%** (20 of 20) | 86.85% (17) | 86.54% (10; early stop at 16) |
| **Test Top-1** | 80.48% | **80.73%** (+0.25) | 86.13% | **83.72%** (−2.4) |
| **Test avg per class** | 79.87% | 80.17% | 85.71% | 83.43% |
| With TTA: val / test | 82.30 / 81.64% | 83.33 / 82.23% | 86.65 / 86.88% | 86.96 / 84.30% |
| Augmented train acc, last epoch | 96.1% | **89.7%** | 98.6% | 96.0% |
| Clean train accuracy | 99.90% | 99.90% | 99.90% | 99.87% |
| Min val loss (epoch) | 0.919 (15) | 0.837 (19) | 0.773 (5) | 0.711 (10) |
| Test confidence: correct / wrong | 0.672 / 0.322 | 0.704 / 0.346 | 0.674 / 0.272 | 0.747 / 0.337 |
| Training time | 4.7 min | 4.7 min | 5.0 min | 4.0 min |

**Paired tests (McNemar, exact), L3 (A) vs L4 (B):**

| Model | Val: only A / only B, p | Test: only A / only B, p |
|---|---|---|
| ResNet-50 | 32 / 43, p = 0.25 | 46 / 49, p = 0.84 |
| ConvNeXt-Tiny | 38 / 35, p = 0.82 | **66** / 37, **p = 0.0055** |

| Epoch | 1 | 5 | 10 | 15 | 16 | 20 |
|---|---|---|---|---|---|---|
| ResNet-50 L4 val acc | .292 | .731 | .796 | .823 | .818 | **.825** |
| ResNet-50 L4 train acc (aug) | .036 | .693 | .845 | .880 | .892 | .897 |
| ConvNeXt-Tiny L4 val acc | .388 | .833 | **.865** | .859 | .862 | stopped |
| ConvNeXt-Tiny L4 train acc (aug) | .097 | .853 | .933 | .955 | .960 | stopped |

## Observations

1. **Hypothesis on memorisation partly confirmed:** strong augmentation lowers training accuracy on *augmented* images
   (ResNet 96 → 90%), but the clean-image training accuracy stays at 99.9%. As in Track 1 S6, the network still fits the plain
   training images; the overfitting gap on clean images does not shrink.
2. **ResNet-50: a small, non-significant gain** (+1.1 pp val, p = 0.25; +0.25 pp test, p = 0.84). It is **still improving at
   its last epoch** in both L3 and L4 (best epoch 20 of 20, val loss lowest at epoch 19), so it is under-trained for the harder
   data. The same pattern as S6 (strong augmentation needs more epochs to pay off).
3. **ConvNeXt-Tiny: no gain on val (−0.3 pp, p = 0.82) and significantly worse on test (−2.4 pp, p = 0.006).** Its best epoch
   is 10, while the LR is still high (5.4e-5); val accuracy then fluctuates and early stopping ends training at epoch 16
   before the cosine schedule decays. Its val and test disagree here, and val does not favour L4, so by the rules L4 is
   **not kept** for ConvNeXt-Tiny.
4. **Augmentation interacts with the model:** helpful-but-slow for the classic ResNet-50 (which overfits more), unhelpful for
   ConvNeXt-Tiny, consistent with L3, where only ResNet-50 benefited from regularisation.

**Decision (by validation accuracy):**
- **ResNet-50 keeps L4** (strong augmentation): val 82.51% > 81.37%.
- **ConvNeXt-Tiny reverts to L3** (basic augmentation): val 86.85% > 86.54%, and test is significantly worse with L4.

## Conclusion

Strong augmentation gives a small validation gain for ResNet-50 (82.5% val / 80.7% test) and hurts ConvNeXt-Tiny (83.7% test),
so the two models' recipes now diverge. ResNet-50 has peaked at the last epoch twice in a row, so the next step is
**L5: longer training** (40 epochs, patience 10), applied to each model's best recipe (ResNet-50: L4; ConvNeXt-Tiny: L3).
