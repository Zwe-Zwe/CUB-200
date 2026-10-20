# EXP-19: Track 2, L6: higher resolution (progressive 224 → 448 px)

**Date:** 2026-10-17 (trained 14:17 to 14:50) · **Runs:** [resnet50.tv2_in1k__tl_L6_hires_strongaug](runs/resnet50.tv2_in1k__tl_L6_hires_strongaug.md),
[convnext_tiny__tl_L6_hires](runs/convnext_tiny__tl_L6_hires.md) · **Figure:** [`outputs/figures/track2_L6.png`](../outputs/figures/track2_L6.png)
· **Commands:** `python -m birdcls.train --model resnet50.tv2_in1k --config tl_L6_hires_strongaug` and
`python -m birdcls.train --model convnext_tiny --config tl_L6_hires`

## Question and hypothesis (written before training)

**Change vs L5: input resolution, via progressive resizing** (verified: only the training phases differ). The 40-epoch
budget of L5 is split:

| Phase | Resolution | Epochs | Batch | LR backbone / head | Schedule |
|---|---|---|---|---|---|
| 1 | 224 px | 30 | 32 | 1e-4 / 1e-3 | 1-epoch warm-up + cosine |
| 2 | **448 px** (eval: resize 512, centre crop 448) | 10 | 16 | **2e-5 / 2e-4** | fresh cosine, no warm-up |

Phase 2 starts from the best phase-1 weights. Everything else as each model's L5 recipe (ResNet-50: strong augmentation;
ConvNeXt-Tiny: basic augmentation; both with the L3 regularisers). Batch 16 at 448 px fits without gradient accumulation for
both models (important for ResNet-50's BatchNorm).

Why: at 224 px both models plateaued under recipe changes (L3 to L5). Distinguishing bird features are small, and 448 px gives
4× more pixels. Earlier evidence: in Config B (EXP-01/02/03) the 448 px phase added about +1 to 2 pp val for the ConvNeXts,
but ResNet-50 lost 0.6 pp test there (under-trained, only 15 epochs at 224 px). Track 1 S7 showed how much early
downsampling (loss of detail) can hurt.

Hypothesis:
- **Both models gain in the 448 px phase**, by about +1 to 2 pp val. ResNet-50 now has a well-trained 224 px starting point (30
  epochs), so it should not repeat its Config B loss.
- The best epoch falls in phase 2 (448 px) for both models.
- Training takes roughly twice as long as L5 (448 px epochs cost about 4×).

## Results

| | ResNet-50 L5 (224) | **ResNet-50 L6 (224→448)** | ConvNeXt-Tiny L5 (224) | **ConvNeXt-Tiny L6 (224→448)** |
|---|---|---|---|---|
| Best val Top-1 (epoch) | 82.92% (27) | **84.58%** (40, at 448 px) | 87.47% (19) | **89.44%** (38, at 448 px) |
| Best val in the 224 px phase of L6 | | 83.4% | | 87.3% |
| **Test Top-1** | 81.73% | **82.97%** (+1.2) | 85.96% | **88.04%** (+2.1) |
| **Test avg per class** | 81.29% | **82.40%** | 85.68% | **87.57%** |
| With TTA: val / test | 82.51 / 82.64% | 84.58 / 83.31% | 87.78 / 86.96% | 89.44 / 88.46% |
| Clean train accuracy | 99.97% | 99.95% | 99.87% | 99.87% |
| Overfitting gap | 17.1 pp | 15.4 pp | 12.4 pp | **10.4 pp** |
| Training time | 8.6 min | 15.7 min | 7.3 min | 16.9 min |

**Paired tests (McNemar, exact), L5 (A) vs L6 (B):**

| Model | Val: only A / only B, p | Test: only A / only B, p |
|---|---|---|
| ResNet-50 | 42 / 58, p = 0.13 | 46 / 61, p = 0.18 |
| ConvNeXt-Tiny | 24 / **43**, **p = 0.027** | 28 / **53**, **p = 0.007** |

Val accuracy in the 10 epochs at 448 px:

| 448 px epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| ResNet-50 | .825 | .832 | .836 | .829 | .832 | .834 | .843 | .839 | .845 | **.846** |
| ConvNeXt-Tiny | .867 | .871 | .878 | .884 | .892 | .889 | .890 | **.894** | .891 | .890 |

## Observations

1. **Hypothesis confirmed: resolution is the clearest gain since the LR policy.** ConvNeXt-Tiny gains +2.0 pp val and
   **+2.1 pp test, significant on both splits** (p = 0.027 / 0.007), reaching **88.0% test**. ResNet-50 gains +1.7 / +1.2 pp
   (same direction on both splits, p = 0.13 / 0.18, not significant alone), reaching **83.0% test**.
2. **The gain comes from the 448 px phase itself:** the best 224 px val accuracy within L6 (83.4% / 87.3%) is about the same as
   L5's, and val rises steadily during the 448 px epochs. Both best epochs are in the 448 px phase (40 and 38).
3. **The gap shrinks too** (17.1 → 15.4 pp, 12.4 → 10.4 pp): with more detail visible, the models generalise better
   rather than just memorising better.
4. **ResNet-50 does not repeat its Config B loss** (EXP-03: −0.6 pp at 448 px). With a well-trained 224 px start (30 epochs
   instead of 15) the resolution increase helps. It is still improving at its final epoch, so a longer 448 px phase might add
   a little more.
5. **TTA: exactly equal on val** for both models (84.58 / 89.44% with and without), +0.3 / +0.4 pp on test.

## Conclusion

Higher resolution lifts both models to their best results: **ResNet-50 83.0%, ConvNeXt-Tiny 88.0% test** (val 84.6% / 89.4%).
These L6 recipes are the final recipes. Last step, **L7**: decide TTA on validation (a tie, so off by the default rule),
retrain each L6 recipe on train + val with the same 40-epoch schedule (final epoch kept), and evaluate the test set once.
