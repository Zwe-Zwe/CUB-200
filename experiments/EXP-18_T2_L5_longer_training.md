# EXP-18: Track 2, L5: longer training

**Date:** 2026-10-17 (trained 13:58 to 14:14) · **Runs:** [resnet50.tv2_in1k__tl_L5_longer_strongaug](runs/resnet50.tv2_in1k__tl_L5_longer_strongaug.md) (on L4),
[convnext_tiny__tl_L5_longer](runs/convnext_tiny__tl_L5_longer.md) (on L3) · **Figure:** [`outputs/figures/track2_L5.png`](../outputs/figures/track2_L5.png)
· **Commands:** `python -m birdcls.train --model resnet50.tv2_in1k --config tl_L5_longer_strongaug` and
`python -m birdcls.train --model convnext_tiny --config tl_L5_longer`

## Question and hypothesis (written before training)

**One change vs each model's best recipe so far: training length.** Epochs 20 → **40** (the 1-epoch warm-up + cosine
schedule stretches over all 40, so the LR also decays more slowly), early-stopping patience 6 → **10** (verified: the only
settings that differ).

| Model | Starts from | Recipe kept |
|---|---|---|
| ResNet-50 | L4 (val 82.51%, test 80.73%) | strong augmentation + L3 regularisation + L2 LR policy |
| ConvNeXt-Tiny | L3 (val 86.85%, test 86.13%) | basic augmentation + L3 regularisation + L2 LR policy |

Why: ResNet-50 peaked at its **last** epoch in both L3 and L4 (best epoch 20 of 20, val loss still falling), so it is
under-trained, especially on the harder strongly-augmented data (as in Track 1, S4 / S6). ConvNeXt-Tiny is run too, for a
symmetric ladder; its L3 peak was epoch 17, so it probably does not need more epochs.

Hypothesis:
- **ResNet-50: higher val/test than L4**, best epoch well past 20. Strong augmentation may finally pay off on test.
- **ConvNeXt-Tiny: little or no gain**; with the slower LR decay it may peak at a similar accuracy, or slightly overfit more.
- Both stay at about 99.9% clean train accuracy.

## Results

| | ResNet-50 L4 (20 ep) | **ResNet-50 L5 (40 ep)** | ConvNeXt-Tiny L3 (20 ep) | **ConvNeXt-Tiny L5 (40 ep)** |
|---|---|---|---|---|
| Epochs trained | 20 | 37 (early stop) | 20 | 29 (early stop) |
| Best val Top-1 (epoch) | 82.51% (20) | **82.92%** (27) | 86.85% (17) | **87.47%** (19) |
| **Test Top-1** | 80.73% | **81.73%** (+1.0) | 86.13% | **85.96%** (−0.2) |
| **Test avg per class** | 80.17% | 81.29% | 85.71% | 85.68% |
| With TTA: val / test | 83.33 / 82.23% | 82.51 / 82.64% | 86.65 / 86.88% | 87.78 / 86.96% |
| Clean train accuracy | 99.90% | 99.97% | 99.90% | 99.87% |
| Training time | 4.7 min | 8.6 min | 5.0 min | 7.3 min |

**Paired tests (McNemar, exact), previous best (A) vs L5 (B):**

| Model | Val: only A / only B, p | Test: only A / only B, p |
|---|---|---|
| ResNet-50 (L4 vs L5) | 32 / 36, p = 0.72 | 39 / 51, p = 0.25 |
| ConvNeXt-Tiny (L3 vs L5) | 23 / 29, p = 0.49 | 32 / 30, p = 0.90 |

| Epoch | 1 | 10 | 17 | 19 | 20 | 25 | 27 | 29 | 37 |
|---|---|---|---|---|---|---|---|---|---|
| Backbone LR | 1.0e-4 | 8.7e-5 | 6.4e-5 | 5.6e-5 | 5.2e-5 | 3.2e-5 | 2.5e-5 | 1.8e-5 | 1.5e-6 |
| ResNet-50 val acc | .292 | .794 | .817 | .810 | .820 | .814 | **.829** | .825 | .825 |
| ConvNeXt-Tiny val acc | .405 | .850 | .860 | **.875** | .859 | .863 | .866 | .873 | stopped |

## Observations

1. **ResNet-50: under-training resolved, small gain.** The best epoch is now 27 of 37 (no longer the last one), and val/test rise
   by +0.4 / +1.0 pp. Not significant (p = 0.72 / 0.25), but in the same direction on both splits.
2. **ConvNeXt-Tiny: a tie.** +0.6 pp val (p = 0.49), −0.2 pp test (p = 0.90). Its val accuracy fluctuates in an 85.9 to 87.5% band
   from epoch 10 onwards and early stopping ends training at epoch 29, so extra epochs do not move it. ConvNeXt-Tiny at 224 px
   appears to have reached a plateau of about 86 to 87% with this recipe.
3. **Both models keep L5** by the validation rule (val higher for both), with the caveat that the gains are within noise.
4. **TTA:** −0.4 val / +0.9 test (ResNet-50), +0.3 / +1.0 (ConvNeXt). Again positive on test for both; it will be decided once,
   on validation, at the final step.

## Conclusion

Longer training gives at most about +1 pp: **ResNet-50 81.7%, ConvNeXt-Tiny 86.0% test** (val 82.9% / 87.5%). At 224 px both
models have now plateaued under the training-recipe changes, so the next lever is the input itself: **L6, higher
resolution (448 px)** for fine-grained detail.
