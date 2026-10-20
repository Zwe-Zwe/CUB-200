# EXP-16: Track 2, L3: regularisation package

**Date:** 2026-10-17 (trained 13:34 to 13:44) · **Runs:** [resnet50.tv2_in1k__tl_L3_regularise](runs/resnet50.tv2_in1k__tl_L3_regularise.md),
[convnext_tiny__tl_L3_regularise](runs/convnext_tiny__tl_L3_regularise.md) · **Figure:** [`outputs/figures/track2_L2_vs_L3.png`](../outputs/figures/track2_L2_vs_L3.png)
· **Commands:** `python -m birdcls.train --model <model> --config tl_L3_regularise`

## Plan change (from EXP-15)

The original ladder had L3 = longer training. L2 showed the models are not under-trained (best epochs 16 and 10 of 20,
ConvNeXt early-stopped) but overfit (clean train ≈ 99.9%, gaps 19.8 / 13.2 pp), so regularisation comes first. The four
regularisers are added as **one bundled step** (user decision 2026-10-17), because this exact set is the standard
fine-tuning recipe already used in EXP-01/03.

## Question and hypothesis (written before training)

**Change vs L2** (`configs/tl_L3_regularise.yaml`):

| Setting | L2 | **L3** | What it does |
|---|---|---|---|
| Label smoothing | 0 | **0.1** | targets become 90% true class + 10% spread over the others: no incentive to reach 100% confidence |
| Dropout (before the new head) | 0 | **0.3** | randomly drops 30% of the pooled features each step: the head cannot rely on a few features |
| Stochastic depth (drop path) | 0 | **0.1** | randomly skips residual blocks during training: an implicit ensemble of shallower networks |
| Weight decay (AdamW) | 0.01 | **0.05** | stronger pull of weights towards zero |

Everything else unchanged (backbone LR 1e-4 / head 1e-3, 1-epoch warm-up + cosine, basic augmentation, 224 px, batch 32,
20 epochs, patience 6, same seed).

**Reproducibility check built in:** these settings are identical to `configs/A_standard.yaml` (verified programmatically:
no differences). The A_standard runs were trained earlier (ConvNeXt-Tiny in the original notebook, ResNet-50 with
`birdcls`): test 86.13% (ConvNeXt-Tiny) and 80.48% (ResNet-50). L3 re-runs them with the current code, so similar
numbers would confirm that later code changes did not alter training.

Hypothesis:
- **ResNet-50: about +3 pp over L2** (77.5% → ~80.5%, as A_standard suggests), with a smaller gap.
- **ConvNeXt-Tiny: little change** (86.1% → ~86%): its L2 already matched A_standard.
- Lower clean train accuracy and a smaller gap for both (regularisers make memorisation harder), and less confident
  predictions (label smoothing).

## Results

| | ResNet-50 L2 | **ResNet-50 L3** | ConvNeXt-Tiny L2 | **ConvNeXt-Tiny L3** |
|---|---|---|---|---|
| Best val Top-1 (epoch) | 80.02% (16) | **81.37%** (20 of 20) | 86.65% (10) | **86.85%** (17) |
| **Test Top-1** | 77.49% | **80.48%** (+3.0 pp) | 86.13% | **86.13%** (±0) |
| **Test avg per class** | 76.64% | **79.87%** | 85.87% | 85.71% |
| With TTA: val / test | 79.81 / 78.57% | 82.30 / 81.64% | 86.54 / 85.80% | 86.65 / 86.88% |
| Clean train accuracy | 99.84% | 99.90% | 99.87% | 99.90% |
| Overfitting gap | 19.8 pp | 18.5 pp | 13.2 pp | 13.0 pp |
| Test mean confidence: correct / wrong | 0.913 / 0.599 | **0.672 / 0.322** | 0.882 / 0.521 | **0.674 / 0.272** |
| Training time | 4.5 min | 4.7 min | 3.9 min | 5.0 min |

**Paired tests (McNemar, exact), L2 (A) vs L3 (B):**

| Model | Val: only A / only B, p | Test: only A / only B, p |
|---|---|---|
| ResNet-50 | 36 / 49, p = 0.19 | 41 / **77**, **p = 0.0012** |
| ConvNeXt-Tiny | 27 / 29, p = 0.89 | 37 / 37, **p = 1.0** |

**Reproducibility check passed:** the L3 checkpoints give **identical test predictions on 100% of the 1,204 images** to the
earlier A_standard checkpoints for both models (identical val/test numbers and best epochs), including the ConvNeXt-Tiny run
that was trained in the original notebook before the code was restructured. Seeded training is deterministic and the code
changes since then did not alter training.

| Epoch | 1 | 5 | 10 | 15 | 17 | 20 |
|---|---|---|---|---|---|---|
| ResNet-50 L3 val acc | .274 | .735 | .793 | .807 | .812 | **.814** |
| ResNet-50 L3 train acc | .052 | .816 | .931 | .957 | .960 | .961 |
| ConvNeXt-Tiny L3 val acc | .405 | .848 | .855 | .862 | **.869** | .867 |
| ConvNeXt-Tiny L3 train acc | .109 | .941 | .985 | .985 | .986 | .986 |

## Observations

1. **Hypotheses confirmed for both models.** ResNet-50 gains **+3.0 pp test** (77.5% → 80.5%, p = 0.001); on validation the gain
   is +1.4 pp (p = 0.19), smaller but in the same direction. ConvNeXt-Tiny is **unchanged** (86.13% test both times, p = 1).
2. **Regularisation helps the model that overfits more.** ResNet-50 had the larger L2 gap (19.8 vs 13.2 pp) and gains; ConvNeXt
   does not. Its architecture and pretraining (trained with strong regularisation on ImageNet) may already provide what these
   regularisers add.
3. **The gap barely shrinks** (−1.3 / −0.2 pp; clean train accuracy still about 99.9%). The regularisers mainly change
   *confidence*, not memorisation: mean confidence on wrong test predictions halves (0.60 → 0.32 ResNet, 0.52 → 0.27 ConvNeXt),
   the expected effect of label smoothing. For ResNet-50 that comes with better top-1 accuracy.
4. **ResNet-50 is still improving at its last epoch** (best epoch 20 of 20), so longer training is now relevant for it.
   ConvNeXt peaks at epoch 17.
5. **TTA keeps helping ResNet-50** (+0.9 val / +1.2 test; it was positive on test in every ladder step so far), but not ConvNeXt
   (−0.2 val). It stays off during the ladder and is decided at the end on validation.

Kept for the next steps: L3 settings for both models (higher val for both; standard practice; better calibration).

## Conclusion

The regularisation package lifts ResNet-50 to **80.5%** test and leaves ConvNeXt-Tiny at **86.1%**. Memorisation (train
≈ 99.9%) is still there, so the next step targets the data itself: **L4, strong augmentation** (TrivialAugment + wider crops).
