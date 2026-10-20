# EXP-15: Track 2, L2: discriminative learning rates + warm-up + cosine

**Date:** 2026-10-17 (trained 13:21 to 13:30) · **Runs:** [resnet50.tv2_in1k__tl_L2_discriminative](runs/resnet50.tv2_in1k__tl_L2_discriminative.md),
[convnext_tiny__tl_L2_discriminative](runs/convnext_tiny__tl_L2_discriminative.md) · **Figure:** [`outputs/figures/track2_L0_vs_L2.png`](../outputs/figures/track2_L0_vs_L2.png)
· **Commands:** `python -m birdcls.train --model <model> --config tl_L2_discriminative`

## Question and hypothesis (written before training)

**Change vs L1: the learning-rate policy** (`configs/tl_L2_discriminative.yaml`), treated as one step because the three
settings address the same failure:

| | L1 | **L2** |
|---|---|---|
| Backbone LR | 1e-3 | **1e-4** (10× smaller) |
| Head LR | 1e-3 | 1e-3 |
| Schedule | constant, no warm-up | **1-epoch linear warm-up, then cosine decay to 0** |

Everything else unchanged: AdamW (weight decay 0.01), basic augmentation, no label smoothing / dropout, 224 px, batch 32,
20 epochs, early stopping patience 6, same seed.

Why: L1 failed through catastrophic forgetting in the first epoch (val after epoch 1: 7.7% ResNet-50, 0.5% ConvNeXt-Tiny).
- **Discriminative LRs** (Howard & Ruder, 2018, ULMFiT): pretrained layers already hold useful features and need only small
  adjustments, while the new random head has to learn from scratch. A 10× smaller backbone LR protects the features.
- **Warm-up** keeps the first updates small while the head is still random and its gradients are large and noisy.
- **Cosine decay** lets the weights settle at the end of training (as in Track 1, S4).

Hypothesis:
- No collapse for either model; val after epoch 1 at or above L0 (32.9% / 42.8%).
- **Both models clearly above L0 (66.2% / 74.0%) and far above L1**, i.e. fine-tuning now beats feature extraction.
  The earlier A_standard runs (similar LRs plus regularisation) reached 80.5% / 86.1%, so L2 should land in that region,
  possibly a little lower (no label smoothing, dropout or stochastic depth yet).
- ResNet-50 may again be still improving at epoch 20 (as in EXP-03), which would motivate L3 (longer training).

## Results

| | ResNet-50 L0 / L1 | **ResNet-50 L2** | ConvNeXt-Tiny L0 / L1 | **ConvNeXt-Tiny L2** |
|---|---|---|---|---|
| Best val Top-1 (epoch) | 65.94 / 60.66% | **80.02%** (16 of 20) | 75.57 / 3.93% | **86.65%** (10; early stop at 16) |
| **Test Top-1** | 66.20 / 59.80% | **77.49%** | 74.00 / 4.40% | **86.13%** |
| **Test avg per class** | 65.43 / 58.85% | **76.64%** | 73.57 / 3.97% | **85.87%** |
| With TTA: val / test | | 79.81 / 78.57% | | 86.54 / 85.80% |
| Val accuracy after epoch 1 | 32.9 / 7.7% | **35.7%** | 42.8 / 0.5% | **50.3%** |
| Clean train accuracy | | 99.84% | | 99.87% |
| Overfitting gap | 29.5 / 36.5 pp | **19.8 pp** | 23.1 pp / n.a. | **13.2 pp** |
| Min val loss (epoch) | | 0.828 (15) | | 0.506 (15) |
| Training time | | 4.5 min | | 3.9 min |

Val Top-1 in this table is re-measured by `evaluate` (ResNet-50: 80.02%; the training log showed 79.92%, one image apart).

**Paired tests (McNemar, exact):**

| Comparison | Val: only A / only B, p | Test: only A / only B, p |
|---|---|---|
| ResNet-50 L0 (A) vs L2 (B) | 43 / 179, p = 6e-21 | 71 / **207**, **p = 1e-16** |
| ConvNeXt-Tiny L0 (A) vs L2 (B) | 28 / 135, p = 5e-18 | 33 / **179**, **p = 2e-25** |
| ResNet-50 L2 (A) vs ConvNeXt-Tiny L2 (B) | 35 / 99, p = 3e-8 | 34 / **138**, **p = 5e-16** |

| Epoch | 1 | 2 | 5 | 10 | 15 | 16 | 20 |
|---|---|---|---|---|---|---|---|
| Backbone LR | 1.0e-4 | 9.9e-5 | 8.9e-5 | 5.4e-5 | 1.6e-5 | 1.1e-5 | 0 |
| ResNet-50 val acc | .357 | .617 | .732 | .769 | .790 | **.799** | .790 |
| ResNet-50 train acc | .100 | .555 | .922 | .976 | .985 | .986 | .987 |
| ConvNeXt-Tiny val acc | .503 | .769 | .845 | **.866** | .863 | .856 | stopped |
| ConvNeXt-Tiny train acc | .159 | .692 | .965 | .983 | .988 | .986 | stopped |

## Observations

1. **Hypothesis confirmed: no collapse, and fine-tuning now beats feature extraction by a wide margin:** +11.3 pp
   (ResNet-50, 66.2 → 77.5%) and +12.1 pp (ConvNeXt-Tiny, 74.0 → 86.1%) test, both p < 1e-15. The same full fine-tuning that
   failed in L1 works once the backbone LR is 10× smaller and warmed up: the only difference between L1 and L2 is the LR policy.
2. **The architecture gap persists and grows: ConvNeXt-Tiny +8.6 pp over ResNet-50** (86.1 vs 77.5% test, p = 5e-16).
3. **Overfitting is now the main limit, not training length.** Both models reach about 99.9% clean train accuracy;
   ResNet-50's gap is 19.8 pp and ConvNeXt's 13.2 pp. ResNet-50's best epoch is 16 (not the last), and ConvNeXt's val
   accuracy plateaus from epoch 10 (early stopping at 16). So the original plan's next step, L3 = longer training, is not
   what the evidence asks for (see conclusion).
4. **Comparison with the earlier A_standard runs** (EXP-01/03: same LRs and schedule, plus label smoothing 0.1, dropout 0.3,
   stochastic depth 0.1, weight decay 0.05): ConvNeXt-Tiny L2 = 86.13% test, exactly A_standard's 86.13%; ResNet-50 L2 =
   77.49% vs A_standard's 80.48%. That suggests the regularisation package is worth about 3 pp for ResNet-50 but little for
   ConvNeXt-Tiny; the ladder will test this directly.
5. **TTA:** −0.2 val / +1.1 test (ResNet-50), −0.1 / −0.3 (ConvNeXt), inconsistent, so off.
6. ConvNeXt's early stop at epoch 16 cut the cosine schedule short (backbone LR still 1.1e-5), which is fine here: val had
   plateaued since epoch 10.

## Conclusion

The LR policy is the decisive fine-tuning setting: **ResNet-50 77.5%, ConvNeXt-Tiny 86.1% test**, up from 66.2% / 74.0%
(linear probe) and 59.8% / 4.4% (naive fine-tuning). With training accuracy at about 99.9% and early stopping before 20
epochs, **the evidence points to regularisation as the next step, not longer training.** Proposed order change for the rest
of the ladder: L3 = regularisation (label smoothing etc.), L4 = augmentation, then resolution, then longer training only if a
later step is still improving at its last epoch.
