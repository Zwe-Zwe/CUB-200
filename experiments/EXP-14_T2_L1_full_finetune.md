# EXP-14: Track 2, L1: full fine-tuning with a single learning rate

**Date:** 2026-10-17 (trained 13:09 to 13:19) · **Runs:** [resnet50.tv2_in1k__tl_L1_finetune](runs/resnet50.tv2_in1k__tl_L1_finetune.md),
[convnext_tiny__tl_L1_finetune](runs/convnext_tiny__tl_L1_finetune.md) · **Figure:** [`outputs/figures/track2_L0_vs_L1.png`](../outputs/figures/track2_L0_vs_L1.png)
· **Commands:** `python -m birdcls.train --model resnet50.tv2_in1k --config tl_L1_finetune` and
`python -m birdcls.train --model convnext_tiny --config tl_L1_finetune`

## Question and hypothesis (written before training)

**One change vs L0:** `freeze_backbone` true → **false** (`configs/tl_L1_finetune.yaml`). All 23.9 M (ResNet-50) /
28.0 M (ConvNeXt-Tiny) parameters are trained, and BatchNorm statistics (ResNet-50) update again. Everything else is
unchanged: AdamW (weight decay 0.01), **one constant LR of 1e-3 for every layer**, basic augmentation, no label smoothing /
dropout, 224 px, batch 32, 20 epochs, early stopping patience 6, same seed.

Why: in L0 the fixed ImageNet features were the ceiling (val flattened at 66% / 76%). Fine-tuning lets every layer
adapt to bird-specific patterns.

The single LR is deliberately naive. 1e-3 is the LR the *random* head needed in L0, but it is about 10× larger than
what is usually used for *pretrained* weights with AdamW, so large early updates (driven by the random head's big
gradients) can overwrite useful pretrained features ("catastrophic forgetting"). L2 will address this with separate LRs.

Hypothesis (two competing outcomes):
- **(a) Adaptation wins:** test accuracy rises above L0 (66.2% / 74.0%) because the backbone adapts.
- **(b) Forgetting wins:** the LR is too high for the pretrained layers, so early epochs lose accuracy, training is noisy,
  and the result may be **below** L0, or above it only modestly and far below the earlier tuned fine-tunes (80.5% / 86.1%).
- Expected: (b) is more likely for at least one model, so L1 may be a *step down* that motivates L2. ConvNeXt (LayerNorm,
  AdamW-trained) may tolerate the LR better than ResNet-50 (BatchNorm).

## Results

| | ResNet-50 L0 | **ResNet-50 L1** | ConvNeXt-Tiny L0 | **ConvNeXt-Tiny L1** |
|---|---|---|---|---|
| Trainable parameters | 0.41 M | 23.9 M | 0.15 M | 28.0 M |
| Best val Top-1 (epoch) | 65.94% (20) | 60.66% (14, early stop at 20) | 75.57% (18) | **3.93%** (18) |
| **Test Top-1** | 66.20% | **59.80%** (−6.4 pp) | 74.00% | **4.40%** (−69.6 pp) |
| **Test avg per class** | 65.43% | 58.85% | 73.57% | 3.97% |
| With TTA: val / test | 68.43 / 67.11% | 61.59 / 61.63% | 75.67 / 74.50% | 3.83 / 4.49% |
| Clean train accuracy | 95.42% | 97.15% | 98.71% | 5.93% |
| Overfitting gap | 29.5 pp | 36.5 pp | 23.1 pp | 2.0 pp (nothing learned) |
| Val accuracy after epoch 1 | 32.9% | **7.7%** | 42.8% | **0.5%** (chance) |
| Training time | 2.7 min | 4.6 min | 2.7 min | 4.9 min |

**McNemar ResNet-50 L1 vs L0:** test: only L0 right 224, only L1 right 147, p = 7.5e-5; val p = 0.003. L1 is
**significantly worse** than the linear probe. (ConvNeXt-Tiny's drop needs no test.)

| Epoch | 1 | 2 | 3 | 5 | 10 | 14 | 15 | 18 | 20 |
|---|---|---|---|---|---|---|---|---|---|
| ResNet-50 L1 val acc | .077 | .259 | .367 | .444 | .568 | **.607** | .572 | .559 | .569 |
| ResNet-50 L1 train acc | .057 | .222 | .399 | .604 | .835 | .897 | .922 | .922 | .923 |
| ResNet-50 L1 val loss | 4.23 | 3.07 | 2.47 | 2.19 | 1.91 | 1.85 | 2.11 | 2.42 | 2.16 |
| ConvNeXt-Tiny L1 val acc | .005 | .005 | .006 | .010 | .012 | .018 | .022 | **.039** | .034 |
| ConvNeXt-Tiny L1 val loss | 5.30 | 5.30 | 5.30 | 5.28 | 5.10 | 4.94 | 4.93 | 4.83 | 4.75 |

## Observations

1. **Hypothesis (b) confirmed: forgetting wins.** Unfreezing the backbone with the head's LR (1e-3) makes both models
   *worse* than simply keeping the features frozen. ResNet-50 loses 6.4 pp test (p = 7.5e-5), and **ConvNeXt-Tiny collapses
   completely** (74.0% → 4.4%).
2. **The damage happens in the first epoch.** After epoch 1, ResNet-50 is at 7.7% val (L0: 32.9%) and ConvNeXt-Tiny at
   0.5%, i.e. chance (L0: 42.8%). The pretrained features are overwritten before the new head has learned anything useful:
   the random head produces large, noisy gradients, and at LR 1e-3 with no warm-up these flow straight into the backbone.
3. **ResNet-50 recovers partly, ConvNeXt does not.** ResNet re-learns to 60.7% val (best epoch 14), then overfits (train 92%,
   val loss rising from epoch 14, gap 36.5 pp, the largest of all its runs). ConvNeXt stays near chance and only creeps to
   3.9% after 20 epochs, effectively training from a destroyed initialisation.
4. **My guess about robustness was wrong:** I expected ConvNeXt (LayerNorm, AdamW-trained) to tolerate the high LR better
   than ResNet-50 (BatchNorm); the opposite happened. A first candidate explanation, ConvNeXt's per-channel *layer-scale*
   parameters being tiny and therefore destroyed by Adam's roughly LR-sized steps, was **checked and refuted**: the
   pretrained layer-scale values are about 0.15 (median |w|), and conv weights have similar magnitudes in both models
   (median |w| ≈ 0.037 ConvNeXt vs 0.028 ResNet). The exact mechanism was not isolated. What the run does establish is
   that ConvNeXt's pretrained weights cannot take an un-warmed-up LR of 1e-3, consistent with the small LRs and warm-up
   used in ConvNeXt's published fine-tuning recipes.
5. **TTA:** +0.9 val / +1.8 test for ResNet-50 again (helps the weaker model); irrelevant for the collapsed ConvNeXt. Off.

## Conclusion

Full fine-tuning is not automatically better than feature extraction: **with a single LR of 1e-3 it is significantly
worse** (ResNet-50 59.8%, ConvNeXt-Tiny 4.4% vs 66.2% / 74.0% for the linear probe). This is the textbook catastrophic-
forgetting failure, and it motivates **L2: a much smaller LR for the pretrained backbone than for the new head, plus a
warm-up and cosine schedule** so the head can settle before the backbone is changed much.
