# EXP-13: Track 2, L0: linear probe (frozen pretrained backbone)

**Date:** 2026-10-17 (trained 12:59 to 13:05, both runs) · **Runs:** [resnet50.tv2_in1k__tl_L0_probe](runs/resnet50.tv2_in1k__tl_L0_probe.md),
[convnext_tiny__tl_L0_probe](runs/convnext_tiny__tl_L0_probe.md) · **Figure:** [`outputs/figures/track2_L0.png`](../outputs/figures/track2_L0.png)
· **Commands:** `python -m birdcls.train --model resnet50.tv2_in1k --config tl_L0_probe` and
`python -m birdcls.train --model convnext_tiny --config tl_L0_probe`

## Track 2 design (decided 2026-10-17)

Two CNNs pretrained on the **same data (ImageNet-1k)** and of **similar size** climb the **same tuning ladder**, one change
per step. Differences between the models are therefore due to architecture (classic vs modern CNN), and differences
between steps are due to the one changed setting.

| | Model A: ResNet-50 | Model B: ConvNeXt-Tiny |
|---|---|---|
| Name in code | `resnet50.tv2_in1k` (timm, torchvision V2 weights) | `convnext_tiny` (torchvision, IMAGENET1K_V1) |
| Parameters (200-class head) | 23.9 M | 28.0 M |
| Normalisation layers | BatchNorm | LayerNorm |
| Design | classic CNN (2015) | modernised ResNet (2022) |

| Step | Change (vs previous step) | Concept |
|---|---|---|
| **L0** | linear probe: frozen backbone, train only a new linear head; plain defaults (AdamW 1e-3 constant, 20 epochs) | how good are the ImageNet features on their own? |
| L1 | full fine-tuning (unfreeze), same single LR | adapt all layers |
| L2 | two LRs (backbone small, head large) + warm-up + cosine | protect pretrained features, train the head fast |
| L3 | longer training + early stopping | fix under-training |
| L4 | regularisation (label smoothing, strong augmentation) | fight overfitting |
| L5 | higher resolution (448 px) | fine detail |
| L6 | retrain on train + val, decide TTA on val, evaluate test once | use all data |

The steps may be split or reordered if a result calls for it; every decision is made on validation. The ConvNeXt-Base-22k
result (EXP-02, 93.69%) is kept as an external reference ("larger model + 11× more pretraining data").

## Question and hypothesis (written before training)

**L0 setup** (`configs/tl_L0_probe.yaml`): the pretrained backbone is frozen. Its weights get no gradient, and its
BatchNorm layers stay in eval mode, so the ImageNet running statistics are unchanged too (verified on a smoke run: 0 of 318
backbone tensors changed). Only the new linear layer is trained: 410 k parameters for ResNet-50 (2048 × 200 + 200),
154 k for ConvNeXt-Tiny (768 × 200 + 200). AdamW with default weight decay 0.01, constant LR 1e-3, basic augmentation, no
label smoothing / dropout, 224 px, batch 32, 20 epochs, early stopping patience 6.

Hypothesis:
- The frozen ImageNet features are already useful for birds (ImageNet-1k contains about 59 bird classes), so both models
  reach a reasonable accuracy, far above the from-scratch models (best 58.6%), but below the earlier full fine-tunes
  (ResNet-50 80.5%, ConvNeXt-Tiny 86.1% with the A_standard recipe).
- **ConvNeXt-Tiny > ResNet-50**, as with full fine-tuning (EXP-03: about 6 pp).
- The training/validation gap is small (only a linear layer is trained), so the limitation is the fixed features, not
  overfitting.

## Results

| | **ResNet-50 (L0)** | **ConvNeXt-Tiny (L0)** | ConvNeXt − ResNet |
|---|---|---|---|
| Trainable parameters | 0.410 M (head only) | 0.154 M (head only) | |
| Best val Top-1 (epoch) | 65.94% (20 of 20) | **75.57%** (18) | +9.6 pp |
| **Test Top-1** | 66.20% | **74.00%** | **+7.8 pp** |
| **Test avg per class** | 65.43% | **73.57%** | +8.1 pp |
| With TTA: val / test | 68.43 / 67.11% | 75.67 / 74.50% | TTA off (ladder rule) |
| Clean train accuracy | 95.42% | 98.71% | |
| Overfitting gap | 29.5 pp | 23.1 pp | |
| Min val loss | 1.45 (epoch 20, still falling) | 0.98 (epoch 18 to 20) | |
| Training time | 2.7 min | 2.7 min | |

**Paired test (McNemar, test set): both right 722, only ResNet right 75, only ConvNeXt right 169, both wrong 238;
p = 1.7e-9.** ConvNeXt-Tiny's frozen features are significantly better.

| Epoch | 1 | 2 | 5 | 10 | 15 | 18 | 20 |
|---|---|---|---|---|---|---|---|
| ResNet-50 val acc | .329 | .443 | .550 | .612 | .643 | .642 | **.659** |
| ConvNeXt-Tiny val acc | .428 | .556 | .671 | .730 | .747 | **.756** | .744 |
| ResNet-50 train acc | .137 | .493 | .749 | .865 | .905 | .922 | .930 |
| ConvNeXt-Tiny train acc | .210 | .560 | .807 | .911 | .947 | .954 | .964 |
| ResNet-50 val loss | 3.97 | 3.26 | 2.32 | 1.79 | 1.57 | 1.49 | 1.45 |
| ConvNeXt-Tiny val loss | 3.03 | 2.20 | 1.46 | 1.13 | 1.02 | 0.98 | 0.98 |

## Observations

1. **Hypothesis confirmed: frozen ImageNet features are already strong.** A single linear layer on fixed features reaches
   66.2% (ResNet-50) and **74.0%** (ConvNeXt-Tiny) test in **2.7 minutes**, above the best from-scratch CNN trained for
   27 minutes (S8: 58.6%) and far above the comparable from-scratch run S6 (48.3%). This is the core argument for transfer learning.
2. **Hypothesis confirmed: ConvNeXt-Tiny > ResNet-50 by 7.8 pp test (p = 1.7e-9)**, with the same pretraining data and
   similar size. The modern architecture's ImageNet features separate bird species much better, even untouched. It also
   learns faster (42.8% vs 32.9% val after one epoch).
3. **Hypothesis refuted: the gap is not small.** Even a single linear layer (0.15 to 0.41 M parameters) over-fits the
   ~19 images per class: train 95 to 99% vs val 66 to 76% (gaps 23 to 30 pp). With 200 classes and fixed 2048- or
   768-dimensional features, a linear classifier can nearly separate the training set while generalising much less.
4. **ResNet-50 is still improving at epoch 20** (best epoch 20, val loss still falling), so even the probe is
   under-trained for ResNet. ConvNeXt has flattened out (best epoch 18).
5. **The ceiling is the fixed features:** both curves flatten well below the earlier full fine-tunes (ResNet-50 80.5%,
   ConvNeXt-Tiny 86.1%), which motivates L1, letting the backbone adapt.
6. **TTA:** clearly helps ResNet-50 (+2.5 val, +0.9 test) but not ConvNeXt (+0.1 / +0.5). The same pattern as EXP-03: TTA
   helps the weaker model. Off for the ladder, decided at the end (L6).

## Conclusion

L0 baselines: **ResNet-50 66.2%, ConvNeXt-Tiny 74.0% test.** Pretrained features beat every from-scratch model at a
fraction of the cost, and the modern CNN's features are clearly better. Next, **L1: unfreeze the backbone (full
fine-tuning) with the same single learning rate**.
