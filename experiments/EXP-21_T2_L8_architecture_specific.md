# EXP-21: Track 2, L8: architecture-specific tuning (each model's own best practice)

**Status:** prepared 2026-10-17, trained and evaluated by the user (finished 2026-10-19 20:09), analysed 2026-10-19
· **Figure:** [`outputs/figures/track2_L8.png`](../outputs/figures/track2_L8.png) · **Runs (5):**
`resnet50.tv2_in1k__tl_L8_sgd_{1e-3,3e-3,1e-2}`, `convnext_tiny__tl_L8_llrd_{080,065}`

## Why this step (fairness question, 2026-10-17)

The L0 to L7 ladder gave both models the **same candidate changes** and let validation decide for each model (equal tuning
effort). But those changes were built around a ConvNeXt-style recipe (AdamW, backbone LR 1e-4), and neither model got the
fine-tuning practice its own architecture is usually trained with. L8 adds **one architecture-specific step per model**, so
the final comparison is "each model tuned to its own best practice", and it tests whether part of the 3.7 pp gap
(EXP-20) comes from a recipe that suits ConvNeXt better.

## Question and hypothesis (written before training)

Both branches start from the model's **L6 recipe** (train split only, so validation can decide), and only the stated
setting changes (verified programmatically against L6).

### ResNet-50: SGD with momentum (3-point LR sweep)

| Setting | L6 | **L8** |
|---|---|---|
| Optimiser | AdamW | **SGD, momentum 0.9** |
| Weight decay | 0.05 (decoupled) | **1e-4** (coupled L2, standard for SGD) |
| LR backbone / head, 224 px phase | 1e-4 / 1e-3 | **{1e-3, 3e-3, 1e-2} / 10×** |
| LR at 448 px | 2e-5 / 2e-4 | **backbone / 5, head 2× backbone** (same ratios as L6) |

SGD with momentum is the classic optimiser for BatchNorm ResNets, and torchvision's ResNet weights were trained with SGD.
A sweep is needed because SGD's LR scale differs from AdamW's (SGD steps scale with the gradient; Adam's are normalised).
Hypothesis: the best SGD point is **comparable to or slightly better than AdamW** for ResNet-50 (val 84.58%); a large gain
would mean the shared recipe disadvantaged ResNet-50.

### ConvNeXt-Tiny: layer-wise learning-rate decay (2 values)

| Setting | L6 | **L8** |
|---|---|---|
| Backbone LR | 1e-4 for every layer | **top stage 1e-4; each depth level below × decay** |
| Decay | 1 (off) | **0.8** (stem 0.41×) or **0.65** (stem 0.18×) |

Depth levels (`birdcls/models.py`, `DEPTH_LEVELS`): stem; stage 1; stage 2 (+ its downsampling layer); stage 3 (+ ds);
stage 4 (+ ds + final LayerNorm). Code added: `layer_decay` config option, `backbone_depth_groups()`, per-level parameter
groups in `make_optimizer` (unit-checked: every backbone parameter in exactly one level; LR multipliers as intended).
Layer-wise decay is part of ConvNeXt's published fine-tuning recipe: early layers hold generic features and should change
least. Hypothesis: a **small gain or a tie** vs L6 (val 89.44%), since ConvNeXt-Tiny is already well tuned.

### Decision rule

For each model, the L8 run with the best **validation** accuracy is compared with L6 (McNemar on val, also reported on test).
If it beats L6 on validation, that model's final model is retrained on train + val (as L7) and the test set evaluated once;
otherwise its L7 model stays final.

## Smoke tests (CPU only, 2026-10-17)

`tl_L8_sgd_3e-3` (ResNet-50) and `tl_L8_llrd_065` (ConvNeXt-Tiny): both phases ran (224 px and 448 px). The logs show
"optimizer sgd ... lr backbone 0.003 / head 0.03 ... 448px lr 0.0006 / 0.006" and "layer-wise LR decay 0.65: backbone LR
multipliers (input -> output) [0.179, 0.275, 0.423, 0.65, 1.0]".

## Results

| Run | Best val Top-1 (epoch) | **Test Top-1** | Test avg/class | Gap | TTA val / test | Time |
|---|---|---|---|---|---|---|
| ResNet-50 **L6** (AdamW, reference) | **84.58%** (40) | **82.97%** | 82.40% | 15.4 pp | 84.58 / 83.31% | 15.7 min |
| ResNet-50 L8 SGD, LR 1e-3 | 80.43% (39) | 78.90% | 78.02% | 17.8 pp | 81.68 / 78.57% | 15.4 min |
| ResNet-50 L8 SGD, LR 3e-3 | 83.02% (39) | 81.40% | 80.69% | 16.9 pp | 83.23 / 81.89% | 15.3 min |
| ResNet-50 L8 SGD, LR 1e-2 | 84.47% (38) | 82.72% | 82.24% | 15.5 pp | 84.37 / 83.14% | 15.2 min |
| ConvNeXt-Tiny **L6** (no layer decay, reference) | 89.44% (38) | **88.04%** | 87.57% | 10.4 pp | 89.44 / 88.46% | 16.9 min |
| ConvNeXt-Tiny L8 layer decay 0.8 | 89.03% (35) | 87.04% | 86.44% | 10.8 pp | 89.34 / 87.96% | 17.1 min |
| ConvNeXt-Tiny L8 layer decay 0.65 | **89.75%** (38) | 87.79% | 87.29% | 10.1 pp | 89.23 / 87.79% | 17.1 min |

**Paired tests (McNemar, exact), L6 (A) vs L8 (B):**

| Comparison | Val: only A / only B, p | Test: only A / only B, p |
|---|---|---|
| ResNet-50 L6 vs SGD 1e-2 (best SGD) | 49 / 48, p = 1.0 | 45 / 42, p = 0.83 |
| ResNet-50 L6 vs SGD 3e-3 | 43 / 28, p = 0.096 | 47 / 28, p = 0.037 |
| ConvNeXt-Tiny L6 vs layer decay 0.65 (best) | 6 / 9, p = 0.61 | 19 / 16, p = 0.74 |
| ConvNeXt-Tiny L6 vs layer decay 0.8 | 13 / 9, p = 0.52 | 24 / 12, p = 0.065 |

## Observations

1. **ResNet-50: SGD reaches exactly the AdamW result, not more.** The best SGD point (LR 1e-2) ties L6 on both splits
   (val 49 vs 48 images, p = 1.0; test 45 vs 42, p = 0.83). Lower SGD LRs are clearly worse (3e-3: −1.6 pp test, p = 0.04;
   1e-3: −4.1 pp). Accuracy still rises with LR at the top of the sweep, so a higher LR (3e-2) might add a little; this was
   not tried.
2. **ConvNeXt-Tiny: layer-wise LR decay is a tie.** Decay 0.65 is +0.3 pp val (a net 3 images, p = 0.61) and −0.25 pp test
   (p = 0.74); decay 0.8 is slightly worse on both. ConvNeXt's own fine-tuning practice does not add accuracy on top of the
   L0 to L6 ladder.
3. **Answer to the fairness question:** giving each model its own architecture-specific best practice does **not** change the
   picture. ResNet-50 with the classic SGD recipe performs exactly as with the shared AdamW recipe, so the shared recipe did
   not disadvantage it, and **the 3.7 pp architecture gap (EXP-20) is not a recipe artefact**.
4. As in every step, TTA is inconsistent on val (+1.2 to −0.5 pp), so it stays off.

**Decision (validation rule):**
- **ResNet-50:** L6 kept (val 84.58% > 84.47%), so its final model stays **86.46%** (EXP-20).
- **ConvNeXt-Tiny:** layer decay 0.65 is nominally higher on val (89.75% vs 89.44%, net 3 of 966 images, p = 0.61). Under the strict
  "higher val wins" rule used throughout the ladder it would be adopted, which requires retraining its final model on train +
  val; treated as a tie, L7 (90.20%) stays final. User decision pending (see Progress.md).

## Conclusion

Architecture-specific tuning gives no measurable gain for either model: ResNet-50 SGD ties AdamW (82.7 vs 83.0% test), and
ConvNeXt-Tiny layer-wise decay ties L6 (87.8 vs 88.0%). Each model is therefore already at its own best practice within the
options tried, and the comparison "ConvNeXt-Tiny beats ResNet-50 by 3.7 pp" holds under both definitions of fairness.
