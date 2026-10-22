# COS30082 Assignment 1: Progress Log

Bird species classification on CUB-200 (200 classes, 4,829 train / 1,204 test images).

**Detailed evidence:** [`experiments/`](experiments/README.md) has one note per experiment (`EXP-*.md`: question, setup, results, conclusions) and an auto-generated run card per training run (`experiments/runs/`). This file is the short summary and timeline.

## Setup

| Item | Value |
|---|---|
| Hardware | NVIDIA GeForce RTX 4060 Laptop GPU (8 GB), local machine |
| Software | torch 2.14.1+cu130, torchvision 0.29.1, timm 1.0.30 (`.venv`) |
| Code | `birdcls/` package + `configs/*.yaml` (see README.md). Run 1 used the original notebook (`archive/`) |
| Data split | Stratified 80/20 of `train.txt`: 3,863 train / 966 val (seed 42) |
| Images per class (train) | min 16, max 31, mean 24.1 |
| Test set | 1,204 images, 4 to 8 per class (used only for final numbers) |
| Selection rule | Every decision (early stopping, checkpoint, best run, TTA on/off) uses **validation** accuracy |

## Configurations compared

| Setting | Config A: Standard fine-tune | Config B: Fine-grained fine-tune |
|---|---|---|
| Input resolution | 224 px throughout | Progressive: 224 px, then 448 px |
| Epochs | 20 | 15 at 224 px + 5 at 448 px |
| Batch size | 32 | 32 at 224 px, 16 at 448 px |
| Train augmentation | RandomResizedCrop, flip, rotation, colour jitter, random erasing | Same + CutMix (p = 0.5) |
| Eval preprocessing | Resize 256, CenterCrop 224 | Resize 512, CenterCrop 448 |
| Loss | Cross-entropy, label smoothing 0.1 | Same (on CutMix's mixed labels) |
| Optimiser | AdamW, weight decay 0.05 | AdamW, weight decay 0.05 |
| Learning rate | backbone 1e-4, head 1e-3 | Same, then backbone 2e-5 / head 2e-4 at 448 px |
| LR schedule | 1 warm-up epoch, then cosine (per batch) | Same, plus a fresh cosine cycle at 448 px |
| Regularisation | Dropout 0.3 (head), stochastic depth 0.1 | Same + CutMix |
| Early stopping | Patience 6 on val accuracy | Patience 6 on val accuracy |

---

## Run 1: ConvNeXt-Tiny (ImageNet-1k), 2026-10-06

28.0 M parameters. Trained with the original notebook (kept as `archive/notebook_convnext_tiny_run.ipynb`).
Re-evaluated with the new `birdcls.evaluate`: every number below is reproduced exactly.

### Results

| Run | Eval size | Best epoch | Train acc* | Val Top-1 | Gap (train − val) | **Test Top-1** | **Test avg per class** | Train time |
|---|---|---|---|---|---|---|---|---|
| A_standard | 224 | 17 / 20 | 99.90% | 86.85% | 13.04 pp | 86.13% | 85.71% | 6.9 min |
| A_standard + TTA | 224 | 17 / 20 | 99.90% | 86.65% | 13.25 pp | 86.88% | 86.68% | (same) |
| **B_finegrained** ✅ | 448 | 19 / 20 | 99.84% | **87.16%** | 12.68 pp | 86.63% | 86.39% | 10.2 min |
| B_finegrained + TTA | 448 | 19 / 20 | 99.84% | 86.44% | 13.41 pp | 87.04% | 86.86% | (same) |

\* Train accuracy measured with eval preprocessing (no augmentation, no CutMix), so A and B are comparable.

✅ Selected by validation accuracy and exported to `hf_space/model.pt` (448 px, TTA off).

### Validation accuracy per epoch

| Epoch | 1 | 2 | 3 | 5 | 7 | 10 | 12 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A (224) | .405 | .723 | .813 | .848 | .854 | .855 | .849 | .862 | .863 | **.869** | .865 | .867 | .867 |
| B | .339 | .685 | .777 | .823 | .827 | .843 | .855 | .853 | .858† | .859† | .862† | **.872†** | .871† |

† 448 px phase.

Validation loss: A reaches its minimum of 0.741 at epoch 7, then rises to 0.839 while accuracy keeps improving slightly.
B's validation loss falls during the 448 px phase and ends at 0.758.

### Observations

1. **A and B are statistically tied.** B is +0.3 pp on val and +0.5 pp on test (about 6 of 1,204 images).
   The standard error of test accuracy is about ±1 pp (√(0.87 × 0.13 / 1204)), so a single run cannot separate them.
2. **The 448 px phase clearly helped within B.** B at the end of its 224 px + CutMix phase was at 85.3% val
   (A was at 86.2% at the same epoch). Five epochs at 448 px raised it to 87.2% (+1.9 pp).
   CutMix slows convergence at 224 px; resolution recovers and passes it.
3. **Overfitting is present** (train about 99.9%, val about 87%, gap about 13 pp). CutMix barely reduced the clean gap
   (12.7 vs 13.0 pp).
4. **B is better calibrated.** A's rising val loss with flat accuracy shows growing over-confidence on its errors.
   B's val loss stays lower (0.758 vs 0.839).
5. **TTA (horizontal-flip averaging) is inconclusive.** It raises test accuracy (+0.75 A, +0.4 B) but lowers val
   accuracy (−0.2 A, −0.7 B). Following the validation rule, TTA is **off**. Do not report 87.04% as the result.
6. **Early stopping never triggered**; B's best epoch was its last-but-one, so it may still have been improving.
7. Harmless warning: "`lr_scheduler.step()` before `optimizer.step()`". The AMP GradScaler skipped the first
   step because of fp16 overflow, so the schedule is one step ahead. No real effect.

### Error analysis (B_finegrained)

Worst classes (test accuracy; only 4 to 8 images per class, so one error = 12 to 25 pp):

| Class | Acc | Class | Acc |
|---|---|---|---|
| Ring-billed Gull | 0.20 | Red-eyed Vireo | 0.43 |
| American Crow | 0.40 | Boat-tailed Grackle | 0.43 |
| Elegant Tern | 0.40 | Red-faced Cormorant | 0.50 |
| Bank Swallow | 0.40 | Whip-poor-will | 0.50 |
| Arctic Tern | 0.43 | Belted Kingfisher | 0.50 |

Most common confusions (true → predicted):

| Count | Confusion |
|---|---|
| 4 | Boat-tailed Grackle → Brewer's Blackbird |
| 3 | Ring-billed Gull → Herring Gull |
| 3 | Loggerhead Shrike → Great Grey Shrike |
| 3 | Elegant Tern → Common Tern |
| 2 | American Crow → Fish Crow |
| 2 | California Gull → Ring-billed Gull |
| 2 | Common Raven → American Crow |
| 2 | Clay-colored Sparrow → Field Sparrow |
| 2 | Arctic Tern → Caspian Tern |
| 2 | Common Tern → Black Tern |

Errors concentrate in visually near-identical groups (gulls, terns, crows/ravens, blackbirds/grackles, shrikes),
which is what a fine-grained problem predicts.

### Files

| File | Content |
|---|---|
| `outputs/runs/convnext_tiny__A_standard/`, `outputs/runs/convnext_tiny__B_finegrained/` | `best.pt`, `history.json`, `curves.png` |
| `archive/results_convnext_tiny_run1.csv` | Results table above (original notebook output) |
| `archive/notebook_convnext_tiny_run.ipynb` | Executed notebook with all Run 1 outputs and plots |

---

## Run 2: ConvNeXt-Base pretrained on ImageNet-22k (`timm: convnext_base.fb_in22k`)

88 M parameters. Trained with `python -m birdcls.train --model convnext_base.fb_in22k --config A_standard B_finegrained`,
evaluated with `python -m birdcls.evaluate`.

### Results (all runs)

| Run | Eval size | Best epoch | Train acc | Val Top-1 | Gap | **Test Top-1** | **Test avg/class** | Time |
|---|---|---|---|---|---|---|---|---|
| Base-22k A | 224 | 11 (early stop at 17) | 99.92% | 93.17% | 6.75 pp | 93.19% | 92.96% | 10.7 min |
| Base-22k A +TTA | 224 | 11 | 99.92% | 93.37% | 6.55 pp | 93.02% | 92.88% | |
| **Base-22k B** ✅ | 448 | 18 / 20 | 99.90% | **94.62%** | **5.28 pp** | **93.69%** | **93.55%** | 19.6 min |
| Base-22k B +TTA | 448 | 18 | 99.90% | 94.41% | 5.49 pp | 93.60% | 93.47% | |
| Tiny-1k A | 224 | 17 | 99.90% | 86.85% | 13.04 pp | 86.13% | 85.71% | 6.9 min |
| Tiny-1k A +TTA | 224 | 17 | 99.90% | 86.65% | 13.25 pp | 86.88% | 86.68% | |
| Tiny-1k B | 448 | 19 | 99.84% | 87.16% | 12.68 pp | 86.63% | 86.39% | 10.2 min |
| Tiny-1k B +TTA | 448 | 19 | 99.84% | 86.44% | 13.41 pp | 87.04% | 86.86% | |

✅ Best by validation accuracy: **convnext_base.fb_in22k__B_finegrained, no TTA**.

### Validation accuracy per epoch (Base-22k)

| Epoch | 1 | 2 | 3 | 5 | 7 | 9 | 11 | 13 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A (224) | .719 | .889 | .896 | .909 | .925 | .927 | **.933** | .930 | .929 | .931 | .929 | stop | | |
| B | .672 | .859 | .907 | .917 | .933 | .932 | .935 | .936 | .936 | .935† | .934† | **.946†** | .941† | .941† |

† 448 px phase. B's val loss is also lowest in the 448 px phase (0.360 vs 0.374 at 224 px).

### Observations

1. **Base-22k vs Tiny-1k: +7.1 pp test Top-1 (86.6% → 93.7%), and test errors drop from 161 to 76 images (−53%).**
   The train/val gap halves (12.7 → 5.3 pp) even though training accuracy is the same (~99.9%). Better pretrained
   features generalise better; this reduces overfitting more than any regulariser did.
   *Confound:* model size (28 M → 88 M) and pretraining data (1k → 22k) changed together. A Base-1k run would be
   needed to separate them.
2. **Config B vs A (Base): +1.45 pp val, +0.50 pp test (6 images).** Test SE is about ±0.7 pp at 93.5%, so the test
   difference alone is within noise. But B ≥ A on both val and test for **both** models, and the 448 px phase
   adds +0.9 pp val within B (93.7% → 94.6%) with the lowest val loss of any run. That is consistent evidence that
   higher resolution helps, with modest effect size.
3. **B overfits least** (gap 5.3 vs 6.8 pp for A), consistent with CutMix + resolution acting as regularisers.
4. **TTA: still no reliable effect** (−0.21 / +0.20 pp val, −0.17 / −0.09 pp test for Base). Correctly left off.
5. **Validation optimism:** B's val (94.62%) is about 0.9 pp above its test (93.69%), because the best of 20 epochs was
   picked on val (maximum of noisy values). A's val and test agree (93.17 vs 93.19). This shows why the test set
   must not be used for selection.
6. Small val discrepancy: Base A logged 93.27% val during training but 93.17% in `evaluate` (1 of 966 images).
   Float16 numerics differ slightly with batch size (32 in training, 16 in evaluation). Not significant.

### Error analysis (Base-22k B)

Worst classes: American Crow 0.40, Common Tern 0.43, Whip-poor-will 0.50, Elegant Tern 0.60, Palm Warbler /
Louisiana Waterthrush / Western Gull / Fish Crow 0.67, Brewer's Sparrow / Bewick's Wren 0.71.

Most common confusions (max **2** per pair now, vs 4 for Tiny): American Crow → Common Raven, Western Gull →
California Gull, Whip-poor-will → Chuck-will's-widow, Common Tern → Arctic Tern, Warbling Vireo → Philadelphia
Vireo, Louisiana Waterthrush → Northern Waterthrush. Remaining errors are almost all between near-identical
sister species. Tiny's worst confusion (Boat-tailed Grackle → Brewer's Blackbird, 4×) is largely fixed (1× reverse).

Files: `outputs/results.csv` / `results.md`, `outputs/figures/val_curves.png`, `outputs/error_analysis/`.

Why: bigger model (88 M vs 28 M parameters) **and** bigger pretraining data (ImageNet-22k: ~14 M images,
21,841 classes, vs ImageNet-1k: 1.28 M images, 1,000 classes). ImageNet-22k contains far more fine-grained
categories, including many bird species. There is no ConvNeXt-Base ImageNet-1k result: that run was stopped
before training at the user's request, in favour of ImageNet-22k.

Setup:
- `timm` 1.0.30 (+ `safetensors`) installed in `.venv`. torchvision only has ImageNet-1k weights.
- Same configs A and B, same hyperparameters. The only difference is at 448 px: the batch of 16 does not fit in
  8 GB with Base, so it runs as 2 × 8 with gradient accumulation (equivalent, since ConvNeXt uses LayerNorm).
  Measured peak memory: 224 px batch 32 = 5.0 GiB; 448 px batch 8 = 5.0 GiB; 448 px batch 16 = OOM.

Caveat for the report: ImageNet-22k (and ImageNet-1k to a lesser degree) contains bird photos, and some CUB
images are known to overlap with ImageNet. Pretraining may therefore have seen a few test-like images.

---

## Run 3: ResNet-50 family

Goal: compare against a classic CNN. ConvNeXt was designed by "modernising" ResNet-50, so together with Runs 1
and 2 this gives a 2 × 2 comparison (architecture × pretraining data):

| | ImageNet-1k | ImageNet-21k/22k |
|---|---|---|
| **ResNet-50** (classic CNN, 24 M) | `resnet50.tv2_in1k` | `resnetv2_50x1_bit.goog_in21k` (BiT-M) |
| **ConvNeXt** (modern CNN) | Tiny-1k (28 M): 86.63% test ✓ | Base-22k (88 M): 93.69% test ✓ |

- ResNet-50-1k vs ConvNeXt-Tiny-1k: similar size, same pretraining data, so the difference is mostly architecture.
- ResNet-50-1k vs BiT-21k: same size, different pretraining data, so the difference is mostly data
  (BiT uses ResNetV2: pre-activation blocks, GroupNorm + weight standardisation, a minor architecture difference).

Preparation (2026-10-06, 20:50–20:57):
- Measured training memory: ResNet-50 at 224 px bs 32 = 1.7 GiB, at 448 px bs 16 = 3.0 GiB (fits, no accumulation);
  BiT at 224 px bs 32 = 3.3 GiB, at 448 px bs 16 = 6.2 GiB (too tight), so 2 × 8 accumulation.
- **ResNet-50 uses BatchNorm**: gradient accumulation would change its statistics, so `engine.fit` now refuses
  accumulation for BatchNorm models. BiT uses GroupNorm, so accumulation is exact.
- **BiT expects mean = std = 0.5**, not ImageNet statistics. Normalisation is now taken per model from its timm
  pretrained config, saved in each checkpoint and in `hf_space/model.pt`, and used by `app.py`.
- BiT's features are about 6× larger (std 2.27 vs 0.19 to 0.39), so its new head starts with a higher loss (6.0 vs 5.3).
  A 2-epoch check showed it learns fine with the shared recipe (val 79.0% after 2 epochs; ResNet-50-1k: 60.3%),
  so no model-specific changes were made.
- Smoke tests (Config B, both phases) passed for both models.

Trained by the user (`python -m birdcls.train`, 2026-10-06/07), evaluated with `python -m birdcls.evaluate`
on 2026-10-08 (all 8 runs re-evaluated together; earlier runs reproduce their previous numbers exactly).

### Results

| Run | Best epoch | Train acc | Val Top-1 | Gap | **Test Top-1** | **Test avg/class** | Time |
|---|---|---|---|---|---|---|---|
| ResNet-50-1k A | 20 / 20 | 99.90% | 81.37% | 18.5 pp | 80.48% | 79.87% | 4.8 min |
| ResNet-50-1k A +TTA | 20 | 99.90% | **82.30%** | 17.6 pp | **81.64%** | 81.22% | |
| ResNet-50-1k B | 20 / 20 | 99.15% | 81.06% | 18.1 pp | 79.90% | 79.24% | 8.0 min |
| ResNet-50-1k B +TTA | 20 | 99.15% | 80.85% | 18.3 pp | 80.40% | 79.76% | |
| BiT-R50-21k A | 17 / 20 | 99.87% | 86.23% | 13.6 pp | 85.71% | 85.41% | 7.4 min |
| BiT-R50-21k A +TTA | 17 | 99.87% | 85.92% | 14.0 pp | 85.55% | 85.29% | |
| BiT-R50-21k B | 19 / 20 | 99.66% | 87.89% | 11.8 pp | 86.71% | 86.49% | 15.6 min |
| BiT-R50-21k B +TTA | 19 | 99.66% | **88.20%** | 11.5 pp | **86.30%** | 86.11% | |

### 2 × 2 comparison (test Top-1, no TTA, Config A / Config B)

| | ImageNet-1k | ImageNet-21k/22k | Data effect |
|---|---|---|---|
| **ResNet-50** (24 M) | 80.48 / 79.90 | BiT: 85.71 / 86.71 | **+5.2 / +6.8** |
| **ConvNeXt** | Tiny (28 M): 86.13 / 86.63 | Base (88 M): 93.19 / 93.69 | +7.1 / +7.1 (size also changes) |
| **Architecture effect** (same size, same data) | **+5.7 / +6.7** | | |

### Observations

1. **Architecture matters: ConvNeXt-Tiny beats ResNet-50 by about 6 to 7 pp** with similar size (28 M vs 24 M) and the same
   ImageNet-1k data. ResNet-50 also overfits most (gap about 18 pp vs 13 pp). Well beyond noise (test SE about ±1.1 pp).
2. **Pretraining data matters as much: BiT-21k beats ResNet-50-1k by about 5 to 7 pp** at the same size. Caveat: BiT is
   ResNetV2 (pre-activation, GroupNorm + weight standardisation), so this is not a perfectly pure data comparison.
3. **A classic CNN with 21k pretraining matches a modern CNN with 1k pretraining**: BiT-21k B 86.71% vs ConvNeXt-Tiny-1k
   B 86.63%. The two effects are of similar size; ConvNeXt-Base-22k gets both (plus 3× the parameters) and leads by about 7 pp.
4. **Config B helps every model except ResNet-50-1k** (BiT +1.0, ConvNeXts +0.5 test; ResNet-50 −0.6 test, −0.3 val).
   Likely reasons: ResNet-50 was still improving at its last epoch in **both** configs (best epoch 20/20, val still
   rising), so it is under-trained with this schedule, and B gives it only 15 epochs at 224 px with CutMix, which
   slows learning further (its clean train acc is 99.15% vs 99.90% in A). BatchNorm statistics must also adapt
   to the 448 px resolution change. **Fairness caveat:** the shared recipe (AdamW, LR 1e-4 backbone, 20 epochs) suits
   ConvNeXt. ResNets are usually fine-tuned longer or with SGD at a higher LR, so ResNet-50 may be slightly
   under-represented.
5. **TTA helps the weakest model only**: ResNet-50-1k A +0.93 val / +1.16 test (the first consistent TTA gain), while
   it remains noise for BiT and the ConvNeXts. Plausibly, stronger models are already more flip-invariant.
6. Validation is again slightly optimistic for every run (val above test by 0.4 to 1.2 pp), as expected when the best
   epoch is picked on val.
7. Speed: ResNet-50 is fastest (4.8 min for A); BiT is slowest per parameter (GroupNorm + weight standardisation),
   15.6 min for B.

**Overall best is unchanged: ConvNeXt-Base-22k, Config B (val 94.62%, test 93.69%).**

---

## Direction change (2026-10-08)

Pretrained models reach 90%+ with almost no tuning, which leaves little to discuss. The main story becomes
**improving a CNN step by step using CNN knowledge and hyperparameter tuning**, with the pretrained results kept
as the reference ceiling:

- **Track 2 (chosen as the main track):** tune the pretrained ResNet-50-1k (80.5% test, visibly under-trained)
  with longer training, SGD vs AdamW, layer-wise LR / freezing, resolution, augmentation.
- **Track 1 (first, as an experiment):** a CNN from scratch, built up as an ablation ladder: one change per
  step, each measured on validation.

## Track 1: CNN from scratch (ablation ladder)

| Step | Model | Config | Change | Val Top-1 | Notes |
|---|---|---|---|---|---|
| S0 | `scratch_vgg` | `scratch_base` | Plain VGG-style CNN, no BN, no augmentation, SGD constant LR 0.01, 50 epochs | **5.69%** (epoch 43) | test 4.90% / avg-class 4.78%; clean train acc 98.6%, **gap 92.9 pp**; 8.1 min |
| S1 | `scratch_vgg_bn` | `scratch_base` | + BatchNorm | **15.11%** (epoch 38) | test 14.37% / avg-class 13.81% (2.9× S0); train 99.97%, gap 84.9 pp; 9.1 min. See EXP-05 |
| S2 | `scratch_vgg_bn` | `scratch_aug` | + data augmentation (crop, flip, rotation, colour jitter, erasing) | **25.88%** (epoch 44) | test **25.42%** / avg-class 24.62% (+11.1 pp); clean train 77.1%, gap 51.2 pp; val loss min at epoch 34 (vs 7); 9.1 min. See EXP-06 |
| S3 | `scratch_vgg_gap` | `scratch_aug` | Global average pooling + 1 linear layer instead of the FC head (8.88 M → 2.41 M params) | **30.23%** (epoch 45) | test **30.81%** / avg-class 30.04% (+5.4 pp); clean train 59.2%, gap 29.0 pp; val loss still falling at epoch 45; 8.7 min. See EXP-07 |
| S4 | `scratch_vgg_gap` | `scratch_cosine` | Warm-up (2 ep) + cosine LR, up to 150 epochs, early stopping patience 20 (did not trigger) | **46.89%** (epoch 143) | test **46.26%** / avg-class 45.31% (+15.5 pp); clean train 99.8%, gap 52.9 pp; val loss min at epoch 148; 26.2 min. See EXP-08 |
| S5 | `scratch_vgg_gap` | `scratch_wd` | + weight decay 5e-4 | **48.03%** (epoch 139) | test **48.09%** / avg-class 47.42% (+1.8 pp, McNemar p = 0.13: not significant); gap 51.9 pp (unchanged); val loss 2.68 → 2.39; 27.8 min. See EXP-09 |
| S6 | `scratch_vgg_gap` | `scratch_strongaug` | Strong augmentation: crop 25–100%, TrivialAugmentWide, flip, erasing | **49.90%** (epoch 135) | test **48.26%** / avg-class 46.90%: tie with S5 (McNemar p = 0.95); aug train 72% (S5 97%), gap 46.8 pp, val loss 2.14 (lowest); now capacity / length limited; 26.2 min. See EXP-10 |
| S7 | `scratch_resnet18` | `scratch_strongaug` | ResNet-18 written from scratch (11.28 M params, residual blocks) | 41.61% (epoch 149) | test **41.36%** / avg-class 40.64%: **significantly worse than S6** (−6.9 pp, McNemar p = 4e-8); aug train 63% (S6 72%); 19.8 min. First attempt interrupted at epoch 61; re-run reproduced it exactly. See EXP-11 |
| S8 | `scratch_vgg_gap` | `scratch_strongaug` + `--full-train` | S6 recipe retrained on train + val (4,829 images), no validation / early stopping, final epoch | n/a (val used for training) | test **58.64%** / avg-class **57.60%** (+10.4 pp vs S6, McNemar p = 9e-16): **best from-scratch model**; caveat: 25% more updates too; 27.4 min. See EXP-12 |

`scratch_vgg` (`birdcls/scratch.py`): 5 blocks of (conv3×3-ReLU-conv3×3-ReLU-maxpool), channels
32-64-128-256-256, then flatten (256×7×7) → FC 512 → FC 200. **8.88 M parameters, 6.53 M (74%) of them in the
FC head**, the memorisation risk that later steps (global average pooling, augmentation) target. He (Kaiming)
initialisation. Each epoch takes about 10 s on the RTX 4060.

Code added: `birdcls/scratch.py`, config options `optimizer` (adamw/sgd), `momentum`, `lr_schedule`
(cosine/constant), `augment` (none/basic). Defaults reproduce the earlier behaviour, so existing configs are unchanged.

**S0 result (2026-10-08):** a textbook case of overfitting.
- **Slow start:** at chance (0.5%) for about 4 epochs. Without BatchNorm the deep plain network needs time before the
  gradients become useful.
- **Memorisation:** train accuracy 0.5% → 23% (epoch 10) → 87% (20) → 99.9% (50).
- **No generalisation:** val accuracy reaches about 4% by epoch 11 and stays in the 3 to 6% band (noisy) for the
  remaining 40 epochs. Val loss is lowest at epoch 7 (4.96) and then **explodes to 13.8**: the network becomes
  confidently wrong on unseen birds.
- Test 4.90% is about 10× chance (0.5%), so a little signal is learned, but the 6.5 M-parameter FC head
  memorises about 19 images per class instead of learning species features.
- TTA: +0.2 pp val, +0.1 pp test (noise).

Probe: after 2 epochs the plain network is still at chance (loss 5.29 vs ln 200 = 5.30). Deep plain networks
without BatchNorm often stall at the start.

---

## Track 2: transfer-learning ladder, ResNet-50-1k vs ConvNeXt-Tiny-1k (from 2026-10-17)

Two CNNs pretrained on the same data (ImageNet-1k), of similar size (23.9 M / 28.0 M parameters), climb the same ladder,
one change per step (details in EXP-13). ConvNeXt-Base-22k (EXP-02, 93.69%) is kept as an external reference.

| Step | Change | ResNet-50: val / **test** | ConvNeXt-Tiny: val / **test** | Note |
|---|---|---|---|---|
| L0 | Linear probe: frozen backbone, new linear head; AdamW 1e-3 constant, 20 epochs | 65.94 / **66.20%** | 75.57 / **74.00%** | ConvNeXt +7.8 pp (p = 1.7e-9); gap 23 to 30 pp even for a linear head. EXP-13 |
| L1 | Full fine-tuning (unfreeze), same single LR 1e-3 | 60.66 / **59.80%** | 3.93 / **4.40%** | **Worse than L0** (ResNet p = 7.5e-5); ConvNeXt collapses to chance in epoch 1: catastrophic forgetting. EXP-14 |
| L2 | Backbone LR 1e-4 / head 1e-3 + 1-epoch warm-up + cosine | 80.02 / **77.49%** | 86.65 / **86.13%** | +11.3 / +12.1 pp over L0 (p < 1e-15); ConvNeXt +8.6 pp over ResNet; train 99.9%: overfitting is next. EXP-15 |
| L3 | + label smoothing 0.1, dropout 0.3, stochastic depth 0.1, weight decay 0.05 (= A_standard) | 81.37 / **80.48%** | 86.85 / **86.13%** | ResNet +3.0 pp (p = 0.001), ConvNeXt ±0; reproduces A_standard exactly. EXP-16 |
| L4 | + strong augmentation (TrivialAugment, crop 25–100%) | 82.51 / **80.73%** (kept) | 86.54 / **83.72%** (reverted) | ResNet n.s. gain, still improving at epoch 20; ConvNeXt worse on test (p = 0.006), so it keeps L3. EXP-17 |
| L5 | Longer training: 40 epochs, patience 10 (ResNet on L4, ConvNeXt on L3) | 82.92 / **81.73%** | 87.47 / **85.96%** | small n.s. gains, kept; both plateau at 224 px. EXP-18 |
| L6 | Progressive resolution: 30 epochs at 224 px + 10 at 448 px | 84.58 / **82.97%** | 89.44 / **88.04%** | +1.2 / +2.1 pp (ConvNeXt p = 0.007); best recipes. EXP-19 |
| **L7** | **Final: L6 recipe retrained on train + val, test once (TTA off)** | n/a / **86.46%** | n/a / **90.20%** | +3.5 / +2.2 pp over L6 (p = 3e-5 / 0.003); avg per class 86.12 / 89.94%. EXP-20 |

| L8 | Architecture-specific tuning on L6 (train split): ResNet-50 SGD + momentum, LR sweep 1e-3 / 3e-3 / 1e-2; ConvNeXt-Tiny layer-wise LR decay 0.8 / 0.65 | | | ResNet best SGD (1e-2): 84.47 / **82.72%** | best decay (0.65): 89.75 / **87.79%** | **ties with L6** (McNemar p = 1.0 / 0.61 on val): own best practice changes nothing; ResNet keeps L6, ConvNeXt decision pending (net +3 val images). EXP-21 |

Final Track 2 figure: `outputs/figures/track2_ladder.png`. Total gain over the linear probe: ResNet-50 +20.3 pp, ConvNeXt-Tiny +16.2 pp.

---

## Full re-run of the ladder (2026-10-21, EXP-23)

All 21 Track 2 runs retrained from scratch overnight (`scripts/rerun_ladder.sh`, outputs in `outputs_rerun/`):
identical validation and test accuracy, best epochs and per-image test predictions in every run. Final models
reproduce exactly (ConvNeXt-Tiny 90.20%, ResNet-50 86.46%).

## Error analysis of the final models (2026-10-20, EXP-22)

ConvNeXt-Tiny final: Top-5 98.01%, 118 errors, 125 classes at 100%, worst class 25% (Chuck-will's-widow, n = 4).
ResNet-50 final: Top-5 96.76%, 163 errors. About half of the errors stay inside the same bird group (gulls, terns,
crows/blackbirds, cormorants, sparrows). ConvNeXt fixes 75 of ResNet's errors and makes 30 new ones. Figures in
`outputs/error_analysis/<run>/` (misclassified_examples.png for the report).

## Static web app for the free Hugging Face Space (2026-10-20)

`python -m birdcls.export --run convnext_tiny__tl_L6_hires__full --onnx` writes `hf_static/` (index.html, app.js,
model.onnx 107 MB, meta.json, examples/, README.md with `sdk: static`). The model runs in the visitor's browser with
ONNX Runtime Web 1.20.1 (WebAssembly), so it fits the free Static Space tier.

- ONNX vs PyTorch on all 1,204 test images: **same top-1 on 1204/1204**, ONNX test Top-1 0.9020, max logit diff 5e-6.
- Browser check (local http.server): all six examples give the correct species and match the Python reference's top-1;
  confidences differ by at most about 2 points (canvas resampling vs PIL's antialiased bilinear resize). Blue Jay
  95.8%, about 1.3 s per image on one WASM thread. Phone layout checked (375 px, no horizontal scroll).

## Web app (2026-10-19)

Hugging Face now charges for Gradio Spaces (only Static Spaces are free), so the Gradio app runs **locally** for now
(user decision); a free Static-Space version (ONNX Runtime Web, in-browser) remains an option.

- Exported model: `convnext_tiny__tl_L6_hires__full` (final ConvNeXt-Tiny, 448 px, test 90.20% / 89.94%) →
  `hf_space/model.pt` (107 MB). `birdcls.export` now also stores the test metrics, which the app displays.
- `hf_space/app.py` rewritten (Gradio Blocks): model info + test accuracy header, image upload, six example test images,
  top-5 species with probabilities, prediction summary with a low-confidence note (< 30%), inference time.
- Tested: API calls (Blue Jay → Blue Jay 96%, Cardinal → Cardinal 51%) and in the browser (Pileated Woodpecker 85%);
  about 90 to 120 ms per image on CPU at 448 px.

---

## Log

| When (2026-10-06) | What |
|---|---|
| 19:25 | Run 1 (ConvNeXt-Tiny, ImageNet-1k) finished in the notebook |
| 19:40 | First Base-22k attempt via the notebook stopped. A notebook kernel in PyCharm was training at the same time and hit `CUDA out of memory` (both jobs shared the 8 GB GPU) |
| 19:50 | Code restructured from the notebook into the `birdcls/` package + YAML configs + CLIs (see README.md). Run 1 checkpoints migrated to `outputs/runs/`, notebooks moved to `archive/` |
| 19:51 | `birdcls.evaluate` reproduces Run 1 results exactly (regression check of data, model loading, metrics, TTA) |
| 19:52 | Smoke test of Base-22k Config B passed (both phases, 448 px gradient accumulation) |
| 19:53 | Run 2 training started |
| 20:03 | Run 2 Config A finished (best val 93.27%, early stop at epoch 17). Config B started |
| 20:04 | The same train command was launched a second time in a terminal while Run 2 was training, and it hit `CUDA out of memory` (2.1 GiB free). Run 2 was unaffected. Added safeguards to `train.py`: it refuses to start if the GPU is already mostly in use (`--force` overrides), and a per-run lock file (`train.lock`) blocks two processes from training the same run |
| 20:07 | Background Run 2 Config B stopped at the user's request (epoch 5/15, val 91.7%), so it can be run from the user's terminal. Its partial files are replaced when B is trained again |

---

## Ideas / next steps

- [x] ConvNeXt-Base ImageNet-22k (Run 2): best model, 93.69% test Top-1
- [ ] Export Run 2 B to `hf_space/` (`python -m birdcls.export`); current `hf_space/model.pt` is the old Tiny export
- [ ] Retrain the best setup on train + val (fixed 18 epochs), evaluate test once
- [x] Run 3: ResNet-50 (1k) and BiT-ResNet-50 (21k), configs A and B (ConvNeXt-Base-22k B still best)
- [ ] Base-1k run to separate model size from pretraining data (optional)
- [ ] Ablation: B without CutMix (separates resolution from CutMix)
- [ ] Frozen-backbone baseline (feature extraction vs fine-tuning)
- [ ] Extra seeds for A and B (mean ± std), since differences are within noise
- [ ] Deploy Hugging Face Space, record video, write report (4 pages)
