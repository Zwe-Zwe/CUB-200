# Tuning summary: every step, its settings, why we did it, and why it helped

Quick review of all tuning in this project. Details and raw evidence are in the `EXP-*.md` notes listed in the last
column of each table. Numbers come from `outputs/results.csv` (no TTA).

---

## 0. Ground rules (apply to every step)

| Item | Value | Why |
|---|---|---|
| Data split | `train.txt` → stratified **80 % train / 20 % validation**, seed 42; `test.txt` (1,204 images) kept aside | every class appears in both parts; the test set is never used to make a decision |
| Decisions | made on **validation Top-1** only; McNemar's exact test when the difference is ≤ about 2 pp | prevents tuning on the test set (the sample report used test as validation, which inflates its numbers) |
| One change per step | each step changes one thing vs the previous kept step (verified against the previous config) | so each gain can be attributed to one cause |
| Final model | best recipe retrained on **train + val** (`--full-train`), last epoch kept, **test evaluated once** | 25 % more training images, without selecting on test |
| Evaluation preprocessing | resize shorter side to `size × 256/224`, centre-crop `size`, ImageNet mean/std | standard ImageNet evaluation protocol |
| Training details | mixed precision (AMP), seed 42, one GPU (RTX 4060 8 GB), checkpoint = best val epoch | reproducible: re-runs gave identical predictions |
| Metrics | Top-1 accuracy and average per-class accuracy | required by the assignment |

**Augmentation presets** used below:

| Preset | Transforms |
|---|---|
| `none` | evaluation preprocessing only |
| `basic` | RandomResizedCrop (scale 0.6–1), horizontal flip, rotation ±15°, ColorJitter 0.2, RandomErasing p = 0.25 |
| `strong` | RandomResizedCrop (scale 0.25–1), horizontal flip, TrivialAugmentWide, RandomErasing p = 0.25 |

---

## 1. Track 1: CNN from scratch (S0 → S8)

**Shared settings:** SGD momentum 0.9, LR 0.01, batch 32, 224 px, no label smoothing / dropout / CutMix.

| Step | Model | What changed (all else = previous step) | Settings | Train / Val / **Test** Top-1 | Gap (train − val) | Why we tried it | Why it helped (or not) | Note |
|---|---|---|---|---|---|---|---|---|
| S0 | `scratch_vgg` (VGG-style, 5 × [conv-conv-pool], FC 512 head, 8.9 M) | baseline | no aug, constant LR, wd 0, 50 epochs | 98.6 / 5.7 / **4.9** | 92.9 | reference point | memorises 3.9 k images and does not generalise: pure overfitting | EXP-04 |
| S1 | `scratch_vgg_bn` | **+ BatchNorm** | same | 100 / 15.1 / **14.4** | 84.9 | BN stabilises activations, so a deep net trains at LR 0.01 | better-conditioned optimisation and some regularising noise from batch statistics; still massive overfitting | EXP-05 |
| S2 | `scratch_vgg_bn` | **+ basic augmentation** | `basic` | 77.1 / 25.9 / **25.4** | 51.2 | the gap is overfitting, so make each epoch show new variants of the images | the network can no longer memorise exact pixels; gap halves | EXP-06 |
| S3 | `scratch_vgg_gap` (2.4 M) | **FC head → global average pooling** | same | 59.2 / 30.2 / **30.8** | 29.0 | the FC head held most of the parameters (6.5 M), the main memorisation capacity | 73 % fewer parameters, translation-invariant features; gap shrinks again. Val loss still falling, so under-trained | EXP-07 |
| S4 | `scratch_vgg_gap` | **warm-up 2 ep + cosine LR, up to 150 epochs, early stopping patience 20** | cosine | 99.8 / 46.9 / **46.3** | 52.9 | S3 was under-trained, and a constant LR cannot settle into a minimum | longer training plus LR decay lets the model converge finely at the end (biggest single gain: +15.5 pp) | EXP-08 |
| S5 | `scratch_vgg_gap` | **+ weight decay 5e-4** | wd 5e-4 | 99.9 / 48.0 / **48.1** | 51.9 | classic L2 regularisation against S4's gap | small gain (+1.8, p = 0.13, n.s.); gap unchanged because BatchNorm makes conv weights scale-invariant, so wd mainly changes the effective LR. Kept (val loss −11 %) | EXP-09 |
| S6 | `scratch_vgg_gap` | **basic → strong augmentation** | `strong` | 96.7 / 49.9 / **48.3** | 46.8 | still overfitting, so use more aggressive augmentation | tie on test (p = 0.95) but less memorisation and the lowest val loss; the model is now capacity-limited | EXP-10 |
| S7 | `scratch_resnet18` (hand-written, 11.3 M) | **architecture → ResNet-18** | S6 recipe | 93.2 / 41.6 / **41.4** | 51.6 | residual connections allow a deeper network | **worse** (p = 4e-8): ResNet's early 4× downsampling (7×7 stride-2 conv + max-pool) discards the fine detail that fine-grained birds need; not enough data to train 11 M parameters from scratch. Rejected | EXP-11 |
| S8 | `scratch_vgg_gap` | **retrain S6 on train + val** | S6, 150 epochs fixed | 98.6 / – / **58.6** (avg/class 57.6) | – | more data is the strongest regulariser | +10.4 pp (p = 9e-16): 25 % more images (and more updates). Best from-scratch result | EXP-12 |

**Track 1 lesson:** from scratch, the problem is data, not the optimiser. Every gain came from fighting overfitting
(BN, augmentation, a smaller head, more data) or from training properly (cosine, longer). It peaks at 58.6 %, far below
any pretrained model, which motivates transfer learning.

---

## 2. Track 2: transfer-learning ladder (L0 → L8), two models

Models: **ResNet-50** (`resnet50.tv2_in1k`, timm, 23.9 M, BatchNorm) and **ConvNeXt-Tiny** (`convnext_tiny`,
torchvision, 28.0 M, LayerNorm), both pretrained on **ImageNet-1k only** (same data, similar size, so the comparison is
fair). Both climbed the **same ladder**; validation decided per model whether each change was kept.

**Shared settings unless stated:** AdamW, batch 32, 224 px, new linear head for 200 classes.

### 2.1 Step-by-step

| Step | What changed | Settings | ResNet-50 val / **test** | ConvNeXt-T val / **test** | Why we tried it | Why it helped (or not) | Note |
|---|---|---|---|---|---|---|---|
| L0 | **Linear probe**: backbone frozen (BN in eval mode), train only the head | LR 1e-3 constant, wd 0.01, `basic`, 20 ep | 65.9 / **66.2** | 75.6 / **74.0** | measures how good the ImageNet features already are for birds | pretrained features are general but not bird-specific: a good start but limited | EXP-13 |
| L1 | **Unfreeze everything**, same single LR 1e-3 | as L0, backbone trainable | 60.7 / **59.8** (worse, p = 7.5e-5) | 3.9 / **4.4** (collapse) | fine-tuning should adapt the features | **catastrophic forgetting**: a head-sized LR (1e-3) on pretrained weights, with no warm-up, destroys them in epoch 1 (ConvNeXt), or damages them (ResNet). Shows *why* LR policy matters | EXP-14 |
| L2 | **Discriminative LRs + warm-up + cosine** | backbone **1e-4**, head **1e-3**, 1-ep warm-up, cosine, 20 ep | 80.0 / **77.5** | 86.7 / **86.1** | pretrained layers need small updates, the random head needs big ones; warm-up avoids large early steps while the head is still random | backbone is adapted gently, so no forgetting. Biggest Track 2 gain (+11 / +12 pp over L0). But train ≈ 99.9 %: now overfitting | EXP-15 |
| L3 | **Regularisation bundle** | label smoothing **0.1**, dropout **0.3** (before head), stochastic depth **0.1**, wd **0.05** | 81.4 / **80.5** (+3.0, p = 0.001) ✔ | 86.9 / **86.1** (±0) ✔ | L2 memorised the training set | label smoothing stops over-confident targets on look-alike species; dropout / drop-path add noise to features and residual paths. Helps ResNet a lot; ConvNeXt (already regularised by its design) neutral, kept for both | EXP-16 |
| L4 | **basic → strong augmentation** | `strong` | 82.5 / **80.7** ✔ kept | 86.5 / **83.7** (−2.4, p = 0.006) ✘ reverted | still a train-val gap | ResNet (BatchNorm, still improving at ep 20) benefits from more varied inputs. For ConvNeXt, crops down to 25 % often cut the bird / its field marks, so the label becomes noisy. **Recipes diverge from here** | EXP-17 |
| L5 | **Longer training** | 40 epochs, patience 10 (was 20 / 6) | 82.9 / **81.7** ✔ | 87.5 / **86.0** ✔ (n.s.) | ResNet's best epoch was the last one (under-trained, especially with strong aug) | more epochs to fit the harder augmented data; cosine decays over a longer run. Small gains, kept | EXP-18 |
| L6 | **Progressive resizing 224 → 448 px** | 30 ep at 224 px (batch 32, LR 1e-4/1e-3) + **10 ep at 448 px** (batch 16, LR **2e-5 / 2e-4**, no warm-up) | 84.6 / **83.0** ✔ | 89.4 / **88.0** ✔ (p = 0.007) | fine-grained differences (eye ring, bill, wing bars) are only a few pixels at 224 px | 4× more pixels makes small field marks visible; learning at 224 px first is cheap, then a short low-LR phase adapts to the new scale. TTA (flip) tied on val, so off | EXP-19 |
| **L7** | **Final: retrain L6 on train + val**, last epoch, test once | L6 recipe | – / **86.46** (avg/class 86.12) | – / **90.20** (avg/class 89.94) | more data | 25 % more images, especially for rare classes; +3.5 / +2.2 pp | EXP-20 |
| L8 | **Architecture-specific best practice** (fairness check, starting from L6) | ResNet: **SGD** m 0.9, wd 1e-4, LR {1e-3, 3e-3, **1e-2**} (head 10×); ConvNeXt: **layer-wise LR decay** {0.8, **0.65**} | best 84.5 / 82.7 (tie, p = 1.0) | best 89.8 / 87.8 (tie, p = 0.61) | the ladder recipe was AdamW-style; maybe it favoured ConvNeXt | **no gain for either**: the shared recipe was already near-optimal for both, so the 3.7 pp gap is the architecture, not the recipe. L6 recipes kept | EXP-21 |

✔ = kept, ✘ = reverted (decided on validation).

### 2.2 Final recipes (the models in the report and the web app)

| Hyperparameter | ResNet-50 final | ConvNeXt-Tiny final |
|---|---|---|
| Pretraining | ImageNet-1k (torchvision V2 weights) | ImageNet-1k |
| Optimiser | AdamW | AdamW |
| LR backbone / head, phase 1 (224 px, 30 ep, batch 32) | 1e-4 / 1e-3 | 1e-4 / 1e-3 |
| LR backbone / head, phase 2 (448 px, 10 ep, batch 16) | 2e-5 / 2e-4 | 2e-5 / 2e-4 |
| Schedule | 1-epoch linear warm-up + cosine (per phase) | same |
| Weight decay | 0.05 (decoupled) | 0.05 |
| Label smoothing | 0.1 | 0.1 |
| Dropout before head | 0.3 | 0.3 |
| Stochastic depth | 0.1 | 0.1 |
| Augmentation | **strong** (crop 25–100 %, TrivialAugmentWide, flip, erasing) | **basic** (crop 60–100 %, flip, rotation, colour jitter, erasing) |
| CutMix / layer decay / TTA | off / off / off | off / off / off |
| Training data | train + val (4,829 images), final epoch | same |
| Training time | 18 min | 20 min |
| **Test Top-1 / avg per class** | **86.46 % / 86.12 %** | **90.20 % / 89.94 %** |

ConvNeXt-Tiny vs ResNet-50 on the same 1,204 test images: 75 vs 30 discordant images, McNemar p = 1.3e-5, so the
difference is significant. ConvNeXt-Tiny is the best model (Top-5 98.0 %).

### 2.3 Progress at a glance (test Top-1)

| | L0 probe | L1 | L2 | L3 | L4 | L5 | L6 | **L7 final** | gain vs L0 |
|---|---|---|---|---|---|---|---|---|---|
| ResNet-50 | 66.2 | 59.8 | 77.5 | 80.5 | 80.7 | 81.7 | 83.0 | **86.5** | +20.3 |
| ConvNeXt-Tiny | 74.0 | 4.4 | 86.1 | 86.1 | (83.7 ✘) | 86.0 | 88.0 | **90.2** | +16.2 |

Figure: `outputs/figures/track2_ladder.png`.

---

## 3. Why the model improved: the four ideas behind the gains

1. **Learning-rate policy is what makes transfer learning work** (L1 → L2: +17.7 pp ResNet, +81.7 pp ConvNeXt; +11 / +12 pp over the frozen probe). Pretrained weights are
   already good, so they get a 10× smaller LR than the new head, plus warm-up and cosine decay. Without it, fine-tuning
   is worse than a frozen backbone (L1).
2. **Regularisation against memorisation** (S1–S3, S5–S6, L3–L4). With about 19 training images per class, every
   model reaches about 100 % training accuracy; label smoothing, dropout, stochastic depth, weight decay and
   augmentation trade training fit for generalisation. How much augmentation is right depends on the architecture (L4).
3. **Resolution matters for fine-grained classes** (L6, about +2 pp val for both). The differences between sibling
   species are small details; 448 px keeps them.
4. **More data beats everything** (S8 +10 pp, L7 +2 to +3.5 pp). Retraining on train + val after all decisions were
   made adds 25 % more images without touching the test set.

Plus one negative result each, which are part of the story: ResNet-18 from scratch (S7, too aggressive early
downsampling), strong augmentation for ConvNeXt (L4), and architecture-specific optimisers (L8, no gain).

## 4. What remaining errors look like (EXP-22)

About half of the final model's 118 errors are between sibling species of the same group: gulls, terns, crows and
grackles, cormorants, shrikes and sparrows. Both models struggle with the same species (per-class r = 0.67), so the
difficulty is in the data. See `outputs/error_analysis/convnext_tiny__tl_L6_hires__full/misclassified_examples.png`.

## 5. Reference (not part of the ladder)

ConvNeXt-Base pretrained on **ImageNet-22k** reaches 93.69 % test with no tuning (EXP-02). It is shown as the ceiling
from bigger pretraining data (3× the parameters, about 11× the pretraining images), not as our tuned model.
