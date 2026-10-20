# EXP-00: Setup and methodology

## Task (from the brief, `Assignment1_v2.pdf`)

- Multi-class classification of **200 bird species** (Caltech-UCSD Birds 200, CUB-200).
- Required metrics: **Top-1 accuracy** and **average accuracy per class** (mean of per-class accuracies).
- Report (max 4 pages): Methodology, Results and Discussion. Overfitting is expected and must be addressed.
- Deliverables: report PDF, code zip (generously commented), video (max 10 min), Hugging Face web app.

## Data

| Item | Value |
|---|---|
| Training images (`train.txt`, `Train/`) | 4,829, 200 classes, **16 to 31 per class** (mean 24.1) |
| Test images (`test.txt`, `Test/`) | 1,204, **4 to 8 per class** |
| Annotation format | `<image name> <class label>` per line |
| Checks done | every listed image exists on disk, no extra files; labels 0 to 199 in both sets |
| Data issue found | 91 training files are named `..._xxx.jpg` instead of `..._<number>.jpg`. The original species-name regex failed on them, giving wrong class names in the app and error analysis for 50 classes. Fixed in `birdcls.data.species_name`, with an assertion that each label maps to one name. Accuracy was not affected |

## Split and evaluation protocol

- `train.txt` is split **80/20, stratified by class, seed 42**: **3,863 train / 966 validation**.
- The **validation set drives every decision**: early stopping, best checkpoint, best run, TTA on/off.
- The **test set is used only to report final numbers.** Using it for selection would bias the reported score upward
  (EXP-02 shows the effect: the val score of the best-of-20-epochs checkpoint is about 0.9 pp above its test score).
- Test-time augmentation (TTA) = average of the softmax of the image and its horizontal mirror. Every run is reported
  with and without TTA; the validation set decides which variant counts.
- "Train acc (clean)" = accuracy on the training images with evaluation preprocessing (no augmentation, no CutMix),
  so overfitting gaps are comparable across runs.

## Preprocessing

- Training augmentation (`augment: basic`): RandomResizedCrop(scale 0.6 to 1.0), horizontal flip, rotation ±15°,
  colour jitter 0.2, random erasing (p 0.25). `augment: none` = evaluation preprocessing only.
- Evaluation: resize the short side to `size × 256/224` (256 for 224 px, 512 for 448 px), centre crop.
- Normalisation: the statistics the pretrained weights expect (ImageNet mean/std; BiT: 0.5/0.5). Scratch models use ImageNet statistics.

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB (7.7 GiB usable) |
| Software | Python 3.12, torch 2.14.1+cu130, torchvision 0.29.1, timm 1.0.30 |
| Precision | mixed precision (float16 autocast + GradScaler) |
| Seed | 42 for split, initialisation and data order |

## Code history

1. **Notebook** (Colab-style, then local), 2026-10-06: ConvNeXt-Tiny runs (EXP-01). Kept in `archive/`.
2. **`birdcls` package**, 2026-10-06 19:50: the notebook split into modules (`config`, `data`, `models`, `augment`,
   `engine`, `metrics`, `plots`, `utils`) + CLIs (`train`, `evaluate`, `export`, `report`) + YAML configs. Regression
   check: `birdcls.evaluate` reproduced every EXP-01 number exactly.
3. Later additions: per-model normalisation, BatchNorm guard for gradient accumulation, GPU-busy check and run lock,
   scratch models (`birdcls/scratch.py`), optimiser / LR-schedule / augmentation options, merged results table,
   run cards.

## Engineering decisions and incidents

| Date | Issue | Decision / fix |
|---|---|---|
| 10-06 | ConvNeXt-Base at 448 px, batch 16 runs out of memory on 8 GB (batch 8 = 5.0 GiB) | Gradient accumulation (2 × 8), exact for LayerNorm models. `MAX_BATCH` table in `config.py` from measured peak memory |
| 10-06 | Two training processes on the GPU at once (notebook kernel + script) gave `CUDA out of memory` twice | `train.py` refuses to start if less than 70% of GPU memory is free; per-run `train.lock` |
| 10-06 | BiT expects input mean = std = 0.5, not ImageNet statistics | Normalisation read per model from timm's pretrained config, stored in checkpoints and the app export |
| 10-06 | ResNet-50 uses BatchNorm, so gradient accumulation would change its statistics | `engine.fit` refuses accumulation for BatchNorm models (ResNet-50 fits batch 16 at 448 px anyway) |
| 10-08 | After a reboot the GPU was invisible from the agent's container shell (driver libraries) | Environment-only fix (library links); no effect on code or results |
| 10-08 | `evaluate --runs X` overwrote `results.csv` with only X | Results are now merged (re-evaluated rows replaced, others kept) |
