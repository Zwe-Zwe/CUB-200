---
license: mit
library_name: pytorch
pipeline_tag: image-classification
tags:
  - image-classification
  - fine-grained
  - cub-200
  - convnext
  - transfer-learning
metrics:
  - accuracy
---

# ConvNeXt-Tiny fine-tuned on CUB-200 (COS30082 Assignment 1)

Bird species classifier for the 200 classes of Caltech-UCSD Birds 200, trained for the COS30082 Applied Machine
Learning assignment (Swinburne University of Technology).

| | |
|---|---|
| Architecture | ConvNeXt-Tiny (torchvision), 28.0 M parameters |
| Pretraining | ImageNet-1k |
| Training data | 4,829 CUB-200 training images (assignment split) |
| Input | 448 × 448 RGB (resize shorter side to 512, centre crop 448, ImageNet normalisation) |
| **Test Top-1 accuracy** | **90.20 %** (1,204 held-out test images, evaluated once) |
| **Average per-class accuracy** | **89.94 %** |
| Test Top-5 accuracy | 98.01 % |

## How it was trained

The recipe was found with a step-by-step tuning ladder, where every decision was made on a stratified 20 % validation
split (never on the test set):

1. linear probe (frozen backbone): 74.0 % test
2. full fine-tuning with one LR 1e-3: collapse (catastrophic forgetting)
3. discriminative LRs (backbone 1e-4, head 1e-3) + 1-epoch warm-up + cosine: 86.1 %
4. regularisation: label smoothing 0.1, dropout 0.3, stochastic depth 0.1, AdamW weight decay 0.05
5. strong augmentation tested and rejected on validation (basic augmentation kept)
6. longer training (40 epochs)
7. progressive resizing: 30 epochs at 224 px + 10 epochs at 448 px (LR ÷ 5): 88.0 %
8. final model retrained on train + validation with the same recipe: **90.20 %**

## Files

- `model.pt`: a dict with `model` ("convnext_tiny"), `state_dict`, `class_names` (200), `img_size` (448), `mean`,
  `std`, `tta` (False) and test `metrics`.

## Usage

```python
import torch, torch.nn as nn
from torchvision import models, transforms
from huggingface_hub import hf_hub_download
from PIL import Image

ckpt = torch.load(hf_hub_download("zwe-zwe/CUB-200", "model.pt"), map_location="cpu")
model = models.convnext_tiny(weights=None)
model.classifier[2] = nn.Sequential(nn.Dropout(0.3), nn.Linear(768, len(ckpt["class_names"])))
model.load_state_dict(ckpt["state_dict"]); model.eval()

tf = transforms.Compose([transforms.Resize(512), transforms.CenterCrop(448), transforms.ToTensor(),
                         transforms.Normalize(ckpt["mean"], ckpt["std"])])
x = tf(Image.open("bird.jpg").convert("RGB")).unsqueeze(0)
print(ckpt["class_names"][model(x).argmax(1).item()])
```

## Limitations

Most remaining errors are between near-identical sibling species (gulls, terns, crows and grackles, cormorants).
The model only knows these 200 species; any other image is still assigned to one of them. Confidences are
deliberately moderate because of label smoothing.
