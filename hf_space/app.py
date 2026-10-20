# Gradio web app: upload a bird photo, get the top-5 predicted CUB-200 species.
#
# Run locally (from this folder):   python app.py      then open http://127.0.0.1:7860
#
# model.pt is produced by `python -m birdcls.export` and contains everything the app needs: the model name,
# trained weights, the 200 class names, input size, normalisation statistics, TTA flag and test metrics.
# So this file never needs editing when a different trained model is exported.
import time
from pathlib import Path

import gradio as gr
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------- load the exported model
ckpt = torch.load(HERE / "model.pt", map_location="cpu")
class_names = ckpt["class_names"]
img_size = ckpt["img_size"]          # 224 or 448, whichever the final model was trained / evaluated at
use_tta = ckpt["tta"]                # average image + mirror image (decided on validation; off for the final model)
mean = ckpt.get("mean", [0.485, 0.456, 0.406])   # input statistics the model was trained with
std = ckpt.get("std", [0.229, 0.224, 0.225])
metrics = ckpt.get("metrics", {})

# Rebuild the same architecture as in training (no pretrained download: the weights come from model.pt)
name = ckpt["model"]
if "." in name:                      # timm model, e.g. "resnet50.tv2_in1k"
    model = timm.create_model(name, pretrained=False, num_classes=len(class_names))
else:                                # torchvision model, e.g. "convnext_tiny"
    model = getattr(models, name)(weights=None)
    model.classifier[2] = nn.Sequential(nn.Dropout(0.3),
                                        nn.Linear(model.classifier[2].in_features, len(class_names)))
model.load_state_dict(ckpt["state_dict"])
model.eval()                         # inference mode: no dropout / stochastic depth, BatchNorm uses running stats
n_params = sum(p.numel() for p in model.parameters()) / 1e6

# Exactly the evaluation preprocessing used in training (birdcls.data.eval_transform):
# resize the short side to img_size * 256/224, centre-crop img_size, normalise
tf = transforms.Compose([
    transforms.Resize(int(img_size * 256 / 224)),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])


# ---------------------------------------------------------------------------- prediction
@torch.no_grad()
def predict(img):
    """PIL image -> ({species: probability} for all 200 classes, short text summary)."""
    if img is None:
        return {}, "Upload a bird photo (or click one of the examples)."
    t0 = time.perf_counter()
    x = tf(img.convert("RGB")).unsqueeze(0)          # 1 x 3 x img_size x img_size
    probs = F.softmax(model(x), dim=1)
    if use_tta:
        probs = (probs + F.softmax(model(torch.flip(x, dims=[3])), dim=1)) / 2
    probs = probs[0]
    ms = (time.perf_counter() - t0) * 1000

    top_p, top_i = probs.topk(5)
    best, conf = class_names[top_i[0]], float(top_p[0])
    summary = f"### Prediction: **{best}** ({conf:.0%})\n"
    if conf < 0.3:
        summary += ("Low confidence: the model is unsure, often because similar-looking species compete "
                    "(see the other top-5 candidates).\n")
    summary += f"\n_Inference time: {ms:.0f} ms on CPU at {img_size} x {img_size} px._"
    return {class_names[i]: float(probs[i]) for i in range(len(class_names))}, summary


# ---------------------------------------------------------------------------- page
acc = (f"test Top-1 accuracy **{metrics['test_top1']:.2%}**, average per-class accuracy "
       f"**{metrics['test_avg_per_class']:.2%}** on the 1,204 held-out test images"
       if metrics else "test metrics not available")

header = f"""
# CUB-200 Bird Species Classifier
COS30082 Applied Machine Learning, Assignment 1. Upload a photo of a bird to get the five most likely species
out of **200** (Caltech-UCSD Birds 200).

**Model:** `{name}` ({n_params:.1f} M parameters), a CNN pretrained on ImageNet-1k and fine-tuned on 4,829 bird
images at {img_size} px. Final result: {acc}.
"""

notes = """
**Notes.** The model was trained with label smoothing (0.1), which deliberately keeps its confidence below 100%,
so a correct prediction often shows 50 to 90%. Photos where the bird is small or hidden are harder.
The model only knows the 200 CUB species; any other image is still mapped to one of them.
"""

examples = sorted(str(p) for p in (HERE / "examples").glob("*.jpg"))

with gr.Blocks(title="CUB-200 Bird Classifier") as demo:
    gr.Markdown(header)
    with gr.Row():
        with gr.Column():
            image = gr.Image(type="pil", label="Bird photo", height=380)
            button = gr.Button("Classify", variant="primary")
            if examples:
                gr.Examples(examples=examples, inputs=image, label="Example test images (file name = true species)")
        with gr.Column():
            summary = gr.Markdown()
            label = gr.Label(num_top_classes=5, label="Top-5 predicted species")
    gr.Markdown(notes)

    button.click(predict, inputs=image, outputs=[label, summary])
    image.change(predict, inputs=image, outputs=[label, summary])   # also classify as soon as an image is set

if __name__ == "__main__":
    demo.launch()
