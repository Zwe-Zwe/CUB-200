# Streamlit web app: upload a bird photo, get the top-5 predicted CUB-200 species.
#
# Run locally (from this folder):   streamlit run app.py      then open http://localhost:8501
#
# The model file (model.pt, made by `python -m birdcls.export`) is looked for in this order:
#   1. streamlit_app/model.pt                (copy it here when deploying the folder on its own)
#   2. hf_space/model.pt                     (where birdcls.export writes it, for local runs)
#   3. the Hugging Face Hub model repo https://huggingface.co/zwe-zwe/CUB-200 (or the MODEL_REPO environment
#      variable / Streamlit secret), downloaded once; used on Streamlit Community Cloud, because GitHub does not
#      accept files over 100 MB.
# model.pt contains the model name, trained weights, 200 class names, input size, normalisation and test metrics,
# so this file never needs editing when a different trained model is exported.
import os
import time
from pathlib import Path

import streamlit as st
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

HERE = Path(__file__).resolve().parent
EXAMPLES = sorted(HERE.glob("examples/*.jpg"))      # six test images, file name = true species

st.set_page_config(page_title="CUB-200 Bird Classifier", page_icon="🐦", layout="wide")


def find_model_file():
    """Return a local path to model.pt (see the search order at the top of this file)."""
    for p in (HERE / "model.pt", HERE.parent / "hf_space" / "model.pt"):
        if p.exists():
            return p
    repo = os.environ.get("MODEL_REPO")
    if not repo:
        try:                                         # Streamlit Cloud secret; no secrets file locally is fine
            repo = st.secrets.get("MODEL_REPO")
        except Exception:
            repo = None
    repo = repo or "zwe-zwe/CUB-200"
    if repo:
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(repo_id=repo, filename="model.pt"))
    st.error("model.pt not found: run `python -m birdcls.export` or set MODEL_REPO.")
    st.stop()


@st.cache_resource                                   # load once per server process, not on every click
def load_model():
    ckpt = torch.load(find_model_file(), map_location="cpu")
    n_cls = len(ckpt["class_names"])
    name = ckpt["model"]
    # Rebuild the same architecture as in training (no pretrained download: the weights come from model.pt)
    if "." in name:                                  # timm model, e.g. "resnet50.tv2_in1k"
        model = timm.create_model(name, pretrained=False, num_classes=n_cls)
    else:                                            # torchvision model, e.g. "convnext_tiny"
        model = getattr(models, name)(weights=None)
        model.classifier[2] = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.classifier[2].in_features, n_cls))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()                                     # inference mode: no dropout / stochastic depth
    size = ckpt["img_size"]
    # Exactly the evaluation preprocessing used in training (birdcls.data.eval_transform):
    # resize the shorter side to size * 256/224, centre-crop size x size, normalise with the training statistics
    tf = transforms.Compose([
        transforms.Resize(int(size * 256 / 224)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(ckpt.get("mean", [0.485, 0.456, 0.406]), ckpt.get("std", [0.229, 0.224, 0.225])),
    ])
    return model, tf, ckpt


@torch.no_grad()
def predict(img, model, tf, use_tta):
    """PIL image -> (200 class probabilities, milliseconds)."""
    t0 = time.perf_counter()
    x = tf(img.convert("RGB")).unsqueeze(0)          # 1 x 3 x size x size
    probs = F.softmax(model(x), dim=1)
    if use_tta:                                      # average with the mirror image (off for the final model)
        probs = (probs + F.softmax(model(torch.flip(x, dims=[3])), dim=1)) / 2
    return probs[0], (time.perf_counter() - t0) * 1000


model, tf, ckpt = load_model()
names, metrics, size = ckpt["class_names"], ckpt.get("metrics", {}), ckpt["img_size"]
n_params = sum(p.numel() for p in model.parameters()) / 1e6

# ---------------------------------------------------------------------------- page
st.title("CUB-200 Bird Species Classifier")
st.caption("COS30082 Applied Machine Learning, Assignment 1. Upload a bird photo to get the five most likely "
           "of 200 species (Caltech-UCSD Birds 200).")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Test Top-1 accuracy", f"{metrics['test_top1']:.2%}" if metrics else "-")
c2.metric("Average per-class accuracy", f"{metrics['test_avg_per_class']:.2%}" if metrics else "-")
c3.metric("Parameters", f"{n_params:.1f} M")
c4.metric("Input size", f"{size} px")

left, right = st.columns(2, gap="large")
with left:
    st.subheader("1. Choose a photo")
    upload = st.file_uploader("Upload a bird photo", type=["jpg", "jpeg", "png", "webp"])
    labels = ["(none)"] + [p.stem.replace("_", " ") for p in EXAMPLES]
    choice = st.selectbox("...or pick an example test image (not used in training)", labels)
    img = None
    if upload is not None:
        img = Image.open(upload)
    elif choice != "(none)":
        img = Image.open(EXAMPLES[labels.index(choice) - 1])
    if img is not None:
        st.image(img, width="stretch")

with right:
    st.subheader("2. Prediction")
    if img is None:
        st.info("Upload a photo or pick an example.")
    else:
        probs, ms = predict(img, model, tf, ckpt.get("tta", False))
        top_p, top_i = probs.topk(5)
        best, conf = names[top_i[0]], float(top_p[0])
        st.markdown(f"### {best}  ({conf:.0%})")
        if conf < 0.3:
            st.warning("Low confidence: similar-looking species compete, see the other candidates.")
        for p, i in zip(top_p.tolist(), top_i.tolist()):
            st.progress(p, text=f"{names[i]}: {p:.1%}")
        st.caption(f"Inference time: {ms:.0f} ms on CPU at {size} x {size} px.")

st.divider()
st.markdown(f"""
**Model.** `{ckpt['model']}` ({n_params:.1f} M parameters), pretrained on ImageNet-1k and fine-tuned on the 4,829
CUB-200 training images through a step-by-step tuning ladder (linear probe, two learning rates with warm-up and
cosine decay, regularisation, longer training, 224 to 448 px progressive resizing), then retrained on all training
data. The test set was used once, for the numbers above.

**Reading the confidence.** Training used label smoothing (0.1), which deliberately keeps confidence below 100%, so a
correct answer often shows 50 to 90%. Most mistakes are between near-identical sibling species (gulls, terns, crows
and blackbirds). The model only knows these 200 species, so any other image is still mapped to one of them.
""")
