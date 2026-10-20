---
title: CUB-200 Bird Classifier
emoji: 🐦
colorFrom: green
colorTo: gray
sdk: static
app_file: index.html
pinned: false
short_description: ConvNeXt-Tiny bird species classifier running in the browser
---

# CUB-200 Bird Species Classifier (COS30082 Assignment 1)

A fine-tuned ConvNeXt-Tiny (ImageNet-1k pretrained) that classifies 200 bird species (test Top-1 90.20%,
average per-class accuracy 89.94%). It runs entirely in the browser with ONNX Runtime Web, so this is a free
static Space with no server.

Files: `index.html` (page), `app.js` (preprocessing and inference), `model.onnx` (network, 107 MB),
`meta.json` (class names, input size, normalisation, metrics), `examples/` (test images).
