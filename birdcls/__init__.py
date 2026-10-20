"""birdcls: fine-grained bird species classification on CUB-200 (COS30082 Assignment 1).

Modules
-------
config    project paths, experiment configuration (loaded from configs/*.yaml)
data      annotations, species names, train/val split, transforms, DataLoaders
models    ConvNeXt builders (torchvision ImageNet-1k, timm ImageNet-22k), parameter groups
augment   CutMix
engine    training loop (progressive resizing, AMP, gradient accumulation, early stopping), inference
metrics   Top-1 accuracy and average accuracy per class
plots     training curves and analysis figures
utils     seeding, device, logging, JSON helpers

Entry points (run from the project root)
----------------------------------------
python -m birdcls.train     --model convnext_base.fb_in22k --config A_standard B_finegrained
python -m birdcls.evaluate  # every finished run -> outputs/results.csv + figures + error analysis
python -m birdcls.export    # best run (by validation accuracy) -> hf_space/model.pt
"""
