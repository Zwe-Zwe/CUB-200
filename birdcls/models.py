"""Model builders.

Model names follow three conventions:
  * "scratch_vgg", ...         CNNs defined in scratch.py, randomly initialised (no pretraining)
  * "convnext_tiny"            torchvision model with ImageNet-1k weights (1.28 M images, 1,000 classes)
  * "convnext_base.fb_in22k"   timm model "<architecture>.<pretrained tag>"; fb_in22k = ImageNet-22k
                               (~14 M images, 21,841 classes)

In both cases the pretrained feature extractor is kept and only the final classification layer is
replaced by a new, randomly initialised Dropout -> Linear(features, num_classes) head.
"""
import timm
import torch.nn as nn
from torchvision import models as tv_models

from . import config as C
from .scratch import build_scratch, is_scratch


def is_timm(model_name):
    return "." in model_name


def architecture(model_name):
    """"convnext_base.fb_in22k" -> "convnext_base"."""
    return model_name.split(".")[0]


def build_model(model_name, num_classes, pretrained=True, dropout=0.3, drop_path=0.1):
    if is_scratch(model_name):          # trained from random initialisation; `pretrained` is ignored
        return build_scratch(model_name, num_classes, dropout=dropout)
    if is_timm(model_name):
        # num_classes swaps the pretrained classifier for a new layer, drop_rate adds dropout before it,
        # drop_path_rate is stochastic depth
        return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes,
                                 drop_rate=dropout, drop_path_rate=drop_path)

    # torchvision, e.g. models.convnext_tiny(weights="IMAGENET1K_V1", stochastic_depth_prob=0.1)
    m = getattr(tv_models, model_name)(weights="IMAGENET1K_V1" if pretrained else None,
                                       stochastic_depth_prob=drop_path)
    # classifier = [LayerNorm2d, Flatten, Linear(features -> 1000)]; replace the last layer
    in_features = m.classifier[2].in_features
    m.classifier[2] = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    return m


def normalization(model_name):
    """(mean, std) the model's pretrained weights expect. timm stores them in the pretrained config
    (BiT uses 0.5 / 0.5, most others ImageNet statistics); torchvision models use ImageNet statistics."""
    if is_timm(model_name):
        cfg = timm.get_pretrained_cfg(model_name).to_dict()
        return tuple(cfg["mean"]), tuple(cfg["std"])
    return C.MEAN, C.STD


def has_batchnorm(model):
    return any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules())


# Backbone depth levels for layer-wise LR decay, from the input (level 0) to the output. Each entry lists the
# parameter-name prefixes of one level; the new classification layer (head) is not included.
DEPTH_LEVELS = {
    # torchvision ConvNeXt: features.0 = stem, features.1/3/5/7 = stages 1-4, features.2/4/6 = downsampling layers
    # (grouped with the stage they feed), classifier.0 = final LayerNorm (grouped with stage 4, so the top stage
    # keeps the full backbone LR)
    "convnext": [["features.0."], ["features.1."], ["features.2.", "features.3."], ["features.4.", "features.5."],
                 ["features.6.", "features.7.", "classifier.0."]],
    # timm ResNet: stem (conv1 + bn1), then the four stages
    "resnet": [["conv1.", "bn1."], ["layer1."], ["layer2."], ["layer3."], ["layer4."]],
}


def backbone_depth_groups(model, model_name):
    """Backbone parameters split into depth levels (list of lists, input side first) for layer-wise LR decay."""
    family = next((f for f in DEPTH_LEVELS if architecture(model_name).startswith(f)), None)
    if family is None:
        raise NotImplementedError(f"layer-wise LR decay is not defined for {model_name}")
    head_ids = {id(p) for p in head_layer(model).parameters()}
    levels = [[] for _ in DEPTH_LEVELS[family]]
    for name, p in model.named_parameters():
        if id(p) in head_ids:
            continue
        level = next(i for i, prefixes in enumerate(DEPTH_LEVELS[family]) if name.startswith(tuple(prefixes)))
        levels[level].append(p)
    return levels


def head_layer(model):
    """The new classification layer (timm: get_classifier(), torchvision: classifier[2])."""
    return model.get_classifier() if hasattr(model, "get_classifier") else model.classifier[2]


def split_params(model):
    """(backbone params, head params), so the two can get different learning rates."""
    head = list(head_layer(model).parameters())
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    return backbone, head


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def micro_batch_size(model_name, img_size, batch_size):
    """Largest batch that fits in GPU memory (see config.MAX_BATCH), capped at the requested size."""
    limit = C.MAX_BATCH.get(architecture(model_name), {}).get(img_size, batch_size)
    return min(batch_size, limit)
