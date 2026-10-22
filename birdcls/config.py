"""Project paths and experiment configuration.

An experiment configuration (configs/<name>.yaml) describes *how* a model is trained: augmentation,
regularisation and one or more "phases". A phase is a block of epochs at one input resolution with its
own learning rates and LR schedule; Config B uses two phases (224 px, then 448 px).

The *model* is chosen separately on the command line, so every configuration can be run with any model
and each (model, config) pair is saved as its own run in outputs/runs/<model>__<config>/.
"""
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]          # the assignment folder
TRAIN_TXT = ROOT / "train.txt"                       # "<image name> <class label>" per line
TEST_TXT = ROOT / "test.txt"
TRAIN_IMG_DIR = ROOT / "Train"                       # unzipped Train.zip
TEST_IMG_DIR = ROOT / "Test"                         # unzipped Test.zip
CONFIG_DIR = ROOT / "configs"
# BIRDCLS_OUTPUT_DIR redirects all outputs (runs, results, figures), e.g. for a full re-run of the ladder that must
# not overwrite the original runs:  BIRDCLS_OUTPUT_DIR=outputs_rerun python -m birdcls.train ...
OUTPUT_DIR = Path(os.environ.get("BIRDCLS_OUTPUT_DIR", ROOT / "outputs"))
if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = ROOT / OUTPUT_DIR
RUNS_DIR = OUTPUT_DIR / "runs"                       # one sub-folder per trained (model, config)
FIGURES_DIR = OUTPUT_DIR / "figures"
DEPLOY_DIR = ROOT / "hf_space"                       # Hugging Face Space files
STATIC_DIR = ROOT / "hf_static"                      # free static Hugging Face Space (ONNX Runtime Web, in-browser)

# ---------------------------------------------------------------------------- global settings
SEED = 42
VAL_FRACTION = 0.2            # stratified 80/20 split of train.txt into train / validation
NUM_WORKERS = 2               # DataLoader worker processes
EVAL_BATCH_SIZE = 16

# Default input normalisation (ImageNet statistics). Each model actually uses the statistics it was
# pretrained with (models.normalization), e.g. BiT expects mean = std = 0.5.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# Largest batch that fits in GPU memory per architecture and resolution, measured on an 8 GB
# RTX 4060 Laptop GPU with mixed precision (peak training memory in comments).
# If a config asks for a larger batch, gradients are accumulated over smaller "micro-batches", so the
# effective batch size is unchanged. This is only equivalent for models WITHOUT BatchNorm (LayerNorm /
# GroupNorm), because BatchNorm statistics depend on the micro-batch; engine.fit enforces this.
MAX_BATCH = {
    "convnext_tiny":     {224: 32, 448: 16},
    "convnext_small":    {224: 32, 448: 8},
    "convnext_base":     {224: 32, 448: 8},   # 448 px: bs 8 = 5.0 GiB, bs 16 = OOM       (LayerNorm)
    "resnet50":          {224: 32, 448: 16},  # 224 px: bs 32 = 1.7 GiB, 448 px: bs 16 = 3.0 GiB (BatchNorm)
    "resnetv2_50x1_bit": {224: 32, 448: 8},   # 448 px: bs 16 = 6.2 GiB (too tight), bs 8 = 3.3 GiB (GroupNorm)
}


# ---------------------------------------------------------------------------- experiment config
@dataclass
class Phase:
    img_size: int             # square input size in pixels
    batch_size: int           # effective batch size (may be split into micro-batches)
    epochs: int
    lr_backbone: float        # learning rate for the pretrained layers (scratch models: set both LRs equal)
    lr_head: float            # learning rate for the new 200-class layer
    warmup_epochs: int = 0    # linear LR warm-up before the schedule

    def __post_init__(self):  # YAML may give "1e-4" as a string; force the right types
        self.img_size, self.batch_size = int(self.img_size), int(self.batch_size)
        self.epochs, self.warmup_epochs = int(self.epochs), int(self.warmup_epochs)
        self.lr_backbone, self.lr_head = float(self.lr_backbone), float(self.lr_head)


@dataclass
class ExperimentConfig:
    name: str
    phases: list
    description: str = ""
    cutmix_prob: float = 0.0      # probability that a training batch is CutMix-ed
    label_smoothing: float = 0.1  # softens one-hot targets, discourages over-confidence
    weight_decay: float = 0.05    # AdamW decoupled weight decay
    dropout: float = 0.3          # dropout before the new classification layer
    drop_path: float = 0.1        # stochastic depth: randomly skips residual blocks in training
    patience: int = 6             # early stopping: epochs without val improvement (per phase)
    optimizer: str = "adamw"      # "adamw" (decoupled weight decay) or "sgd" (with momentum)
    momentum: float = 0.9         # SGD only
    lr_schedule: str = "cosine"   # "cosine" (decay to 0) or "constant"
    augment: str = "basic"        # training augmentation preset: "none", "basic" or "strong" (see data.py)
    freeze_backbone: bool = False  # linear probe: train only the new head; pretrained layers (and BN stats) frozen
    layer_decay: float = 1.0      # layer-wise LR decay: depth level d gets lr_backbone * layer_decay**(top - d); 1 = off

    def __post_init__(self):
        self.phases = [p if isinstance(p, Phase) else Phase(**p) for p in self.phases]
        for k in ("cutmix_prob", "label_smoothing", "weight_decay", "dropout", "drop_path", "momentum"):
            setattr(self, k, float(getattr(self, k)))
        self.patience = int(self.patience)
        self.freeze_backbone = bool(self.freeze_backbone)
        self.layer_decay = float(self.layer_decay)
        assert 0 < self.layer_decay <= 1, self.layer_decay
        assert self.optimizer in ("adamw", "sgd"), self.optimizer
        assert self.lr_schedule in ("cosine", "constant"), self.lr_schedule
        assert self.augment in ("none", "basic", "strong"), self.augment

    def to_dict(self):
        return asdict(self)


def load_config(name_or_path):
    """Load configs/<name>.yaml (or an explicit path) into an ExperimentConfig."""
    path = Path(name_or_path)
    if not path.suffix:
        path = CONFIG_DIR / f"{name_or_path}.yaml"
    with open(path) as f:
        return ExperimentConfig(**yaml.safe_load(f))


def available_configs():
    return sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))


# ---------------------------------------------------------------------------- run naming
RUN_SEP = "__"


def run_name(model_name, config_name):
    """e.g. ("convnext_base.fb_in22k", "B_finegrained") -> "convnext_base.fb_in22k__B_finegrained"."""
    return f"{model_name}{RUN_SEP}{config_name}"


def run_dir(name):
    return RUNS_DIR / name


def is_complete(name):
    """A run is complete once both its best checkpoint and its training history exist."""
    d = run_dir(name)
    return (d / "best.pt").exists() and (d / "history.json").exists()


def completed_runs():
    return sorted(d.name for d in RUNS_DIR.glob("*") if d.is_dir() and is_complete(d.name))
