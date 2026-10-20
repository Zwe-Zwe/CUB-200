"""Train one model with one or more experiment configurations.

Examples (from the project root):
    python -m birdcls.train --model convnext_base.fb_in22k --config A_standard B_finegrained
    python -m birdcls.train --model convnext_tiny --config B_finegrained --retrain
    python -m birdcls.train --model convnext_base.fb_in22k --config B_finegrained --smoke
    python -m birdcls.train --model scratch_vgg_gap --config scratch_strongaug --full-train

--full-train retrains a recipe that was already chosen on the validation split on ALL of train.txt
(train + val, 4,829 images): no validation, no early stopping, the final epoch is kept, and the run is
saved as <model>__<config>__full. Use it only as a final step, then evaluate the test set once.

Each (model, config) run is written to outputs/runs/<model>__<config>/:
    best.pt        best checkpoint by validation accuracy (+ model name, config, eval image size)
    history.json   per-epoch loss / accuracy / LR, best epoch, training time
    config.yaml    snapshot of the configuration used
    train.log      full training log
    curves.png     loss and accuracy curves
"""
import argparse
import dataclasses
import shutil

import torch
import yaml

from . import config as C
from .data import load_data
from .engine import fit
from .models import build_model, count_params
from .plots import plot_history
from .utils import (get_device, get_logger, gpu_is_busy, gpu_memory_report, run_lock, save_json,
                    set_seed)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help='"convnext_tiny" (torchvision, ImageNet-1k) or a timm name such as '
                         '"convnext_base.fb_in22k" (ImageNet-22k)')
    ap.add_argument("--config", nargs="+", required=True,
                    help=f"experiment config name(s) from configs/: {', '.join(C.available_configs())}")
    ap.add_argument("--retrain", action="store_true", help="train again even if the run is complete")
    ap.add_argument("--smoke", action="store_true",
                    help="quick pipeline check: 1 epoch per phase, 4 batches; written to a __smoke run")
    ap.add_argument("--full-train", action="store_true",
                    help="final retraining on train + val (no validation / early stopping); run name gets __full")
    ap.add_argument("--force", action="store_true",
                    help="start even if another process is already using most of the GPU memory")
    return ap.parse_args()


def train_one(model_name, cfg_name, data, device, retrain=False, smoke=False, force=False, full_train=False):
    cfg = C.load_config(cfg_name)
    name = C.run_name(model_name, cfg.name) + ("__full" if full_train else "") + ("__smoke" if smoke else "")
    out = C.run_dir(name)
    if C.is_complete(name) and not retrain and not smoke:
        print(f"[{name}] already complete, skipping (use --retrain to train again)")
        return
    if device == "cuda" and gpu_is_busy() and not force:
        raise SystemExit(f"[{name}] not started: {gpu_memory_report()}. Another process (a second "
                         "training run or a notebook kernel) is using the GPU, and training would run "
                         "out of memory. Stop it first, or pass --force.")
    if smoke:
        shutil.rmtree(out, ignore_errors=True)
        for ph in cfg.phases:
            ph.epochs = 1
    out.mkdir(parents=True, exist_ok=True)
    if full_train:   # train on all of train.txt; the validation images become training images
        data = dataclasses.replace(data, train=data.full_train)
    with run_lock(out / "train.lock"):
        _train(model_name, cfg, name, out, data, device, smoke, full_train)


def _train(model_name, cfg, name, out, data, device, smoke, full_train=False):
    (out / "history.json").unlink(missing_ok=True)   # the run only counts as complete when it finishes

    logger = get_logger(name, out / "train.log")
    logger.info(f"run {name}")
    logger.info(gpu_memory_report())
    if full_train:
        logger.info(f"FULL-TRAIN: training on train + val = {len(data.train)} images; no validation, "
                    "no early stopping, final epoch kept")
    with open(out / "config.yaml", "w") as f:
        yaml.safe_dump({"model": model_name, "full_train": full_train, **cfg.to_dict()}, f, sort_keys=False)

    set_seed()                                       # same initial head / data order for every run
    model = build_model(model_name, data.num_classes, pretrained=True,
                        dropout=cfg.dropout, drop_path=cfg.drop_path).to(device)
    logger.info(f"model {model_name}: {count_params(model):.1f} M parameters")

    hist = fit(model, model_name, cfg, data, device, out / "best.pt", logger,
               max_batches=4 if smoke else None, full_train=full_train)
    save_json(hist, out / "history.json")
    plot_history(hist, name, out / "curves.png")
    logger.info(f"saved to {out}")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    device = get_device()
    if device != "cuda":
        print("WARNING: no CUDA GPU found, training on CPU will be very slow")
    data = load_data()
    print(f"data: train {len(data.train)}, val {len(data.val)}, test {len(data.test)}, "
          f"classes {data.num_classes}")
    for cfg_name in args.config:
        train_one(args.model, cfg_name, data, device, retrain=args.retrain, smoke=args.smoke,
                  force=args.force, full_train=args.full_train)


if __name__ == "__main__":
    main()
