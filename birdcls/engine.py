"""Training loop and inference.

fit() trains one model with one ExperimentConfig. For every phase it builds loaders at that phase's
resolution, a fresh AdamW optimiser (small LR for the pretrained backbone, larger LR for the new head)
and a warm-up + cosine LR schedule. The best epoch by validation accuracy is checkpointed, and the next
phase starts from the best weights so far (progressive resizing: 224 px, then 448 px).
"""
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .augment import cutmix
from .data import make_loader
from .models import backbone_depth_groups, has_batchnorm, micro_batch_size, normalization, split_params


# ---------------------------------------------------------------------------- LR schedule
def warmup_cosine(optimizer, total_steps, warmup_steps, schedule="cosine"):
    """LR multiplier: linear warm-up (protects the pretrained weights from the random head's large
    early gradients), then cosine decay to 0 (or a constant LR). Stepped once per optimiser update."""
    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        if schedule == "constant":
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def make_optimizer(cfg, backbone, head, lr_backbone, lr_head, depth_groups=None):
    """AdamW (adaptive, decoupled weight decay) or SGD with momentum.

    Parameter groups: the backbone (one group, or one group per depth level with layer-wise LR decay), then the
    new head. With layer-wise decay the TOP backbone level gets lr_backbone and each level below it gets
    `layer_decay` times the LR of the level above: early layers hold generic features (edges, textures) that should
    change least, later layers hold task-specific features that need to adapt most (ConvNeXt / BEiT fine-tuning)."""
    groups = [{"params": head, "lr": lr_head}]
    if not cfg.freeze_backbone:
        if depth_groups is None:
            groups.insert(0, {"params": backbone, "lr": lr_backbone})
        else:
            top = len(depth_groups) - 1
            groups = [{"params": ps, "lr": lr_backbone * cfg.layer_decay ** (top - d)}
                      for d, ps in enumerate(depth_groups)] + groups
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(groups, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    return torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)


# ---------------------------------------------------------------------------- inference
@torch.no_grad()
def predict_probs(model, loader, device, tta=False, max_batches=None):
    """Class probabilities for every image in the loader.

    tta=True (test-time augmentation) averages the probabilities of the image and its mirror image.
    """
    model.eval()                                   # disables dropout / stochastic depth
    use_amp = device == "cuda"
    all_p, all_y = [], []
    for i, (imgs, labels) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        imgs = imgs.to(device, non_blocking=True)
        with torch.autocast(device_type=device, enabled=use_amp):
            p = F.softmax(model(imgs).float(), dim=1)
            if tta:
                p = (p + F.softmax(model(torch.flip(imgs, dims=[3])).float(), dim=1)) / 2
        all_p.append(p.cpu())
        all_y.append(labels)
    return torch.cat(all_p), torch.cat(all_y)


def evaluate(model, loader, device, tta=False, max_batches=None):
    """(cross-entropy loss, top-1 accuracy). The loss is plain CE (no label smoothing)."""
    probs, y = predict_probs(model, loader, device, tta, max_batches)
    loss = F.nll_loss(torch.log(probs.clamp_min(1e-8)), y).item()
    acc = (probs.argmax(1) == y).float().mean().item()
    return loss, acc


# ---------------------------------------------------------------------------- training
def fit(model, model_name, cfg, data, device, ckpt_path, logger, max_batches=None, full_train=False):
    """Train `model` with ExperimentConfig `cfg`. Returns the training history (dict).

    max_batches limits batches per epoch (smoke tests only).
    full_train=True: data.train already contains the validation images (final retraining on all of
    train.txt). There is then nothing left to validate on, so validation is skipped (logged as NaN),
    early stopping is off, and the checkpoint is the FINAL epoch. All choices (recipe, number of epochs)
    must have been made beforehand on the validation split.
    """
    use_amp = device == "cuda"
    norm = normalization(model_name)           # input statistics the pretrained weights expect
    backbone, head = split_params(model)
    depth_groups = None
    if cfg.layer_decay < 1.0:
        depth_groups = backbone_depth_groups(model, model_name)
        assert sum(len(g) for g in depth_groups) == len(backbone), "every backbone parameter needs a depth level"
    if cfg.freeze_backbone:
        # Linear probe / feature extraction: the pretrained layers are not trained at all. Their weights get no
        # gradient, and their BatchNorm layers stay in eval mode (see below) so the ImageNet running statistics
        # are kept too. Only the new classification layer learns.
        for p in backbone:
            p.requires_grad_(False)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
            "img_size": [], "lr_backbone": [], "phase_starts": []}
    best = {"val_acc": -1.0}
    t0 = time.time()

    for p_i, ph in enumerate(cfg.phases, 1):
        hist["phase_starts"].append(len(hist["val_acc"]))

        # Gradient accumulation: if the phase's batch does not fit in GPU memory, run `accum`
        # micro-batches and update the weights once (same effective batch size).
        micro_bs = micro_batch_size(model_name, ph.img_size, ph.batch_size)
        accum = ph.batch_size // micro_bs
        assert micro_bs * accum == ph.batch_size, "batch_size must be a multiple of the micro-batch"
        if accum > 1 and has_batchnorm(model):
            raise ValueError(f"{model_name} uses BatchNorm, so gradient accumulation ({accum} x {micro_bs}) "
                             f"would change its statistics. Use a batch size that fits in memory.")
        pin = device == "cuda"
        train_loader = make_loader(data.train, ph.img_size, micro_bs, train=True, pin_memory=pin, norm=norm,
                                   augment=cfg.augment)
        val_loader = make_loader(data.val, ph.img_size, micro_bs, train=False, pin_memory=pin, norm=norm)
        n_batches = len(train_loader) if max_batches is None else min(max_batches, len(train_loader))
        steps_per_epoch = max(1, n_batches // accum)            # optimiser updates per epoch

        # Two parameter groups: gentle updates for pretrained layers, faster learning for the head
        # (scratch models use the same LR for both)
        optimizer = make_optimizer(cfg, backbone, head, ph.lr_backbone, ph.lr_head, depth_groups)
        scheduler = warmup_cosine(optimizer, ph.epochs * steps_per_epoch,
                                  ph.warmup_epochs * steps_per_epoch, cfg.lr_schedule)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)  # avoids float16 gradient underflow
        bad_epochs = 0
        if p_i == 1:
            logger.info(f"input normalisation: mean {norm[0]}, std {norm[1]} | optimizer {cfg.optimizer}, "
                        f"LR schedule {cfg.lr_schedule}, augmentation {cfg.augment}")
            if depth_groups is not None:
                logger.info("layer-wise LR decay %.2f: backbone LR multipliers (input -> output) %s" % (
                    cfg.layer_decay, [round(cfg.layer_decay ** (len(depth_groups) - 1 - d), 3)
                                      for d in range(len(depth_groups))]))
            n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"trainable parameters: {n_train / 1e6:.3f} M"
                        + (" (backbone frozen: linear probe)" if cfg.freeze_backbone else ""))
        logger.info(f"phase {p_i}/{len(cfg.phases)}: {ph.img_size}px, {ph.epochs} epochs, "
                    f"batch {ph.batch_size} (= {accum} x {micro_bs}), {steps_per_epoch} updates/epoch, "
                    f"lr backbone {ph.lr_backbone:g} / head {ph.lr_head:g}")

        for epoch in range(1, ph.epochs + 1):
            model.train()                         # enables dropout and stochastic depth
            if cfg.freeze_backbone:               # frozen backbone: keep its BN running statistics fixed
                for m in model.modules():
                    if isinstance(m, nn.modules.batchnorm._BatchNorm):
                        m.eval()
            run_loss, run_correct, seen = 0.0, 0.0, 0
            optimizer.zero_grad(set_to_none=True)
            for i, (imgs, labels) in enumerate(train_loader):
                if i >= steps_per_epoch * accum:  # skip an incomplete accumulation group
                    break
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                use_cm = np.random.rand() < cfg.cutmix_prob
                if use_cm:
                    imgs, y_a, y_b, lam = cutmix(imgs, labels)

                with torch.autocast(device_type=device, enabled=use_amp):
                    logits = model(imgs)
                    if use_cm:   # loss weighted by how much of each image is visible
                        loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
                    else:
                        loss = criterion(logits, labels)
                # backpropagation; dividing by accum turns the summed gradients into an average
                scaler.scale(loss / accum).backward()
                if (i + 1) % accum == 0:
                    scaler.step(optimizer)        # weight update
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()              # LR schedule advances once per update

                preds = logits.argmax(1)
                if use_cm:   # "soft" accuracy on mixed images (only used for the training curves)
                    correct = lam * (preds == y_a).sum().item() + (1 - lam) * (preds == y_b).sum().item()
                else:
                    correct = (preds == labels).sum().item()
                run_loss += loss.item() * labels.size(0)
                run_correct += correct
                seen += labels.size(0)

            if full_train:      # validation images are part of the training data now: no validation
                val_loss = val_acc = float("nan")
            else:
                val_loss, val_acc = evaluate(model, val_loader, device, max_batches=max_batches)
            hist["train_loss"].append(run_loss / seen)
            hist["train_acc"].append(run_correct / seen)
            hist["val_loss"].append(val_loss)
            hist["val_acc"].append(val_acc)
            hist["img_size"].append(ph.img_size)
            # LR of the top backbone group (= lr_backbone also with layer-wise decay); head only if frozen
            pg = optimizer.param_groups
            hist["lr_backbone"].append(pg[-2 if len(pg) > 1 else 0]["lr"])

            improved = full_train or val_acc > best["val_acc"]   # full_train: always keep the latest epoch
            val_txt = "val n/a (full-train)" if full_train else f"val loss {val_loss:.3f} acc {val_acc:.4f}"
            logger.info(f"phase {p_i} ({ph.img_size}px) epoch {epoch:02d}/{ph.epochs} | "
                        f"train loss {hist['train_loss'][-1]:.3f} acc {hist['train_acc'][-1]:.3f} | "
                        f"{val_txt}" + ("  *best*" if improved and not full_train else ""))

            if improved:                          # save the new best checkpoint
                best = {"model": model_name, "config": cfg.name, "img_size": ph.img_size,
                        "mean": norm[0], "std": norm[1],
                        "epoch": len(hist["val_acc"]) - 1, "val_acc": val_acc, "full_train": full_train,
                        "state_dict": {k: v.detach().cpu().clone()
                                       for k, v in model.state_dict().items()}}
                torch.save(best, ckpt_path)
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= cfg.patience:
                    logger.info(f"early stopping in phase {p_i}")
                    break

        model.load_state_dict(best["state_dict"])  # next phase (or the caller) gets the best weights

    hist.update(model=model_name, config=cfg.name, best_epoch=best["epoch"],
                best_val_acc=best["val_acc"], best_img_size=best["img_size"],
                minutes=(time.time() - t0) / 60, full_train=full_train)
    if full_train:
        logger.info(f"full-train: kept the final epoch ({best['epoch'] + 1}) | {hist['minutes']:.1f} min")
    else:
        logger.info(f"best val acc {best['val_acc']:.4f} at epoch {best['epoch'] + 1} "
                    f"({best['img_size']}px) | {hist['minutes']:.1f} min")
    return hist
