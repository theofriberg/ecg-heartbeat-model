"""Training orchestration: wires data, model, optimiser and the loop together."""

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from heartbeat.config import Config
from heartbeat.data import NUM_CLASSES, class_weights, load_splits, make_loaders
from heartbeat.engine import compute_metrics, predict, train_one_epoch
from heartbeat.model import ECGResNet, count_parameters
from heartbeat.utils import get_device, seed_everything

log = logging.getLogger(__name__)


def build_model(cfg: Config) -> ECGResNet:
    return ECGResNet(num_classes=NUM_CLASSES, width=cfg.width, depth=cfg.depth, dropout=cfg.dropout)


def param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Apply weight decay to conv/linear weights only, not to biases or BatchNorm
    parameters (decaying those just fights normalisation and hurts accuracy)."""
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim > 1 else no_decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def train(cfg: Config) -> Path:
    seed_everything(cfg.seed)
    device = get_device(cfg.device)

    run_dir = cfg.output_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    log.info("Run directory: %s | device: %s", run_dir, device)

    splits = load_splits(cfg)
    train_loader, val_loader = make_loaders(cfg, splits)

    model = build_model(cfg).to(device)
    log.info("Model has %s trainable parameters", f"{count_parameters(model):,}")

    weight = class_weights(splits.y_train).to(device) if cfg.class_weighting else None
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.AdamW(param_groups(model, cfg.weight_decay), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr, epochs=cfg.epochs, steps_per_epoch=len(train_loader)
    )

    best_f1, epochs_without_improvement = -1.0, 0
    history = []
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.perf_counter()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, cfg.grad_clip
        )
        logits, targets = predict(model, val_loader, device)
        val_loss = criterion(logits.to(device), targets.to(device)).item()
        metrics = compute_metrics(targets.numpy(), logits.argmax(1).numpy())

        row = {
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"val_{k}": v for k, v in metrics.items()},
        }
        history.append(row)
        log.info(
            "epoch %3d | train_loss %.4f | val_loss %.4f | val_f1 %.4f | val_acc %.4f | %.1fs",
            epoch,
            train_loss,
            val_loss,
            metrics["macro_f1"],
            metrics["accuracy"],
            time.perf_counter() - t0,
        )

        # Model selection + early stopping on validation macro-F1.
        if metrics["macro_f1"] > best_f1:
            best_f1, epochs_without_improvement = metrics["macro_f1"], 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg.to_dict(),
                    "epoch": epoch,
                    "val_macro_f1": best_f1,
                },
                run_dir / "best.pt",
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.patience:
                log.info("Early stopping: no improvement for %d epochs", cfg.patience)
                break

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    log.info("Best val macro-F1 %.4f. Checkpoint: %s", best_f1, run_dir / "best.pt")
    return run_dir
