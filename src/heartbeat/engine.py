"""The inner loops: one training epoch, and inference over a loader.

Kept free of I/O and config so they're easy to test and reuse.
"""

from collections.abc import Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader


def make_ema(model: nn.Module, decay: float) -> AveragedModel:
    """EMA copy of ``model`` whose decay warms up as min(decay, (1 + t) / (10 + t)).

    Without warmup, a fixed decay of 0.999 keeps 0.999**t of the random initial
    weights: still ~54% after 2 epochs (~600 steps), so early EMA models are
    useless. With warmup the EMA starts by closely following the model, and its
    window (~1 / (1 - decay_t) steps) grows to roughly the last ~10% of training,
    capped at ~1 / (1 - decay) steps.

    use_buffers=True averages BatchNorm's running mean/var too, so they match
    the averaged weights.
    """

    @torch.no_grad()
    def ema_update(
        ema_params: Sequence[torch.Tensor],
        model_params: Sequence[torch.Tensor],
        num_averaged: torch.Tensor | int,
    ) -> None:
        # Called once per (device, dtype) group, so every tensor in a group shares a dtype.
        # A per-tensor loop (instead of torch's private _foreach ops) costs ~60 small
        # kernel launches per step for this model: negligible next to forward/backward.
        if not torch.is_floating_point(ema_params[0]):
            # Integer buffers (BatchNorm's num_batches_tracked) are counters, not weights.
            for p_ema, p_model in zip(ema_params, model_params, strict=True):
                p_ema.copy_(p_model)
            return
        t = int(num_averaged)
        weight = 1 - min(decay, (1 + t) / (10 + t))
        for p_ema, p_model in zip(ema_params, model_params, strict=True):
            p_ema.lerp_(p_model, weight)

    return AveragedModel(model, multi_avg_fn=ema_update, use_buffers=True)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    grad_clip: float = 0.0,
    ema: AveragedModel | None = None,
) -> float:
    model.train()  # enables dropout + BatchNorm batch statistics
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()  # one-cycle schedules step per batch, not per epoch
        if ema is not None:
            ema.update_parameters(model)  # nudge the averaged copy towards the new weights

        total_loss += loss.item() * len(y)
        n += len(y)
    return total_loss / n


@torch.inference_mode()  # no autograd bookkeeping: faster, less memory
def predict(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (logits, targets) for the whole loader, on CPU."""
    model.eval()  # disables dropout, BatchNorm uses running statistics
    all_logits, all_targets = [], []
    for x, y in loader:
        all_logits.append(model(x.to(device)).cpu())
        all_targets.append(y)
    return torch.cat(all_logits), torch.cat(all_targets)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Accuracy is misleading here (~83% of beats are 'N'), so we track macro
    metrics that weigh every class equally.

    sklearn returns NumPy scalars (and f1_score is typed as possibly returning
    an array, for average=None), so convert to plain Python floats: this keeps
    the declared return type honest and makes the values JSON-serialisable.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }
