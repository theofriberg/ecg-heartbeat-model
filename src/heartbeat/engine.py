"""The inner loops: one training epoch, and inference over a loader.

Kept free of I/O and config so they're easy to test and reuse.
"""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    grad_clip: float = 0.0,
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
