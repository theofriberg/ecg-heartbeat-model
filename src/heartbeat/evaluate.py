"""Evaluate a saved checkpoint and write a report + confusion matrix.

Run this on the *test* split only once you're done making decisions based on
validation results; otherwise the test score stops being an honest estimate.
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
from matplotlib.figure import Figure
from sklearn.metrics import classification_report, confusion_matrix

from heartbeat.config import Config
from heartbeat.data import CLASS_NAMES, ECGDataset, load_splits, make_eval_loader
from heartbeat.engine import compute_metrics, predict
from heartbeat.train import build_model
from heartbeat.utils import get_device

log = logging.getLogger(__name__)


def evaluate_checkpoint(checkpoint: Path, split: str = "test") -> dict:
    device = get_device()
    # weights_only=True refuses to unpickle arbitrary objects: safe loading.
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = Config.from_dict(ckpt["config"])

    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    splits = load_splits(cfg)  # same seed -> identical train/val split as training
    x, y = (splits.x_test, splits.y_test) if split == "test" else (splits.x_val, splits.y_val)
    logits, targets = predict(model, make_eval_loader(cfg, ECGDataset(x, y)), device)
    y_true, y_pred = targets.numpy(), logits.argmax(1).numpy()

    metrics = compute_metrics(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    log.info("%s results (checkpoint from epoch %d):\n%s", split, ckpt["epoch"], report)

    out_dir = checkpoint.parent
    (out_dir / f"{split}_metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / f"{split}_report.txt").write_text(report)
    plot_confusion_matrix(y_true, y_pred, out_dir / f"{split}_confusion_matrix.png")
    log.info("Wrote metrics, report and confusion matrix to %s", out_dir)
    return metrics


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    """Row-normalised: each row shows how beats of a true class were classified."""
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    labels = [name.split()[0] for name in CLASS_NAMES]
    fig = Figure(figsize=(6, 5), layout="constrained")  # no pyplot -> no GUI backend needed
    ax = fig.subplots()
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > 0.5 else "black",
            )
    ax.set(
        xticks=range(len(labels)),
        yticks=range(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted",
        ylabel="True",
        title="Confusion matrix (recall per row)",
    )
    fig.colorbar(im, ax=ax)
    fig.savefig(path, dpi=150)
