"""Data loading for the MIT-BIH arrhythmia heartbeat dataset.

Kaggle dataset ``shayanfazeli/heartbeat`` ships pre-segmented heartbeats:
  * each row = one beat, 187 samples at 125 Hz, already min-max scaled to [0, 1]
  * shorter beats are zero-padded at the end
  * last column = label in {0..4}  (AAMI classes N, S, V, F, Q)

Splits:
  train (official) -> stratified train / validation
  test  (official) -> held out, only touched by `heartbeat evaluate`
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

from heartbeat.config import Config

log = logging.getLogger(__name__)

SIGNAL_LENGTH = 187
CLASS_NAMES = (
    "N (normal)",
    "S (supraventricular ectopic)",
    "V (ventricular ectopic)",
    "F (fusion)",
    "Q (unknown)",
)
NUM_CLASSES = len(CLASS_NAMES)


def download_dataset(handle: str) -> Path:
    """Download (or reuse the cached copy) and return the local directory."""
    path = Path(kagglehub.dataset_download(handle))
    log.info("Dataset available at %s", path)
    return path


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path, header=None)
    x = df.iloc[:, :-1].to_numpy(dtype=np.float32)
    y = df.iloc[:, -1].to_numpy(dtype=np.int64)
    assert x.shape[1] == SIGNAL_LENGTH, f"unexpected signal length {x.shape[1]}"
    return x, y


@dataclass
class Splits:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def load_splits(cfg: Config) -> Splits:
    root = download_dataset(cfg.dataset_handle)
    x, y = load_csv(root / "mitbih_train.csv")
    x_test, y_test = load_csv(root / "mitbih_test.csv")

    # Stratify so the rare classes (F has ~640 examples) appear in validation
    # in the same proportion as in training. sklearn's return type is loosely
    # annotated (list | array), so np.asarray narrows it; it doesn't copy.
    x_train, x_val, y_train, y_val = (
        np.asarray(a)
        for a in train_test_split(
            x, y, test_size=cfg.val_fraction, stratify=y, random_state=cfg.seed
        )
    )
    log.info("train=%d  val=%d  test=%d", len(y_train), len(y_val), len(y_test))
    log.info("train class counts: %s", np.bincount(y_train, minlength=NUM_CLASSES).tolist())
    return Splits(x_train, y_train, x_val, y_val, x_test, y_test)


def class_weights(y: np.ndarray) -> torch.Tensor:
    """'Balanced' weights: w_c = n_samples / (n_classes * n_c)."""
    w = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=y)
    return torch.tensor(w, dtype=torch.float32)


class ECGDataset(Dataset):
    """Wraps (N, 187) arrays as (1, 187) float tensors for a Conv1d model.

    With ``augment=True`` it applies light, label-preserving perturbations:
    random amplitude scaling and small Gaussian noise on the real (non-padded)
    part of the beat. Augmentation is only ever enabled for the training set.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, augment: bool = False):
        self.x = torch.from_numpy(x).unsqueeze(1)  # (N, C=1, L)
        self.y = torch.from_numpy(y)
        self.augment = augment
        # Length of each beat before the zero padding starts.
        nonzero = x != 0
        self.lengths = torch.from_numpy(x.shape[1] - np.argmax(nonzero[:, ::-1], axis=1))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.x[idx]
        if self.augment:
            scale = torch.empty(1).uniform_(0.9, 1.1)
            noise = torch.randn_like(x) * 0.01
            noise[..., self.lengths[idx] :] = 0.0  # keep the padding exactly zero
            x = x * scale + noise
        return x, self.y[idx]


def make_loaders(cfg: Config, splits: Splits) -> tuple[DataLoader, DataLoader]:
    train_ds = ECGDataset(splits.x_train, splits.y_train, augment=cfg.augment)
    val_ds = ECGDataset(splits.x_val, splits.y_val)
    g = torch.Generator().manual_seed(cfg.seed)  # reproducible shuffling
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,  # avoids a tiny last batch upsetting BatchNorm
        num_workers=cfg.num_workers,
        generator=g,
    )
    val_loader = make_eval_loader(cfg, val_ds)
    return train_loader, val_loader


def make_eval_loader(cfg: Config, ds: Dataset) -> DataLoader:
    return DataLoader(ds, batch_size=cfg.batch_size * 4, shuffle=False, num_workers=cfg.num_workers)
