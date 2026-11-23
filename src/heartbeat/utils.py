"""Small cross-cutting helpers: reproducibility, device selection, logging."""

import logging
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed every RNG we use so runs are repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds CUDA and MPS


def get_device() -> torch.device:
    """Prefer CUDA (NVIDIA), then MPS (Apple Silicon GPU), then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
