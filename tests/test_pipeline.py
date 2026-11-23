"""Fast sanity tests that need no dataset download (synthetic data only)."""

import numpy as np
import torch
from torch import nn

from heartbeat.data import NUM_CLASSES, SIGNAL_LENGTH, ECGDataset
from heartbeat.model import ECGResNet


def synthetic(n: int = 64, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.random((n, SIGNAL_LENGTH), dtype=np.float32)
    x[:, 150:] = 0.0  # mimic zero padding
    y = rng.integers(0, NUM_CLASSES, n)
    return x, y


def test_model_output_shape():
    model = ECGResNet(num_classes=NUM_CLASSES).eval()
    out = model(torch.randn(8, 1, SIGNAL_LENGTH))
    assert out.shape == (8, NUM_CLASSES)


def test_augmentation_keeps_padding_zero():
    x, y = synthetic()
    ds = ECGDataset(x, y, augment=True)
    sample, _ = ds[0]
    assert sample.shape == (1, SIGNAL_LENGTH)
    assert torch.all(sample[..., 150:] == 0)


def test_can_overfit_single_batch():
    """Classic debugging check: if a model can't memorise 32 examples, there's
    a bug in the model, loss or optimisation code."""
    torch.manual_seed(0)
    x, y = synthetic(32)
    xb, yb = torch.from_numpy(x).unsqueeze(1), torch.from_numpy(y)
    model = ECGResNet(num_classes=NUM_CLASSES, dropout=0.0)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(150):
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
    model.eval()
    acc = (model(xb).argmax(1) == yb).float().mean().item()
    assert acc > 0.95
