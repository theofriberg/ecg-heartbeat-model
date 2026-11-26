"""Fast sanity tests that need no dataset download (synthetic data only)."""

import numpy as np
import pytest
import torch
from sklearn.metrics import f1_score
from torch import nn

from heartbeat.compare import confusion, macro_f1_from_confusion, paired_bootstrap
from heartbeat.config import Config
from heartbeat.data import NUM_CLASSES, SIGNAL_LENGTH, ECGDataset
from heartbeat.engine import make_ema
from heartbeat.model import ECGResNet


def synthetic(n: int = 64, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.random((n, SIGNAL_LENGTH), dtype=np.float32)
    x[:, 150:] = 0.0  # mimic zero padding
    y = rng.integers(0, NUM_CLASSES, n)
    return x, y


def test_config_rejects_class_weighting_with_label_smoothing():
    with pytest.raises(ValueError, match="cannot both be enabled"):
        Config(class_weighting=True, label_smoothing=0.05).validate()
    # Either one on its own is fine.
    Config(class_weighting=True, label_smoothing=0.0).validate()
    Config(class_weighting=False, label_smoothing=0.05).validate()


def test_macro_f1_from_confusion_matches_sklearn():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, NUM_CLASSES, 500)
    y_pred = np.where(rng.random(500) < 0.7, y_true, rng.integers(0, NUM_CLASSES, 500))
    y_pred[y_pred == 3] = 0  # a class that is never predicted -> its F1 is 0
    ours = macro_f1_from_confusion(confusion(y_true, y_pred, NUM_CLASSES))
    ref = f1_score(y_true, y_pred, average="macro", labels=range(NUM_CLASSES))
    assert ours == pytest.approx(ref)


def test_paired_bootstrap():
    rng = np.random.default_rng(0)
    y = rng.integers(0, NUM_CLASSES, 1000)
    noisy = np.where(rng.random(1000) < 0.6, y, rng.integers(0, NUM_CLASSES, 1000))

    # A model compared with itself: every resample gives exactly 0 difference.
    same = paired_bootstrap(y, noisy, noisy.copy(), n_boot=200)
    assert same.n_disagree == 0
    assert np.all(same.diffs == 0)

    # A perfect model vs a 60%-accurate one: the whole interval is above 0.
    r = paired_bootstrap(y, y.copy(), noisy, n_boot=200)
    assert r.score_a == pytest.approx(1.0)
    assert r.interval(r.diffs)[0] > 0


def test_config_rejects_bad_ema_decay():
    for bad in (1.0, -0.1):
        with pytest.raises(ValueError, match="ema_decay"):
            Config(ema_decay=bad).validate()
    Config(ema_decay=0.0).validate()  # 0 disables EMA


def test_ema_trails_live_weights_and_loads_as_plain_model():
    torch.manual_seed(0)
    x, y = synthetic(32)
    xb, yb = torch.from_numpy(x).unsqueeze(1), torch.from_numpy(y)
    model = ECGResNet(num_classes=NUM_CLASSES)
    ema = make_ema(model, decay=0.9)
    ema.update_parameters(model)  # first call copies the weights
    initial = [p.clone() for p in ema.module.parameters()]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for _ in range(5):
        opt.zero_grad()
        nn.functional.cross_entropy(model(xb), yb).backward()
        opt.step()
        ema.update_parameters(model)

    live, averaged = model.state_dict(), ema.module.state_dict()
    # It moved away from the initial weights, but not all the way to the live ones.
    assert any(not torch.equal(a, b) for a, b in zip(initial, ema.module.parameters(), strict=True))
    assert not torch.equal(live["stem.0.weight"], averaged["stem.0.weight"])
    # BatchNorm running stats are averaged too, not just copied from the live model.
    assert not torch.equal(live["stem.1.running_mean"], averaged["stem.1.running_mean"])
    # Its state_dict loads into a plain ECGResNet, which is what evaluate.py relies on.
    ECGResNet(num_classes=NUM_CLASSES).load_state_dict(averaged)


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
