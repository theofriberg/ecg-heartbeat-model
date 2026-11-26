"""Compare two checkpoints with a paired bootstrap on the same split.

A single macro-F1 score depends partly on which beats happen to be in the
evaluation set. The bootstrap estimates how much: it resamples the set's beats
with replacement many times and rescores both models on each resample. Because
both models are scored on the *same* resample ("paired"), noise they share
cancels out, and what remains is the uncertainty in their difference.

Only covers evaluation-set noise. Training noise (seed, batch order, init) is
not included, so compare over several seeds as well.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from heartbeat.data import NUM_CLASSES
from heartbeat.evaluate import predict_checkpoint

log = logging.getLogger(__name__)


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """Macro-F1 from a (C, C) confusion matrix (rows = true, cols = predicted).

    Per class F1 = 2TP / (2TP + FP + FN), and 0 when the class never occurs in
    either labels or predictions (same as sklearn with zero_division=0).
    """
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(f1.mean())


def confusion(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    # Encode each (true, pred) pair as one integer and count: much faster than
    # sklearn's confusion_matrix, which matters when it runs thousands of times.
    counts = np.bincount(y_true * num_classes + y_pred, minlength=num_classes**2)
    return counts.reshape(num_classes, num_classes)


@dataclass
class BootstrapResult:
    score_a: float  # on the full set
    score_b: float
    diffs: np.ndarray  # score_a - score_b on each resample
    scores_a: np.ndarray  # score_a on each resample
    n_disagree: int  # samples where the two models predict different classes

    @property
    def diff(self) -> float:
        return self.score_a - self.score_b

    def interval(self, values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
        """Percentile interval: the middle `level` share of the bootstrap values."""
        tail = (1 - level) / 2 * 100
        lo, hi = np.percentile(values, [tail, 100 - tail])
        return float(lo), float(hi)


def paired_bootstrap(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
    num_classes: int = NUM_CLASSES,
) -> BootstrapResult:
    if not len(y_true) == len(pred_a) == len(pred_b):
        raise ValueError("y_true, pred_a and pred_b must have the same length")
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs, scores_a = np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)  # n draws with replacement: some beats twice, some never
        yt = y_true[idx]
        fa = macro_f1_from_confusion(confusion(yt, pred_a[idx], num_classes))
        fb = macro_f1_from_confusion(confusion(yt, pred_b[idx], num_classes))
        diffs[i], scores_a[i] = fa - fb, fa
    return BootstrapResult(
        score_a=macro_f1_from_confusion(confusion(y_true, pred_a, num_classes)),
        score_b=macro_f1_from_confusion(confusion(y_true, pred_b, num_classes)),
        diffs=diffs,
        scores_a=scores_a,
        n_disagree=int((pred_a != pred_b).sum()),
    )


def compare_checkpoints(
    checkpoint_a: Path,
    checkpoint_b: Path,
    split: str = "val",
    n_boot: int = 2000,
    seed: int = 0,
    device_name: str = "auto",
) -> BootstrapResult:
    y_a, pred_a, cfg_a, _ = predict_checkpoint(checkpoint_a, split, device_name)
    y_b, pred_b, cfg_b, _ = predict_checkpoint(checkpoint_b, split, device_name)

    # The validation split depends on the training seed, so runs with different
    # seeds were validated on different beats and can't be compared beat by beat.
    if split == "val" and (cfg_a.seed, cfg_a.val_fraction) != (cfg_b.seed, cfg_b.val_fraction):
        raise ValueError(
            f"The checkpoints use different validation splits (seed {cfg_a.seed} vs "
            f"{cfg_b.seed}, val_fraction {cfg_a.val_fraction} vs {cfg_b.val_fraction}). "
            "A paired comparison needs the same beats: compare runs with the same seed."
        )
    if not np.array_equal(y_a, y_b):
        raise ValueError("The two checkpoints were evaluated on different labels.")

    r = paired_bootstrap(y_a, pred_a, pred_b, n_boot=n_boot, seed=seed)
    lo_a, hi_a = r.interval(r.scores_a)
    lo_d, hi_d = r.interval(r.diffs)
    log.info(
        "%s macro-F1, %d beats, %d bootstrap resamples\n"
        "  A  %.4f   %s\n"
        "  B  %.4f   %s\n"
        "  A - B: %+.4f   95%% interval [%+.4f, %+.4f]   sd %.4f\n"
        "  A better in %.0f%% of resamples\n"
        "  A alone: sd %.4f, 95%% interval [%.4f, %.4f]\n"
        "  The models disagree on %d beats.",
        split,
        len(y_a),
        n_boot,
        r.score_a,
        checkpoint_a,
        r.score_b,
        checkpoint_b,
        r.diff,
        lo_d,
        hi_d,
        r.diffs.std(),
        (r.diffs > 0).mean() * 100,
        r.scores_a.std(),
        lo_a,
        hi_a,
        r.n_disagree,
    )
    if lo_d <= 0 <= hi_d:
        log.info("The interval includes 0: no clear difference on this split.")
    return r
