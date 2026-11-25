"""Experiment configuration.

All hyperparameters live in a single dataclass. This gives you:
  * one place to look up every knob,
  * a CLI generated automatically from the fields (see cli.py),
  * a JSON copy saved next to every run, so results are always reproducible.
"""

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass
class Config:
    # --- data ---
    dataset_handle: str = "shayanfazeli/heartbeat"
    val_fraction: float = 0.1  # carved out of the official train split, stratified
    augment: bool = True

    # --- model ---
    width: int = 32  # channels in the first stage; doubled at every stage
    depth: int = 3  # number of residual stages (each halves the time resolution)
    dropout: float = 0.2

    # --- optimisation ---
    epochs: int = 30
    batch_size: int = 256
    lr: float = 3e-3  # peak LR of the one-cycle schedule
    weight_decay: float = 1e-2
    label_smoothing: float = 0.05  # must be 0 when class_weighting is on, see validate()
    class_weighting: bool = False  # balanced class weights in the loss
    grad_clip: float = 1.0  # 0 disables
    patience: int = 7  # early stopping on validation macro-F1

    # --- misc ---
    seed: int = 42
    device: str = "auto"  # auto | cpu | mps | cuda
    num_workers: int = 0  # data is in memory; extra workers only add overhead here
    output_dir: Path = Path("runs")

    def validate(self) -> None:
        """Reject setting combinations that train without errors but give a broken model.

        Called by train() rather than __post_init__, so checkpoints saved with
        an older, invalid config can still be loaded and evaluated.
        """
        if self.class_weighting and self.label_smoothing > 0:
            raise ValueError(
                "class_weighting and label_smoothing cannot both be enabled. "
                "PyTorch's CrossEntropyLoss multiplies the smoothing term by the "
                "class weights too, which pushes every sample towards the rarest "
                "classes (normal beats get predicted as F/S). Use "
                "--no-class-weighting or --label-smoothing 0."
            )

    def to_dict(self) -> dict[str, Any]:
        """JSON/torch-safe dict (Paths become strings)."""
        return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        known = {f.name for f in fields(cls)}
        kwargs: dict[str, Any] = {k: v for k, v in d.items() if k in known}
        if "output_dir" in kwargs:
            kwargs["output_dir"] = Path(kwargs["output_dir"])
        return cls(**kwargs)
