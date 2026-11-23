"""Command-line entry point: `uv run heartbeat {download,train,evaluate}`.

Training flags are generated from the Config dataclass, so adding a field
there automatically adds a `--flag` here.
"""

import argparse
from dataclasses import fields
from pathlib import Path

from heartbeat.config import Config
from heartbeat.utils import setup_logging


def add_config_args(parser: argparse.ArgumentParser) -> None:
    defaults = Config()
    for f in fields(Config):
        default = getattr(defaults, f.name)
        flag = "--" + f.name.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=default)
        else:
            parser.add_argument(
                flag, type=type(default), default=default, help=f"(default: {default})"
            )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="heartbeat", description="ECG heartbeat classification")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("download", help="download the Kaggle dataset into the kagglehub cache")

    p_train = sub.add_parser("train", help="train a model")
    add_config_args(p_train)

    p_eval = sub.add_parser("evaluate", help="evaluate a checkpoint")
    p_eval.add_argument("checkpoint", type=Path, help="path to best.pt")
    p_eval.add_argument("--split", choices=["val", "test"], default="test")

    args = parser.parse_args(argv)
    setup_logging()

    # Imports are deferred so `--help` stays fast (torch is slow to import).
    if args.command == "download":
        from heartbeat.data import download_dataset

        download_dataset(Config().dataset_handle)
    elif args.command == "train":
        from heartbeat.train import train

        cfg_kwargs = {f.name: getattr(args, f.name) for f in fields(Config)}
        train(Config(**cfg_kwargs))
    elif args.command == "evaluate":
        from heartbeat.evaluate import evaluate_checkpoint

        evaluate_checkpoint(args.checkpoint, args.split)
