# heartbeat

A PyTorch pipeline for training and evaluating a neural network that classifies ECG heartbeats into five arrhythmia classes. It uses the MIT-BIH Arrhythmia dataset.

The pipeline downloads the data, splits it into train, validation and test sets, trains a 1D residual CNN, and writes evaluation reports. It's built to be reproducible and easy to extend.

## Dataset

The project uses the Kaggle dataset [`shayanfazeli/heartbeat`](https://www.kaggle.com/datasets/shayanfazeli/heartbeat), which contains heartbeats already cut out of the [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/).

- Each row is one heartbeat: 187 samples at 125 Hz, scaled to [0, 1] and zero-padded at the end.
- The last column is the class label:

| Label | Class | Meaning | Share of training data |
|---|---|---|---|
| 0 | N | Normal beat | 83% |
| 1 | S | Supraventricular ectopic beat | 2.5% |
| 2 | V | Ventricular ectopic beat | 6.6% |
| 3 | F | Fusion beat | 0.7% |
| 4 | Q | Unknown / paced beat | 7.3% |

The classes are very imbalanced, and that shapes several design choices (see [Training](#training)).

The pipeline uses `mitbih_train.csv` (87,554 beats) and `mitbih_test.csv` (21,892 beats). The download also includes the PTB Diagnostic ECG files (`ptbdb_*.csv`), which the pipeline doesn't use yet.

## Setup

You need [uv](https://docs.astral.sh/uv/). It installs the right Python version (3.13) and all dependencies:

```bash
uv sync
```

### Kaggle access

The dataset is public, so **no Kaggle account is needed**. `kagglehub` downloads it on first use and caches it in `~/.cache/kagglehub/`.

You only need credentials for private data or competitions. To set them up, create a token at kaggle.com under **Settings → API → Create New Token**, then do one of the following:

- save the token to `~/.kaggle/access_token` and run `chmod 600 ~/.kaggle/access_token`, or
- export it as an environment variable: `export KAGGLE_API_TOKEN=...`

## Usage

```bash
uv run heartbeat download                         # optional: fetch the dataset in advance
uv run heartbeat train                            # train with the default settings
uv run heartbeat train --epochs 50 --lr 1e-3      # change any setting with a flag
uv run heartbeat train --help                     # list all settings

uv run heartbeat evaluate runs/<timestamp>/best.pt --split val    # while you're developing
uv run heartbeat evaluate runs/<timestamp>/best.pt --split test   # once, at the very end
```

Training uses an NVIDIA GPU (CUDA) or Apple Silicon GPU (MPS) if one is available, and falls back to the CPU otherwise.

### Outputs

Each training run creates its own folder:

```
runs/<timestamp>/
├── config.json               every setting used for the run
├── best.pt                   checkpoint with the best validation macro-F1
├── history.csv               loss and metrics for each epoch
├── <split>_metrics.json      accuracy, balanced accuracy, macro-F1   (from evaluate)
├── <split>_report.txt        precision / recall / F1 for each class  (from evaluate)
└── <split>_confusion_matrix.png
```

### Validation vs. test

The official training file is split 90/10 into **train** and **validation** sets. The split is *stratified*, so every class appears in both sets in the same proportion.

The validation set is used to pick the best checkpoint and to tune settings. The official **test** set is never used during training. Only run `evaluate --split test` once you've stopped making decisions. If you tune settings against the test score, it no longer tells you how the model performs on new data.

## Model

`ECGResNet` (in `model.py`) is a small 1D residual CNN with about 393k parameters:

```
input (B, 1, 187)
  stem:     Conv1d k=7 → BatchNorm → ReLU              (B,  32, 187)
  stage 1:  ResBlock (stride 2) → ResBlock             (B,  32,  94)
  stage 2:  ResBlock (stride 2) → ResBlock             (B,  64,  47)
  stage 3:  ResBlock (stride 2) → ResBlock             (B, 128,  24)
  head:     global average pool → dropout → Linear     (B, 5) logits
```

`B` is the batch size.

Each residual block computes `relu(x + F(x))`, where `F` is conv → BN → ReLU → conv → BN. The shortcut connection gives gradients a direct path back through the network, so deeper networks train reliably. Each block only has to learn a correction to its input.

Every stage halves the time resolution and doubles the number of channels. Global average pooling then combines features from the whole beat into one prediction.

You can change the size of the network with `--width` (channels in the first stage) and `--depth` (number of stages).

## Training

| Choice | Why |
|---|---|
| **AdamW**, with weight decay only on conv and linear weights | Decaying biases and BatchNorm parameters works against normalisation |
| **One-cycle learning-rate schedule**, updated every batch | Warm up, then anneal: fast, stable convergence |
| **Class-weighted cross entropy** | Counteracts the 83% majority class |
| **Label smoothing** (0.05) | Keeps the model from becoming overconfident |
| **Gradient clipping** | Prevents rare exploding updates |
| **Augmentation**: amplitude scaling ±10% plus slight noise | More variety in the training data. Padding stays zero, and only the training set is augmented |
| **Best checkpoint and early stopping chosen by validation macro-F1** | Accuracy is misleading on imbalanced data. Always predicting "N" scores 83% |
| **Seeded random numbers**, config saved with each run | Reproducible results |
| **Weights-only checkpoints**, loaded with `weights_only=True` | Loading a checkpoint can't run arbitrary code |

## Project structure

```
src/heartbeat/
├── config.py     Config dataclass: one place for every setting
├── data.py       download, load, stratified split, Dataset / DataLoader, augmentation
├── model.py      ECGResNet and ResidualBlock
├── engine.py     inner loops (train one epoch, predict) and metrics
├── train.py      wires everything together: optimiser, schedule, checkpoints, early stopping
├── evaluate.py   evaluates a checkpoint, writes report and confusion matrix
├── cli.py        `heartbeat` command; training flags are generated from Config
└── utils.py      seeding, device selection, logging
tests/
└── test_pipeline.py   quick checks on synthetic data (no download needed)
```

To add a new setting, add a field to `Config`. It automatically becomes a `--flag` on the command line and is saved in each run's `config.json`.

## Development

```bash
uv run pytest            # tests: output shape, augmentation, overfitting a single batch
uv run ruff check .      # lint
uv run ruff format .     # format
```

The "overfit a single batch" test is a standard debugging check. If the model can't memorise 32 examples, something is wrong in the model, loss or optimiser code.

## Known issues

- **PyTorch is pinned to `<2.15`.** PyTorch 2.15's Apple GPU (MPS) code needs macOS 15 or later and crashes on macOS 14. Version 2.14 works on MPS. Remove the pin in `pyproject.toml` once you're on macOS 15 or later.
