"""A small 1D residual CNN for single-lead heartbeat classification.

Input  (B, 1, 187)
  stem:    Conv1d k7 -> BN -> ReLU                       (B, w, 187)
  stage 1: ResBlock(stride 2) + ResBlock                 (B, w, 94)
  stage 2: ResBlock(stride 2) + ResBlock                 (B, 2w, 47)
  stage 3: ResBlock(stride 2) + ResBlock                 (B, 4w, 24)
  head:    global average pool -> dropout -> Linear      (B, num_classes)
Output: raw logits (softmax is applied inside the loss / at inference time).
"""

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """conv-BN-ReLU-conv-BN + shortcut, then ReLU (the classic ResNet 'basic block')."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dropout: float = 0.0):
        super().__init__()
        # bias=False because BatchNorm right after has its own learnable shift.
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=5, stride=stride, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        if stride != 1 or in_ch != out_ch:
            # 1x1 projection so the shortcut matches the main path's shape.
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        return self.act(out + self.shortcut(x))


class ECGResNet(nn.Module):
    def __init__(self, num_classes: int = 5, width: int = 32, depth: int = 3, dropout: float = 0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, width, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(),
        )
        stages = []
        in_ch = width
        for i in range(depth):
            out_ch = width * 2**i
            stages += [
                ResidualBlock(in_ch, out_ch, stride=2, dropout=dropout),
                ResidualBlock(out_ch, out_ch, stride=1, dropout=dropout),
            ]
            in_ch = out_ch
        self.stages = nn.Sequential(*stages)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # makes the model independent of input length
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_ch, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        # Zero-init the last BN in each residual branch: every block starts as
        # (almost) the identity, which makes early training more stable.
        for m in self.modules():
            if isinstance(m, ResidualBlock):
                nn.init.zeros_(m.bn2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.stages(self.stem(x)))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
