"""Compact hierarchical ECG network for binary detection and rhythm labelling.

The network has one shared multi-scale 1-D convolutional encoder and two heads:

* binary_head: normal versus abnormal for every supported binary annotation;
* class_head: N, PVC, PAC, LBBB, RBBB, or AFib when a project class exists.

This design lets unsupported abnormal MIT-BIH beat symbols contribute to the
binary task without forcing them into an incorrect rhythm class.
"""
from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn
import torch.nn.functional as F

from app.ml.classifier import ARRHYTHMIA_CLASSES


def _groups(channels: int) -> int:
    """Return the largest practical GroupNorm divisor up to eight."""
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class InceptionResidual1D(nn.Module):
    """Multi-scale temporal block with a residual connection.

    Parallel kernels inspect short, medium, and longer ECG morphology.  A
    pooling branch supplies a locally robust summary.  Their outputs are
    concatenated and added to a projected residual path.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        branch_channels = out_channels // 4
        if branch_channels * 4 != out_channels:
            raise ValueError("out_channels must be divisible by four")

        self.reduce = nn.Conv1d(in_channels, branch_channels, kernel_size=1)
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(
                    branch_channels,
                    branch_channels,
                    kernel_size=kernel,
                    stride=stride,
                    padding=kernel // 2,
                )
                for kernel in (5, 11, 23)
            ]
        )
        self.pool = nn.MaxPool1d(kernel_size=3, stride=stride, padding=1)
        self.pool_projection = nn.Conv1d(
            in_channels, branch_channels, kernel_size=1
        )
        self.norm = nn.GroupNorm(_groups(out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.shortcut = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reduced = F.gelu(self.reduce(x))
        paths = [branch(reduced) for branch in self.branches]
        paths.append(self.pool_projection(self.pool(x)))
        combined = torch.cat(paths, dim=1)
        return self.dropout(F.gelu(self.norm(combined) + self.shortcut(x)))


class HierarchicalECGNet(nn.Module):
    """Shared multi-scale encoder with binary and six-class output heads."""

    def __init__(
        self,
        num_classes: int = len(ARRHYTHMIA_CLASSES),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )
        self.encoder = nn.Sequential(
            InceptionResidual1D(32, 48, stride=1, dropout=0.08),
            InceptionResidual1D(48, 64, stride=2, dropout=0.10),
            InceptionResidual1D(64, 96, stride=2, dropout=0.12),
            InceptionResidual1D(96, 128, stride=2, dropout=0.15),
            InceptionResidual1D(128, 128, stride=1, dropout=0.15),
        )
        # Mean, maximum, and standard-deviation pooling retain complementary
        # information about average rhythm, strong morphology, and variability.
        self.shared = nn.Sequential(
            nn.Linear(128 * 3, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(192, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.binary_head = nn.Linear(128, 1)
        self.class_head = nn.Linear(128, num_classes)
        self.num_classes = num_classes
        self.dropout = dropout

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(self.stem(x))
        mean = features.mean(dim=-1)
        maximum = features.amax(dim=-1)
        standard_deviation = features.std(dim=-1, unbiased=False)
        return self.shared(torch.cat([mean, maximum, standard_deviation], dim=1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(x)
        return self.binary_head(features).squeeze(1), self.class_head(features)

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def architecture_summary(self) -> Dict[str, Any]:
        return {
            "name": "Hierarchical-Multitask-Inception-ECG",
            "input_shape": [1, 512],
            "shared_encoder": "five residual multi-scale 1-D convolution blocks",
            "kernel_sizes": [5, 11, 23],
            "pooling": ["mean", "maximum", "standard_deviation"],
            "binary_output": "normal versus abnormal logit",
            "class_output": list(ARRHYTHMIA_CLASSES),
            "dropout": self.dropout,
            "total_parameters": self.count_parameters(),
        }


def build_hierarchical_model(
    num_classes: int = len(ARRHYTHMIA_CLASSES),
    dropout: float = 0.2,
) -> HierarchicalECGNet:
    return HierarchicalECGNet(num_classes=num_classes, dropout=dropout)


def load_hierarchical_checkpoint(
    path: str,
    device: torch.device | str = "cpu",
) -> tuple[HierarchicalECGNet, Dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    model = build_hierarchical_model(
        num_classes=len(checkpoint.get("class_names", ARRHYTHMIA_CLASSES)),
        dropout=float(config.get("dropout", 0.2)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint
