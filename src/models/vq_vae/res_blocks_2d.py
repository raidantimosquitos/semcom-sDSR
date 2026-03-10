"""
2-D residual blocks for spectrogram encoders/decoders.

- ReZero: residual block with learnable scale (reference VQ-VAE-2 style).
- ResidualStack: stack of ReZero blocks.
- Residual / ResidualStack (legacy): original residual blocks, kept for compatibility.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2


class ReZero(nn.Module):
    """
    ReZero residual block: x + alpha * block(x).
    Matches reference VQ-VAE-2 implementation.
    """

    def __init__(self, in_channels: int, res_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, res_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(res_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(res_channels, in_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x) * self.alpha + x


class ResidualStack(nn.Module):
    """
    Stack of ReZero residual blocks (reference VQ-VAE-2 style).
    """

    def __init__(
        self,
        in_channels: int,
        res_channels: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.stack = nn.Sequential(
            *[ReZero(in_channels, res_channels) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stack(x)


# ---------------------------------------------------------------------------
# Legacy residual blocks (kept for any external use)
# ---------------------------------------------------------------------------


class Residual(nn.Module):
    """Single residual block: x + block(x). Middle channels = hidden_channels // 2."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
    ) -> None:
        super().__init__()
        mid = hidden_channels // 2
        self._block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, mid, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(mid, hidden_channels, kernel_size=1, stride=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self._block(x)


class ResidualStackLegacy(nn.Module):
    """Stack of legacy residual blocks with final ReLU."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_residual_layers: int,
    ) -> None:
        super().__init__()
        self._layers = nn.ModuleList([
            Residual(in_channels, hidden_channels)
            for _ in range(num_residual_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self._layers:
            x = layer(x)
        return F.relu(x)
