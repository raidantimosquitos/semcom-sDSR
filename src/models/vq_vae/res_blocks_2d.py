"""
2-D residual blocks for spectrogram encoders/decoders.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual(nn.Module):
    """Single residual block: x + block(x)."""

    def __init__(
        self,
        in_channels: int,
        num_hiddens: int,
        num_residual_hiddens: int,
    ) -> None:
        super().__init__()
        self._block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, num_residual_hiddens, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(num_residual_hiddens, num_hiddens, kernel_size=1, stride=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self._block(x)


class ResidualStack(nn.Module):
    """Stack of residual blocks with final ReLU."""

    def __init__(
        self,
        in_channels: int,
        num_hiddens: int,
        num_residual_layers: int,
        num_residual_hiddens: int,
    ) -> None:
        super().__init__()
        self._layers = nn.ModuleList([
            Residual(in_channels, num_hiddens, num_residual_hiddens)
            for _ in range(num_residual_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self._layers:
            x = layer(x)
        return F.relu(x)
