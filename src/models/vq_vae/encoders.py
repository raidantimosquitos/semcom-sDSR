"""
2-D CNN encoders for spectrogram input.

- EncoderFine: first stage (input -> high-res latent), 4x4 symmetric downsampling (32x80 for 128x320)
- EncoderCoarse: second stage (high-res -> low-res latent), 4x4 symmetric from f_fine (8x20), out 4*hidden_channels
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .res_blocks_2d import ResidualStack


class EncoderFine(nn.Module):
    """
    Fine (high-res) encoder: input spectrogram -> high-resolution latent.
    Symmetric 4x4 down (two 2x2 strides): 128x320 -> 32x80. Output hidden_channels.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_residual_layers: int,
        num_residual_hiddens: int,
    ) -> None:
        super().__init__()
        self._conv1 = nn.Conv2d(in_channels, hidden_channels // 4, kernel_size=4, stride=2, padding=1)
        self._conv2 = nn.Conv2d(hidden_channels // 4, hidden_channels // 2, kernel_size=4, stride=2, padding=1)
        self._conv3 = nn.Conv2d(hidden_channels // 2, hidden_channels, kernel_size=3, stride=1, padding=1)
        self._residual = ResidualStack(
            hidden_channels, hidden_channels,
            num_residual_layers, num_residual_hiddens,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self._conv1(x))
        x = F.relu(self._conv2(x))
        x = self._conv3(x)
        return self._residual(x)


class EncoderCoarse(nn.Module):
    """
    Coarse (low-res) encoder: high-res latent -> low-res latent.
    Symmetric 4x4 down from f_fine (32x80 -> 8x20). Output 4*hidden_channels.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_residual_layers: int,
        num_residual_hiddens: int,
    ) -> None:
        super().__init__()
        out_channels = hidden_channels * 4
        self._conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=4, stride=2, padding=1)
        self._conv2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=4, stride=2, padding=1)
        self._residual = ResidualStack(
            out_channels, out_channels,
            num_residual_layers, num_residual_hiddens,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self._conv1(x))
        x = F.relu(self._conv2(x))
        return self._residual(x)
