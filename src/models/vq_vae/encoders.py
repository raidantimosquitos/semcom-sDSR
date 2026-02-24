"""
2-D CNN encoders for spectrogram input.

- EncoderBot: first stage (input -> high-res latent), 4x spatial downsampling
- EncoderTop: second stage (high-res -> low-res latent), 2x further downsampling
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .res_blocks_2d import ResidualStack


class EncoderBot(nn.Module):
    """
    Bottom encoder: input spectrogram -> high-resolution latent.
    4x downsampling in spatial dims (2 strided convs).
    """

    def __init__(
        self,
        in_channels: int,
        num_hiddens: int,
        num_residual_layers: int,
        num_residual_hiddens: int,
    ) -> None:
        super().__init__()
        self._conv1 = nn.Conv2d(in_channels, num_hiddens // 2, kernel_size=4, stride=2, padding=1)
        self._conv2 = nn.Conv2d(num_hiddens // 2, num_hiddens, kernel_size=4, stride=2, padding=1)
        self._conv3 = nn.Conv2d(num_hiddens, num_hiddens, kernel_size=3, stride=1, padding=1)
        self._residual = ResidualStack(
            num_hiddens, num_hiddens,
            num_residual_layers, num_residual_hiddens,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self._conv1(x))
        x = F.relu(self._conv2(x))
        x = self._conv3(x)
        return self._residual(x)


class EncoderTop(nn.Module):
    """
    Top encoder: high-res latent -> low-res latent.
    2x additional downsampling.
    """

    def __init__(
        self,
        in_channels: int,
        num_hiddens: int,
        num_residual_layers: int,
        num_residual_hiddens: int,
    ) -> None:
        super().__init__()
        self._conv1 = nn.Conv2d(in_channels, num_hiddens, kernel_size=4, stride=2, padding=1)
        self._conv2 = nn.Conv2d(num_hiddens, num_hiddens, kernel_size=3, stride=1, padding=1)
        self._residual = ResidualStack(
            num_hiddens, num_hiddens,
            num_residual_layers, num_residual_hiddens,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self._conv1(x))
        x = F.relu(self._conv2(x))
        return self._residual(x)
