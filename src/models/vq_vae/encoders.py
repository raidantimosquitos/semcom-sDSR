"""
2-D CNN encoders for spectrogram input.

- EncoderBot: first stage (input -> high-res latent), 2x freq + 4x time downsampling
- EncoderTop: second stage (high-res -> low-res latent), 2x further in both dims
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .res_blocks_2d import ResidualStack


class EncoderBot(nn.Module):
    """
    Bottom encoder: input spectrogram -> high-resolution latent.
    2x down in frequency, 4x down in time (conv1 stride 2x2, conv2 stride 1x2).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_residual_layers: int,
        num_residual_hiddens: int,
    ) -> None:
        super().__init__()
        self._conv1 = nn.Conv2d(in_channels, hidden_channels // 2, kernel_size=4, stride=2, padding=1)
        self._conv2 = nn.Conv2d(hidden_channels // 2, hidden_channels, kernel_size=(3, 4), stride=(1, 2), padding=(1, 1))
        self._conv3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self._residual = ResidualStack(
            hidden_channels, hidden_channels,
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
    2x down in frequency, 2x down in time.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_residual_layers: int,
        num_residual_hiddens: int,
    ) -> None:
        super().__init__()
        self._conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=4, stride=2, padding=1)
        self._conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self._residual = ResidualStack(
            hidden_channels, hidden_channels,
            num_residual_layers, num_residual_hiddens,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self._conv1(x))
        x = F.relu(self._conv2(x))
        return self._residual(x)
