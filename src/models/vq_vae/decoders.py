"""
2-D CNN decoders for spectrogram reconstruction.

- DecoderCoarse: low-res quantized (8x20) -> high-res (32x80), 4x symmetric upsample, out 4*hidden_channels
- DecoderFine: concatenated [Q_coarse, Q_fine] (32x80) -> spectrogram (128x320), 4x symmetric upsample
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .res_blocks_2d import ResidualStack


class DecoderCoarse(nn.Module):
    """
    Coarse decoder: low-res quantized latent (8x20) -> high-res feature (32x80).
    Two 2x symmetric upsamples. Output 4*hidden_channels (to concat with f_fine).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_residual_layers: int,
    ) -> None:
        super().__init__()
        out_channels = hidden_channels * 4
        self._conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self._residual = ResidualStack(out_channels, out_channels, num_residual_layers)
        self._conv_trans1 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self._conv_trans2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv(x)
        x = self._residual(x)
        x = F.relu(self._conv_trans1(x))
        return self._conv_trans2(x)


class DecoderFine(nn.Module):
    """
    Fine decoder: concatenated [Q_coarse, Q_fine] (32x80) -> reconstructed spectrogram (128x320).
    Two symmetric 2x2 transposed convs: 32x80 -> 64x160 -> 128x320.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_residual_layers: int,
    ) -> None:
        super().__init__()
        self._conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self._residual = ResidualStack(hidden_channels, hidden_channels, num_residual_layers)
        self._conv_trans1 = nn.ConvTranspose2d(
            hidden_channels, hidden_channels // 2, kernel_size=4, stride=2, padding=1
        )
        self._conv_trans2 = nn.ConvTranspose2d(
            hidden_channels // 2, 1, kernel_size=4, stride=2, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv(x)
        x = self._residual(x)
        x = F.relu(self._conv_trans1(x))
        return self._conv_trans2(x)
