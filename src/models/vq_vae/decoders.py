"""
2-D CNN decoders for spectrogram reconstruction.

- DecoderTop: low-res quantized -> high-res (2x upsample)
- DecoderBot: concatenated [Q_top, Q_bot] -> reconstructed spectrogram (4x upsample)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .res_blocks_2d import ResidualStack


class DecoderTop(nn.Module):
    """
    Top decoder: low-res quantized latent -> high-res feature.
    2x spatial upsample.
    """

    def __init__(
        self,
        in_channels: int,
        num_hiddens: int,
        num_residual_layers: int,
        num_residual_hiddens: int,
    ) -> None:
        super().__init__()
        self._conv = nn.Conv2d(in_channels, num_hiddens, kernel_size=3, stride=1, padding=1)
        self._residual = ResidualStack(
            num_hiddens, num_hiddens,
            num_residual_layers, num_residual_hiddens,
        )
        self._conv_trans = nn.ConvTranspose2d(num_hiddens, num_hiddens, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv(x)
        x = self._residual(x)
        return self._conv_trans(x)


class DecoderBot(nn.Module):
    """
    Bottom decoder: concatenated [Q_top, Q_bot] -> reconstructed spectrogram.
    4x spatial upsample to original resolution.
    """

    def __init__(
        self,
        in_channels: int,
        num_hiddens: int,
        num_residual_layers: int,
        num_residual_hiddens: int,
    ) -> None:
        super().__init__()
        self._conv = nn.Conv2d(in_channels, num_hiddens, kernel_size=3, stride=1, padding=1)
        self._residual = ResidualStack(
            num_hiddens, num_hiddens,
            num_residual_layers, num_residual_hiddens,
        )
        self._conv_trans1 = nn.ConvTranspose2d(
            num_hiddens, num_hiddens // 2, kernel_size=4, stride=2, padding=1
        )
        self._conv_trans2 = nn.ConvTranspose2d(
            num_hiddens // 2, 1, kernel_size=4, stride=2, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv(x)
        x = self._residual(x)
        x = F.relu(self._conv_trans1(x))
        return self._conv_trans2(x)
