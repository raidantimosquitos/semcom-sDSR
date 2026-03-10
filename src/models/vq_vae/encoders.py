"""
2-D CNN encoders for spectrogram input.

Generic Encoder (reference VQ-VAE-2 style): configurable downscale_factor,
BatchNorm, ReLU, and ReZero ResidualStack. Used for both fine and coarse levels.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2

from .res_blocks_2d import ResidualStack


class Encoder(nn.Module):
    """
    Generic encoder: input -> latent with configurable downscale.
    Downscale must be a power of 2. Uses BatchNorm + ReLU and ReZero residual stack.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        res_channels: int,
        num_res_layers: int,
        downscale_factor: int,
    ) -> None:
        super().__init__()
        assert log2(downscale_factor) % 1 == 0, "Downscale must be a power of 2"
        downscale_steps = int(log2(downscale_factor))
        layers = []
        c_channel, n_channel = in_channels, hidden_channels // 2
        for _ in range(downscale_steps):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(c_channel, n_channel, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                )
            )
            c_channel, n_channel = n_channel, hidden_channels
        layers.append(nn.Conv2d(c_channel, n_channel, 3, stride=1, padding=1))
        layers.append(ResidualStack(n_channel, res_channels, num_res_layers))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# Backward-compatible aliases: build fine and coarse as Encoder with specific args
def EncoderFine(
    in_channels: int,
    hidden_channels: int,
    num_residual_layers: int,
    res_channels: int = 32,
    downscale_factor: int = 4,
) -> Encoder:
    """Fine (first-stage) encoder: input -> high-res latent (e.g. 4x down)."""
    return Encoder(
        in_channels, hidden_channels, res_channels, num_residual_layers, downscale_factor
    )


def EncoderCoarse(
    in_channels: int,
    hidden_channels: int,
    num_residual_layers: int,
    res_channels: int = 32,
    downscale_factor: int = 2,
) -> Encoder:
    """Coarse (second-stage) encoder: high-res latent -> low-res latent (e.g. 2x down)."""
    return Encoder(
        in_channels, hidden_channels, res_channels, num_residual_layers, downscale_factor
    )
