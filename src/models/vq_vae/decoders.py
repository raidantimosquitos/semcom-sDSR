"""
2-D CNN decoders for spectrogram reconstruction.

Generic Decoder and Upscaler (reference VQ-VAE-2 style): configurable upscale_factor,
BatchNorm, ReLU, ReZero ResidualStack. Coarse decoder outputs embed_dim; fine decoder outputs image channels.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2

from .res_blocks_2d import ResidualStack


class Decoder(nn.Module):
    """
    Generic decoder: latent -> output with configurable upscale.
    Upscale must be a power of 2. Uses conv, ResidualStack, then ConvTranspose2d steps.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        res_channels: int,
        num_res_layers: int,
        upscale_factor: int,
    ) -> None:
        super().__init__()
        assert log2(upscale_factor) % 1 == 0, "Upscale must be a power of 2"
        upscale_steps = int(log2(upscale_factor))
        layers = [
            nn.Conv2d(in_channels, hidden_channels, 3, stride=1, padding=1),
            ResidualStack(hidden_channels, res_channels, num_res_layers),
        ]
        c_channel, n_channel = hidden_channels, hidden_channels // 2
        for _ in range(upscale_steps):
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(c_channel, n_channel, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                )
            )
            c_channel, n_channel = n_channel, out_channels
        layers.append(nn.Conv2d(c_channel, n_channel, 3, stride=1, padding=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Upscaler(nn.Module):
    """
    Upscale quantized codes by given scaling rates (each rate is a power of 2).
    Uses ConvTranspose2d steps to match reference VQ-VAE-2.
    """

    def __init__(self, embed_dim: int, scaling_rates: list[int]) -> None:
        super().__init__()
        self.stages = nn.ModuleList()
        for sr in scaling_rates:
            upscale_steps = int(log2(sr))
            stage_layers = []
            for _ in range(upscale_steps):
                stage_layers.append(
                    nn.ConvTranspose2d(embed_dim, embed_dim, 4, stride=2, padding=1)
                )
                stage_layers.append(nn.ReLU(inplace=True))
            self.stages.append(nn.Sequential(*stage_layers))

    def forward(self, x: torch.Tensor, stage: int) -> torch.Tensor:
        return self.stages[stage](x)
