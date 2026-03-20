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


class DecoderFine(nn.Module):
    """
    Upsampling: 4x in frequency, 4x in time (inverse of EncoderFine).
    Input (n_mels//4, T//4) -> output (n_mels, T), e.g. (32, 80) -> (128, 320).
    """
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(DecoderFine, self).__init__()

        self._conv_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_hiddens,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
        )
        self._conv_trans_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels=num_hiddens, 
                out_channels=num_hiddens//2,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )
        self._conv_trans_2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels=num_hiddens//2,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = self._residual_stack(x)
        x = self._conv_trans_1(x)
        x = F.relu(x)
        x = self._conv_trans_2(x)
        return x

class DecoderCoarse(nn.Module):
    """
    Upsamples coarse latent to fine latent grid (2x in each dimension).
    Input (emb_dim, n_mels//8, T//8) e.g. (16, 40);
    output (num_hiddens, n_mels//4, T//4) e.g. (32, 80).
    """
    def __init__(self, in_channels, num_hiddens, out_channels, num_residual_layers, num_residual_hiddens):
        super(DecoderCoarse, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)

        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

        self._conv_trans_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels=num_hiddens, 
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1, padding=1)
        )

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = self._residual_stack(x)
        x = self._conv_trans_1(x)
        x = F.relu(x)
        return x
