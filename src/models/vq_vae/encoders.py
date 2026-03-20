from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F

from .res_blocks_2d import ResidualStack

class EncoderFine(nn.Module):
    """
    Downsampling: 4x in frequency, 4x in time.
    Input (n_mels, T) -> output (n_mels//2, T//4), e.g. (128, 320) -> (32, 80).
    """
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(EncoderFine, self).__init__()

        # One conv: 4x down freq, 4x down time
        self._conv_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_hiddens // 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self._conv_2 = nn.Conv2d(
            in_channels=num_hiddens // 2,
            out_channels=num_hiddens,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self._conv_3 = nn.Conv2d(
            in_channels=num_hiddens,
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

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)
        x = self._conv_2(x)
        x = F.relu(x)
        x = self._conv_3(x)
        return self._residual_stack(x)


class EncoderCoarse(nn.Module):
    """
    Downsampling: 2x in frequency, 2x in time (from fine grid).
    Input (n_mels//2, T//4) -> output (n_mels//8, T//16), e.g. (64, 80) -> (16, 20).
    """
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(EncoderCoarse, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens//2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens//2,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)
        x = self._conv_2(x)
        x = F.relu(x)
        x = self._conv_3(x)
        return self._residual_stack(x)