"""
Anomaly Detection Module for sDSR.

UNet: [X_G, X_S] (2 ch) -> segmentation logits (2 ch: normal vs anomaly).
Trained with Focal loss during stage 2.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _UnetEncoder(nn.Module):
    def __init__(self, in_channels: int, base_width: int) -> None:
        super().__init__()
        norm = nn.InstanceNorm2d
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, base_width, kernel_size=3, padding=1),
            norm(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            norm(base_width),
            nn.ReLU(inplace=True),
        )
        self.mp1 = nn.MaxPool2d(2)
        self.block2 = nn.Sequential(
            nn.Conv2d(base_width, base_width * 2, kernel_size=3, padding=1),
            norm(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            norm(base_width * 2),
            nn.ReLU(inplace=True),
        )
        self.mp2 = nn.MaxPool2d(2)
        self.block3 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width * 4, kernel_size=3, padding=1),
            norm(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            norm(base_width * 4),
            nn.ReLU(inplace=True),
        )
        self.mp3 = nn.MaxPool2d(2)
        self.block4 = nn.Sequential(
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            norm(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            norm(base_width * 4),
            nn.ReLU(inplace=True),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b1 = self.block1(x)
        mp1 = self.mp1(b1)
        b2 = self.block2(mp1)
        mp2 = self.mp2(b2)
        b3 = self.block3(mp2)
        mp3 = self.mp3(b3)
        b4 = self.block4(mp3)
        return b1, b2, b3, b4


class _UnetDecoder(nn.Module):
    def __init__(self, base_width: int, out_channels: int) -> None:
        super().__init__()
        norm = nn.InstanceNorm2d
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            norm(base_width * 4),
            nn.ReLU(inplace=True),
        )
        self.db1 = nn.Sequential(
            nn.Conv2d(base_width * 8, base_width * 4, kernel_size=3, padding=1),
            norm(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            norm(base_width * 4),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_width * 4, base_width * 2, kernel_size=3, padding=1),
            norm(base_width * 2),
            nn.ReLU(inplace=True),
        )
        self.db2 = nn.Sequential(
            nn.Conv2d(base_width * 4, base_width * 2, kernel_size=3, padding=1),
            norm(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            norm(base_width * 2),
            nn.ReLU(inplace=True),
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_width * 2, base_width, kernel_size=3, padding=1),
            norm(base_width),
            nn.ReLU(inplace=True),
        )
        self.db3 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width, kernel_size=3, padding=1),
            norm(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            norm(base_width),
            nn.ReLU(inplace=True),
        )
        self.fin_out = nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1)

    def forward(
        self,
        b1: torch.Tensor,
        b2: torch.Tensor,
        b3: torch.Tensor,
        b4: torch.Tensor,
    ) -> torch.Tensor:
        up1 = self.up1(b4)
        cat1 = torch.cat([up1, b3], dim=1)
        db1 = self.db1(cat1)
        up2 = self.up2(db1)
        cat2 = torch.cat([up2, b2], dim=1)
        db2 = self.db2(cat2)
        up3 = self.up3(db2)
        cat3 = torch.cat([up3, b1], dim=1)
        db3 = self.db3(cat3)
        return self.fin_out(db3)


class AnomalyDetectionModule(nn.Module):
    """
    UNet: [X_G, X_S] -> segmentation logits.

    Input: (B, 2, n_mels, T) — concatenation of general and object-specific reconstructions
    Output: (B, 2, n_mels, T) — logits (normal vs anomaly)
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_width: int = 64,
    ) -> None:
        super().__init__()
        self.unet_enc = _UnetEncoder(in_channels, base_width)
        self.unet_dec = _UnetDecoder(base_width, out_channels)

    def forward(
        self,
        x_g: torch.Tensor,
        x_s: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_g: (B, 1, n_mels, T) general reconstruction (anomalies preserved)
            x_s: (B, 1, n_mels, T) object-specific reconstruction (anomaly-free)

        Returns:
            logits: (B, 2, n_mels, T) segmentation logits [normal, anomaly]
        """
        x = torch.cat([x_g, x_s], dim=1)
        b1, b2, b3, b4 = self.unet_enc(x)
        return self.unet_dec(b1, b2, b3, b4)
