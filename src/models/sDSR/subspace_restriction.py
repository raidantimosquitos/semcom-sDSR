"""
Subspace restriction module for sDSR (DSR-style).

Maps anomaly-augmented quantized features to a subspace-restricted representation F̃,
then quantizes F̃ via the frozen VQ for the Object Specific Decoder.
Loss term: L2(F̃, Q) where Q is the non-anomalous quantized (detached).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from ..vq_vae.res_blocks_2d import ResidualStack


class FeatureEncoder(nn.Module):
    """Encoder for SubspaceRestrictionNetwork: 3 blocks, 2× down each → 4× total down."""

    def __init__(self, in_channels: int, base_width: int) -> None:
        super().__init__()
        norm = nn.InstanceNorm2d
        self._block1 = nn.Sequential(
            nn.Conv2d(in_channels, base_width, kernel_size=3, padding=1),
            norm(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            norm(base_width),
            nn.ReLU(inplace=True),
        )
        self._mp1 = nn.MaxPool2d(2)
        # self._mp1 = nn.Sequential(
        #     nn.Conv2d(base_width, base_width * 2, kernel_size=3, stride=2, padding=1),
        #     norm(base_width * 2),
        #     nn.ReLU(inplace=True),
        # )
        self._block2 = nn.Sequential(
            nn.Conv2d(base_width, base_width * 2, kernel_size=3, padding=1),
            norm(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            norm(base_width * 2),
            nn.ReLU(inplace=True),
        )
        self._mp2 = nn.MaxPool2d(2)
        # self._mp2 = nn.Sequential(
        #     nn.Conv2d(base_width * 2, base_width * 4, kernel_size=3, stride=2, padding=1),
        #     norm(base_width * 4),
        #     nn.ReLU(inplace=True),
        # )
        self._block3 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width * 4, kernel_size=3, padding=1),
            norm(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            norm(base_width * 4),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b1 = self._block1(x)
        mp1 = self._mp1(b1)
        b2 = self._block2(mp1)
        mp2 = self._mp2(b2)
        b3 = self._block3(mp2)
        return b1, b2, b3


class FeatureDecoder(nn.Module):
    """Decoder for SubspaceRestrictionNetwork: 2 ups from bottleneck only (no skip
    connections).  The information bottleneck forces the network to learn the clean
    subspace rather than memorising local corrections for synthetic anomalies."""

    def __init__(self, base_width: int, out_channels: int) -> None:
        super().__init__()
        norm = nn.InstanceNorm2d
        self._up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_width * 4, base_width * 2, kernel_size=3, stride=1, padding=1),
            norm(base_width * 2),
            nn.ReLU(inplace=True),
        )
        self._db2 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            norm(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            norm(base_width * 2),
            nn.ReLU(inplace=True),
        )
        self._up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_width * 2, base_width, kernel_size=3, stride=1, padding=1),
            norm(base_width),
            nn.ReLU(inplace=True),
        )
        self._db3 = nn.Sequential(
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            norm(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            norm(base_width),
            nn.ReLU(inplace=True),
        )
        self._fin_out = nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1)

    def forward(self, b3: torch.Tensor) -> torch.Tensor:
        up2 = self._up2(b3)
        db2 = self._db2(up2)

        up3 = self._up3(db2)
        db3 = self._db3(up3)

        return self._fin_out(db3)


class SubspaceRestrictionNetwork(nn.Module):
    """
    Small UNet in latent space. Input/output: (B, embedding_dim, H, W) -> (B, embedding_dim, H, W).
    Same structure as DSR SubspaceRestrictionNetwork (FeatureEncoder + FeatureDecoder).
    """

    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 128,
        base_width: int = 64,
    ) -> None:
        super().__init__()
        self._encoder = FeatureEncoder(in_channels, base_width)
        self._decoder = FeatureDecoder(base_width, out_channels=out_channels)
        # self.residual_stack = ResidualStack(in_channels*2, in_channels*2, 2, in_channels//2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, b3 = self._encoder(x)
        # b3 = self.residual_stack(b3)
        return self._decoder(b3)


class SubspaceRestrictionModule(nn.Module):
    """
    Maps anomaly-augmented quantized features to F̃ via UNet, then quantizes F̃
    with the frozen VQ. Returns (F̃, quantized(F̃), vq_loss) for decoder and loss.
    """

    def __init__(self, embedding_size: int = 128) -> None:
        super().__init__()
        self._unet = SubspaceRestrictionNetwork(
            in_channels=embedding_size,
            out_channels=embedding_size,
            base_width=embedding_size,
        )

    def forward(
        self,
        x: torch.Tensor,
        quantization: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, emb_dim, H, W) anomaly-augmented quantized features.
            quantization: Frozen VQ module (_vq_coarse or _vq_fine); callable as quantization(inputs).

        Returns:
            feat: F̃ (B, emb_dim, H, W) subspace-restricted continuous features.
            quantized: quantized(F̃) for feeding to Object Specific Decoder.
            loss_vq: VQ commitment loss (can be ignored in total loss).
        """
        feat = self._unet(x)
        loss_vq, quantized, _perp, _enc = quantization(feat)
        return feat, quantized, loss_vq


if __name__ == "__main__":
    B, C, H, W = 2, 128, 32, 80
    x = torch.randn(B, C, H, W)
    module = SubspaceRestrictionModule(embedding_size=128)
    # Mock VQ: same interface as VectorQuantizerEMA
    class MockVQ(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self._embedding_dim = 128
            self._num_embeddings = 4096
        def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            loss = inputs.mean()
            return loss, inputs, torch.tensor(1.0), torch.zeros(inputs.shape[0] * inputs.shape[2] * inputs.shape[3], self._num_embeddings)
    vq = MockVQ()
    feat, quantized, loss_vq = module(x, vq)
    assert feat.shape == x.shape
    assert quantized.shape == x.shape
    print("SubspaceRestrictionModule smoke test passed.")
