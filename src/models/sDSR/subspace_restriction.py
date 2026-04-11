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
import torch.nn.functional as F


class FeatureEncoder(nn.Module):
    """Encoder for SubspaceRestrictionNetwork: 3 blocks, 2× down each → 4× total down."""

    def __init__(self, in_channels: int, base_width: int) -> None:
        super().__init__()
        norm = nn.GroupNorm
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, base_width, kernel_size=3, padding=1),
            norm(8, base_width),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            norm(8, base_width),
            nn.SiLU(inplace=True),
        )
        # self.mp1 = nn.MaxPool2d(2)
        self.mp1 = nn.Sequential(
            nn.Conv2d(base_width, base_width, kernel_size=3, stride=2, padding=1),
            norm(8, base_width),
            nn.SiLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(base_width, base_width * 2, kernel_size=3, padding=1),
            norm(8, base_width * 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            norm(8, base_width * 2),
            nn.SiLU(inplace=True),
        )
        # self.mp2 = nn.MaxPool2d(2)
        self.mp2 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, stride=2, padding=1),
            norm(8, base_width * 2),
            nn.SiLU(inplace=True),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width * 4, kernel_size=3, padding=1),
            norm(8, base_width * 4),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            norm(8, base_width * 4),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b1 = self.block1(x)
        mp1 = self.mp1(b1)
        b2 = self.block2(mp1)
        mp2 = self.mp2(b2)
        b3 = self.block3(mp2)
        return b1, b2, b3


class FeatureDecoder(nn.Module):
    """Decoder for SubspaceRestrictionNetwork: 2 ups with skip connections, same spatial as input."""

    def __init__(self, base_width: int, out_channels: int) -> None:
        super().__init__()
        norm = nn.GroupNorm
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_width * 4, base_width * 2, kernel_size=3, padding=1),
            norm(8, base_width * 2),
            nn.SiLU(inplace=True),
        )
        self.db2 = nn.Sequential(
            nn.Conv2d(base_width * 4, base_width * 2, kernel_size=3, padding=1),
            norm(8, base_width * 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            norm(8, base_width * 2),
            nn.SiLU(inplace=True),
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_width * 2, base_width, kernel_size=3, padding=1),
            norm(8, base_width),
            nn.SiLU(inplace=True),
        )
        self.db3 = nn.Sequential(
            nn.Conv2d(base_width*2, base_width, kernel_size=3, padding=1),
            norm(8, base_width),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            norm(8, base_width),
            nn.SiLU(inplace=True),
        )
        self.fin_out = nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.fin_out.weight)
        if self.fin_out.bias is not None:
            nn.init.zeros_(self.fin_out.bias)

    def forward(
        self,
        b1: torch.Tensor,
        b2: torch.Tensor,
        b3: torch.Tensor,
    ) -> torch.Tensor:
        up2 = self.up2(b3)
        db2 = self.db2(torch.cat([up2, b2], dim=1))

        up3 = self.up3(db2)
        db3 = self.db3(torch.cat([up3, b1], dim=1))

        out = self.fin_out(db3)
        return out


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
        self.encoder = FeatureEncoder(in_channels, base_width)
        self.decoder = FeatureDecoder(base_width, out_channels=out_channels)
        bottleneck_dim = base_width * 4  # 256 with default base_width=64
        self.bottleneck_attn = nn.MultiheadAttention(
            embed_dim=bottleneck_dim,
            num_heads=8,
            batch_first=True,
        )
        # Learned 2D positional embedding for the 8×20 bottleneck grid
        # H/4=8, W/4=20 → 160 tokens
        self.pos_embed_fine = nn.Parameter(
            torch.zeros(1, 8 * 20, bottleneck_dim)
        )
        self.pos_embed_coarse = nn.Parameter(
            torch.zeros(1, 4 * 10, bottleneck_dim)
        )
        nn.init.trunc_normal_(self.pos_embed_fine, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_coarse, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1, b2, b3 = self.encoder(x)
        B, C, H, W = b3.shape
        if H == 8 and W == 20:
            pos_embed = self.pos_embed_fine
        elif H == 4 and W == 10:
            pos_embed = self.pos_embed_coarse
        else:
            raise ValueError(f"Invalid input shape: {H}x{W}")
        tokens = b3.flatten(2).permute(0, 2, 1)
        tokens = tokens + pos_embed
        tokens, _ = self.bottleneck_attn(tokens, tokens, tokens)
        b3 = tokens.permute(0, 2, 1).reshape(B, C, H, W)

        return self.decoder(b1, b2, b3)


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
            base_width=embedding_size//2,
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
        feat = self._unet(x) + x
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
