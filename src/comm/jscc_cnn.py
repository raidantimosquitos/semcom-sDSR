"""
Lightweight JSCC encoder/decoder for VQ-VAE-2 index maps over AWGN.

Design:
- indices -> embedding lookup (fixed codebook embedding dim)
- lightweight **depthwise-separable conv** encoder over (H,W)
- flatten -> linear to fixed number of channel uses (cu) per clip
- power normalize -> AWGN -> linear back -> reshape
- lightweight **depthwise-separable conv** decoder + 1x1 head

This is intentionally simple and fast. It does NOT aim for bit-exact index recovery;
it reconstructs quantized feature tensors (q_coarse/q_fine) directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


def awgn_real(x: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
    """
    Real AWGN channel with per-sample SNR.
    Assumes x is power-normalized to mean power 1.
    """
    # SNR = Es/N0 for real baseband with unit power -> noise_var = 1/snr
    snr = 10.0 ** (snr_db / 10.0)
    noise_var = 1.0 / snr
    std = torch.sqrt(noise_var).view(-1, 1)
    return x + torch.randn_like(x) * std


def power_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # Normalize each sample to unit average power
    p = (x.pow(2).mean(dim=1, keepdim=True)).clamp(min=eps)
    return x / torch.sqrt(p)


@dataclass(frozen=True)
class JSCCMapConfig:
    num_embeddings: int
    embedding_dim: int
    H: int
    W: int
    alpha: int  # channel uses per symbol (integer)

    @property
    def n_symbols(self) -> int:
        return int(self.H * self.W)

    @property
    def n_channel_uses(self) -> int:
        return int(self.alpha * self.n_symbols)


class JSCCMapAE(nn.Module):
    """
    JSCC autoencoder for one index map (coarse OR fine).
    """

    def __init__(self, cfg: JSCCMapConfig, hidden: int = 256, n_blocks: int = 4) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.num_embeddings, cfg.embedding_dim)
        in_dim = hidden * cfg.n_symbols
        out_dim = cfg.n_channel_uses

        def dsconv(in_ch: int, out_ch: int) -> nn.Module:
            return nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False),
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.GELU(),
            )

        # Encoder conv: C -> hidden
        enc_blocks = [dsconv(cfg.embedding_dim, hidden)]
        for _ in range(max(0, n_blocks - 1)):
            enc_blocks.append(dsconv(hidden, hidden))
        self.enc_conv = nn.Sequential(*enc_blocks)

        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
        self.dec = nn.Sequential(
            nn.Linear(out_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, in_dim),
        )
        # Decoder conv: hidden -> hidden -> C
        dec_blocks = []
        for _ in range(max(1, n_blocks)):
            dec_blocks.append(dsconv(hidden, hidden))
        self.dec_conv = nn.Sequential(*dec_blocks)
        self.head = nn.Conv2d(hidden, cfg.embedding_dim, kernel_size=1)

    def forward(self, indices: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        """
        Args:
            indices: (B, H, W) long
            snr_db: (B,) float tensor
        Returns:
            q_hat: (B, C=embedding_dim, H, W) float
        """
        B, H, W = indices.shape
        assert H == self.cfg.H and W == self.cfg.W
        e = self.embed(indices).permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        h = self.enc_conv(e)  # (B, hidden, H, W)
        x = h.reshape(B, -1)
        z = self.enc(x)
        z = power_normalize(z)
        z_noisy = awgn_real(z, snr_db=snr_db)
        h_hat = self.dec(z_noisy).reshape(B, -1, H, W)  # (B, hidden, H, W)
        h_hat = self.dec_conv(h_hat)
        return self.head(h_hat)


class JSCCDualMap(nn.Module):
    """
    Dual-map JSCC for (coarse,fine) maps jointly (separate AEs for simplicity).
    """

    def __init__(self, coarse: JSCCMapConfig, fine: JSCCMapConfig) -> None:
        super().__init__()
        self.coarse_cfg = coarse
        self.fine_cfg = fine
        self.coarse_ae = JSCCMapAE(coarse)
        self.fine_ae = JSCCMapAE(fine)

    @property
    def channel_uses_per_clip(self) -> int:
        return int(self.coarse_cfg.n_channel_uses + self.fine_cfg.n_channel_uses)

    def forward(
        self,
        idx_coarse: torch.Tensor,
        idx_fine: torch.Tensor,
        snr_db: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_coarse_hat = self.coarse_ae(idx_coarse, snr_db=snr_db)
        q_fine_hat = self.fine_ae(idx_fine, snr_db=snr_db)
        return q_coarse_hat, q_fine_hat

