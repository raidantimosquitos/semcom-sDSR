"""
sDSR-specific anomaly generation: augment VQ-VAE quantized features.

Wires generic generate_fake_anomalies_distant (or generate_fake_anomalies_uniform)
with VQ codebooks (vq_fine, vq_coarse).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.anomalies import (
    generate_fake_anomalies_distant,
    generate_fake_anomalies_uniform,
)


def _match_mask_size(mask: torch.Tensor, target: tuple[int, int]) -> torch.Tensor:
    """Pad or crop mask to target (H, W)."""
    H, W = target
    _, _, h, w = mask.shape
    if h < H or w < W:
        mask = F.pad(mask, (0, max(0, W - w), 0, max(0, H - h)), value=0.0)
    if mask.shape[2] > H or mask.shape[3] > W:
        mask = mask[:, :, :H, :W].contiguous()
    return mask


class AnomalyGeneration(nn.Module):
    """
    Augment (q_fine, q_coarse) with synthetic anomalies using VQ codebook sampling.

    sampling: "distant" = similarity-ordered subset (skip closest 5%, strength
    controls distance); "uniform" = draw uniformly from full codebook (no z/strength).
    """

    def __init__(self, sampling: str = "distant") -> None:
        super().__init__()
        self.sampling = sampling

    def forward(
        self,
        q_fine: torch.Tensor,
        q_coarse: torch.Tensor,
        M: torch.Tensor,
        vq_fine: nn.Module,
        vq_coarse: nn.Module,
        z_fine: torch.Tensor | None = None,
        z_coarse: torch.Tensor | None = None,
        strength_fine: torch.Tensor | float = 0.5,
        strength_coarse: torch.Tensor | float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Augment quantized features in mask regions.

        Args:
            q_fine, q_coarse: quantized features (B, emb_dim, H, W) each
            M: (B, 1, n_mels, T) anomaly mask at spectrogram shape; resized to
               fine/coarse latent grid dimensions (paper: "M is then resized to fit").
            vq_fine, vq_coarse: VectorQuantizerEMA modules (for codebook access)
            z_fine, z_coarse: optional pre-quantize features (used only for "distant")
            strength_fine, strength_coarse: anomaly strength [0.2, 1.0] (used only for "distant")

        Returns:
            (q_fine_a, q_coarse_a): augmented quantized features
        """
        cb_fine = vq_fine._embedding.weight
        cb_coarse = vq_coarse._embedding.weight
        M_float = M.float()
        # Any-active projection: max-pool so any latent cell with ≥1 active spec pixel is flagged
        H_spec, W_spec = M_float.shape[2], M_float.shape[3]
        H_fine, W_fine = q_fine.shape[2], q_fine.shape[3]
        H_coarse, W_coarse = q_coarse.shape[2], q_coarse.shape[3]
        stride_fine = (max(1, H_spec // H_fine), max(1, W_spec // W_fine))
        stride_coarse = (max(1, H_spec // H_coarse), max(1, W_spec // W_coarse))
        M_fine = F.avg_pool2d(M_float, kernel_size=stride_fine, stride=stride_fine)
        M_coarse = F.avg_pool2d(M_float, kernel_size=stride_coarse, stride=stride_coarse)
        if M_fine.shape[2] != H_fine or M_fine.shape[3] != W_fine:
            M_fine = _match_mask_size(M_fine, (H_fine, W_fine))
        if M_coarse.shape[2] != H_coarse or M_coarse.shape[3] != W_coarse:
            M_coarse = _match_mask_size(M_coarse, (H_coarse, W_coarse))

        if self.sampling == "uniform":
            q_coarse_a = generate_fake_anomalies_uniform(q_coarse, cb_coarse, M_coarse)
            q_fine_a = generate_fake_anomalies_uniform(q_fine, cb_fine, M_fine)
        else:
            z_fine = z_fine if z_fine is not None else q_fine
            z_coarse = z_coarse if z_coarse is not None else q_coarse
            q_coarse_a = generate_fake_anomalies_distant(
                z_coarse, q_coarse, cb_coarse, M_coarse, strength_coarse
            )
            q_fine_a = generate_fake_anomalies_distant(
                z_fine, q_fine, cb_fine, M_fine, strength_fine
            )
        return q_fine_a, q_coarse_a
