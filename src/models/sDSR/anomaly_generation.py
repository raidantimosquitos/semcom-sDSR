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


def spec_to_latent_pool_stride(
    H_spec: int,
    W_spec: int,
    H_lat: int,
    W_lat: int,
) -> tuple[int, int]:
    """Kernel/stride for pooling a spectrogram-shaped mask onto a latent grid."""
    return (max(1, H_spec // H_lat), max(1, W_spec // W_lat))


def project_spec_mask_to_latent_binary(
    M: torch.Tensor,
    H_lat: int,
    W_lat: int,
) -> torch.Tensor:
    """
    Pool ``M`` from spectrogram resolution to ``(H_lat, W_lat)``, then binarize.

    Same rule as :meth:`AnomalyGeneration.forward`: avg_pool over each latent
    footprint, pad/crop to exact latent shape, then ``> 0`` so any active spec
    pixel marks the latent cell.

    Args:
        M: (B, 1, H_spec, W_spec) mask at spectrogram resolution
        H_lat, W_lat: target latent height and width

    Returns:
        (B, 1, H_lat, W_lat) float in ``{0.0, 1.0}``
    """
    M_float = M.float()
    H_spec, W_spec = M_float.shape[2], M_float.shape[3]
    stride = spec_to_latent_pool_stride(H_spec, W_spec, H_lat, W_lat)
    M_lat = F.avg_pool2d(M_float, kernel_size=stride, stride=stride)
    if M_lat.shape[2] != H_lat or M_lat.shape[3] != W_lat:
        M_lat = _match_mask_size(M_lat, (H_lat, W_lat))
    return (M_lat > 0).float()


def upsample_latent_mask_to_spec(
    M_lat: torch.Tensor,
    n_mels: int,
    T: int,
) -> torch.Tensor:
    """
    Nearest upsample a latent mask to spectrogram size (piecewise-constant blocks).

    Matches DSR-style focal supervision: latent cell values are expanded to
    full-resolution targets without soft blending between cells.

    Args:
        M_lat: (B, 1, H_lat, W_lat)
        n_mels, T: target spectrogram height and width

    Returns:
        (B, 1, n_mels, T)
    """
    return F.interpolate(M_lat, size=(n_mels, T), mode="nearest")


class AnomalyGeneration(nn.Module):
    """
    Augment (q_fine, q_coarse) with synthetic anomalies using VQ codebook sampling.

    Always returns both augmented tensors (mask projected to fine and coarse grids).
    :class:`sDSR.forward_train` then applies fine-only, coarse-only, or both,
    per anomaly sample.

    sampling: "distant" = similarity-ordered subset (skip closest 5%, strength
    controls distance); "uniform" = draw uniformly from full codebook (no z/strength).
    """

    def __init__(self, sampling: str = "distant", neighbor_prob: float = 0.05) -> None:
        super().__init__()
        self.sampling = sampling
        self.neighbor_prob = neighbor_prob

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
        *,
        augment_coarse: bool = True,
        augment_fine: bool = True,
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
            augment_coarse, augment_fine: set False to skip that level (e.g. fine-only second pass).

        Returns:
            (q_fine_a, q_coarse_a): augmented quantized features
        """
        cb_fine = vq_fine._embedding.weight
        cb_coarse = vq_coarse._embedding.weight
        H_fine, W_fine = q_fine.shape[2], q_fine.shape[3]
        H_coarse, W_coarse = q_coarse.shape[2], q_coarse.shape[3]
        M_fine = project_spec_mask_to_latent_binary(M, H_fine, W_fine)
        M_coarse = project_spec_mask_to_latent_binary(M, H_coarse, W_coarse)

        if not augment_coarse:
            q_coarse_a = q_coarse
        elif self.sampling == "uniform":
            q_coarse_a = generate_fake_anomalies_uniform(q_coarse, cb_coarse, M_coarse)
        else:
            z_c = z_coarse if z_coarse is not None else q_coarse
            q_coarse_a = generate_fake_anomalies_distant(
                z_c, q_coarse, cb_coarse, M_coarse, strength_coarse,
                neighbor_prob=self.neighbor_prob,
            )

        if not augment_fine:
            q_fine_a = q_fine
        elif self.sampling == "uniform":
            q_fine_a = generate_fake_anomalies_uniform(q_fine, cb_fine, M_fine)
        else:
            z_f = z_fine if z_fine is not None else q_fine
            q_fine_a = generate_fake_anomalies_distant(
                z_f, q_fine, cb_fine, M_fine, strength_fine,
                neighbor_prob=self.neighbor_prob,
            )
        return q_fine_a, q_coarse_a
