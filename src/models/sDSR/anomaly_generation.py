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


def spec_to_latent_pool_stride(
    H_spec: int,
    W_spec: int,
    H_lat: int,
    W_lat: int,
) -> tuple[int, int]:
    """
    DSR down_ratio per axis: ``(H_spec//H_lat, W_spec//W_lat)`` with floor division.

    Used as ``kernel_size`` and ``stride`` in :func:`project_spec_mask_to_latent_binary`
    (same as the reference ``int(mask/emb)`` rule). Each component is at least 1.
    """
    return (max(1, H_spec // H_lat), max(1, W_spec // W_lat))


def project_spec_mask_to_latent_binary(
    M: torch.Tensor,
    H_lat: int,
    W_lat: int,
) -> torch.Tensor:
    """
    DSR rule: pool the spectrogram mask with ``max_pool2d`` using
    ``kernel_size = stride = (H_spec//H_lat, W_spec//W_lat)`` — no padding,
    no crop, no forced resize to ``(H_lat, W_lat)``.

    Binarize with ``> 0`` so any positive mass in a window yields 1. Output
    height/width follow PyTorch pooling; they match ``(H_lat, W_lat)`` when
    dimensions are commensurate (typical DSR / VQ case). If they differ,
    ``generate_fake_anomalies_distant`` / ``generate_fake_anomalies_uniform``
    align the mask to the feature map with nearest interpolation.

    Args:
        M: (B, 1, H_spec, W_spec) mask at spectrogram resolution
        H_lat, W_lat: spatial size of the corresponding latent / quantized map
            (used only to define the down_ratio, as in DSR)

    Returns:
        (B, 1, H', W') with ``H'``/``W'`` the pooled size; values in ``{0.0, 1.0}``
    """
    M_float = M.float()
    H_spec, W_spec = M_float.shape[2], M_float.shape[3]
    k = spec_to_latent_pool_stride(H_spec, W_spec, H_lat, W_lat)
    M_lat = F.max_pool2d(M_float, kernel_size=k, stride=k)
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

    sampling: "distant" = similarity-ordered subset (skip closest codebook fraction:
    3% coarse, 5% fine; strength controls range); "uniform" = draw uniformly from
    full codebook (no z/strength).
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
        *,
        augment_coarse: bool = True,
        augment_fine: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Augment quantized features in mask regions.

        Args:
            q_fine, q_coarse: quantized features (B, emb_dim, H, W) each
            M: (B, 1, n_mels, T) anomaly mask at spectrogram shape; DSR
               max-pool to latent scale (no forced resize; see
               :func:`project_spec_mask_to_latent_binary`).
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
                closest_skip_frac=0.03,
            )

        if not augment_fine:
            q_fine_a = q_fine
        elif self.sampling == "uniform":
            q_fine_a = generate_fake_anomalies_uniform(q_fine, cb_fine, M_fine)
        else:
            z_f = z_fine if z_fine is not None else q_fine
            q_fine_a = generate_fake_anomalies_distant(
                z_f, q_fine, cb_fine, M_fine, strength_fine,
                closest_skip_frac=0.05,
            )
        return q_fine_a, q_coarse_a
