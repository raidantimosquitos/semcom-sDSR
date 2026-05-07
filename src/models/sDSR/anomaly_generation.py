"""
sDSR-specific anomaly generation: augment VQ-VAE quantized features.

Defines two codebook sampling strategies for codeword replacement in VQ codebooks (vq_fine, vq_coarse) and handles projection of the anomaly mask to the latent space.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def shuffle_patches(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Shuffle non-overlapping patches using unfold/fold (reference DSR).

    Note: if H/W are not divisible by patch_size, unfold will drop the border
    and fold will leave those regions as zeros (same as the reference code).
    """
    if patch_size <= 1:
        return x
    ps = int(patch_size)
    u = F.unfold(x, kernel_size=ps, stride=ps, padding=0)
    pu = torch.cat([b_[:, torch.randperm(b_.shape[-1], device=x.device)][None, ...] for b_ in u], dim=0)
    f = F.fold(pu, x.shape[-2:], kernel_size=ps, stride=ps, padding=0)
    return f


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


def generate_fake_anomalies_distant(
    z: torch.Tensor,
    embeddings: torch.Tensor,
    codebook: torch.Tensor,
    mask: torch.Tensor,
    strength: torch.Tensor | float,
    closest_skip_frac: float = 0.05,
    use_shuffle: bool = True,
) -> torch.Tensor:
    """
    Replace feature vectors in mask regions with codebook samples.

    Distant regime: skip closest ``closest_skip_frac`` of codebook by rank,
    then sample from the nearest subset sized by ``strength``.

    Args:
        z: (B, emb_dim, H, W) continuous features for distance computation
        embeddings: (B, emb_dim, H, W) quantized features to augment
        codebook: (num_embeddings, emb_dim) VQ codebook weights
        mask: (B, 1, H, W) anomaly mask, positive = replace
        strength: (B,) or scalar; fraction controlling distant-mode range
        closest_skip_frac: in distant mode, fraction of nearest codebook entries
            to exclude before sampling

    Returns:
        Augmented embeddings (B, emb_dim, H, W)
    """
    B, C = embeddings.shape[0], embeddings.shape[1]
    H, W = embeddings.shape[2], embeddings.shape[3]
    device = embeddings.device
    N = codebook.shape[0]
    if mask.shape[-2:] != (H, W):
        mask = F.interpolate(mask.float(), size=(H, W), mode="nearest")

    if isinstance(strength, (int, float)):
        strength = torch.full((B,), float(strength), device=device)

    random_embeddings = torch.empty_like(embeddings, device=device)
    inputs = z.permute(0, 2, 3, 1).contiguous()
    cb = codebook.to(device)

    for k in range(B):
        flat_input = inputs[k].view(-1, C)
        distances = (
            flat_input.pow(2).sum(dim=1, keepdim=True)
            + cb.pow(2).sum(dim=1)
            - 2 * flat_input @ cb.t()
        )
        pct = strength[k].item()
        topk = max(1, min(int(pct * N) + 1, N - 1))
        _, topk_indices = torch.topk(distances, topk, dim=1, largest=False)
        skip = int(N * closest_skip_frac)
        topk_indices = topk_indices[:, skip:]

        topk_n = topk_indices.shape[1]
        if topk_n < 1:
            topk_indices = topk_indices[:, -1:]
            topk_n = 1
        rand_col = torch.randint(topk_n, (topk_indices.shape[0],), device=device)
        chosen = topk_indices[torch.arange(topk_indices.shape[0], device=device), rand_col]
        random_vecs = cb[chosen]
        random_embeddings[k] = random_vecs.view(H, W, C).permute(2, 0, 1)

    # Optional patch-shuffle mode (as in reference DSR code)
    if use_shuffle:
        use_shuffle_draw = torch.rand((), device=device).item()
        if use_shuffle_draw > 0.5:
            psize_factor = int(torch.randint(0, 4, (1,), device=device).item())  # 0..3 => 1,2,4,8
            random_embeddings = shuffle_patches(embeddings, 2**psize_factor)

    mask_exp = mask.expand_as(embeddings)
    return mask_exp * random_embeddings + (1 - mask_exp) * embeddings


def generate_fake_anomalies_uniform(
    embeddings: torch.Tensor,
    codebook: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Replace feature vectors in mask regions with codebook samples drawn uniformly.

    Each masked position is replaced by codebook[random_index] with random_index
    uniform in [0, N). No distance computation or strength parameter.

    Args:
        embeddings: (B, emb_dim, H, W) quantized features to augment
        codebook: (num_embeddings, emb_dim) VQ codebook weights
        mask: (B, 1, H, W) anomaly mask, positive = replace

    Returns:
        Augmented embeddings (B, emb_dim, H, W)
    """
    B, C = embeddings.shape[0], embeddings.shape[1]
    H, W = embeddings.shape[2], embeddings.shape[3]
    device = embeddings.device
    N = codebook.shape[0]
    if mask.shape[-2:] != (H, W):
        mask = F.interpolate(mask.float(), size=(H, W), mode="nearest")

    random_indices = torch.randint(
        0, N, (B, H, W), device=device, dtype=torch.long
    )
    cb = codebook.to(device)
    random_vecs = cb[random_indices]
    random_vecs = random_vecs.permute(0, 3, 1, 2)

    mask_exp = mask.expand_as(embeddings)
    return mask_exp * random_vecs + (1 - mask_exp) * embeddings



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
                closest_skip_frac=0.05,
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
