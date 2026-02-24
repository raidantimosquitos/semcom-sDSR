"""
Generic anomaly generation: replace masked feature vectors with codebook samples.

sDSR-style distant sampling (sample from top-k most distant codebook entries).
Model-agnostic: accepts (z, embeddings, codebook, mask, strength).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def generate_fake_anomalies_distant(
    z: torch.Tensor,
    embeddings: torch.Tensor,
    codebook: torch.Tensor,
    mask: torch.Tensor,
    strength: torch.Tensor | float,
) -> torch.Tensor:
    """
    Replace feature vectors in mask regions with distant codebook samples.

    sDSR-style: compute distances to codebook, take top-k nearest (smallest dist),
    skip closest 5%, sample from remaining; higher strength = more distant.

    Args:
        z: (B, emb_dim, H, W) continuous features for distance computation
        embeddings: (B, emb_dim, H, W) quantized features to augment
        codebook: (num_embeddings, emb_dim) VQ codebook weights
        mask: (B, 1, H, W) anomaly mask, positive = replace
        strength: (B,) or scalar; fraction [anom_par, 1.0] controlling how
                  distant the replacement is (higher = more distant)

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

    for k in range(B):
        flat_input = inputs[k].view(-1, C)
        distances = (
            flat_input.pow(2).sum(dim=1, keepdim=True)
            + codebook.to(device).pow(2).sum(dim=1)
            - 2 * flat_input @ codebook.to(device).t()
        )
        pct = strength[k].item()
        topk = max(1, min(int(pct * N) + 1, N - 1))
        _, topk_indices = torch.topk(distances, topk, dim=1, largest=False)
        skip = int(N * 0.05)
        topk_indices = topk_indices[:, skip:]
        topk_n = topk_indices.shape[1]
        if topk_n < 1:
            topk_indices = topk_indices[:, -1:]
            topk_n = 1
        rand_col = torch.randint(topk_n, (topk_indices.shape[0],), device=device)
        chosen = topk_indices[torch.arange(topk_indices.shape[0], device=device), rand_col]
        random_vecs = codebook.to(device)[chosen]
        random_embeddings[k] = random_vecs.view(H, W, C).permute(2, 0, 1)

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
