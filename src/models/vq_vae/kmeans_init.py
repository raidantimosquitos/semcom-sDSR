"""
K-means codebook initialization for VectorQuantizerEMA / VQ_VAE_2Layer.

Typical use after LR warmup: encoder has meaningful geometry; cluster latent vectors,
then call ``init_from_centroids`` so EMA state matches the embedding table.

Example (inside your training loop after ``warmup_iters`` steps):

    from src.models.vq_vae.kmeans_init import init_vqvae_codebooks_from_loader

    model.eval()
    init_vqvae_codebooks_from_loader(model, trainer.loader, trainer.device)
    model.train()
    # Optional: reset Adam moments for VQ params only (avoid stale momentum).
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from .autoencoders import VQ_VAE_2Layer
from .quantizer import VectorQuantizerEMA


def kmeans_lloyd(
    x: torch.Tensor,
    k: int,
    *,
    n_iter: int = 15,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Lloyd's algorithm on rows of ``x`` (N, D). Returns centroids (k, D) and counts (k,).

    Empty clusters are re-seeded from random training points each iteration.

    Builds permutation indices on CPU with a CPU :class:`torch.Generator`, then moves
    indices to ``x.device``. That avoids PyTorch inconsistencies where ``randperm(...,
    device=cuda, generator=cpu)`` may error depending on version.
    """
    n, d = x.shape
    if n < k:
        raise ValueError(f"need at least k={k} samples, got n={n}")
    device = x.device
    dtype = x.dtype
    cpu_g = torch.Generator(device="cpu")
    cpu_g.manual_seed(seed)

    perm = torch.randperm(n, generator=cpu_g)[:k].to(device=device)
    centroids = x[perm].clone()

    assign = torch.empty(n, dtype=torch.long, device=device)
    for _ in range(n_iter):
        dist = (
            x.pow(2).sum(dim=1, keepdim=True)
            + centroids.pow(2).sum(dim=1)
            - 2 * x @ centroids.t()
        )
        assign = dist.argmin(dim=1)

        sums = torch.zeros(k, d, device=device, dtype=dtype)
        sums.scatter_add_(0, assign.unsqueeze(1).expand(n, d), x)
        counts = torch.zeros(k, device=device, dtype=dtype)
        counts.scatter_add_(0, assign, torch.ones(n, device=device, dtype=dtype))

        nonempty = counts > 0
        centroids[nonempty] = sums[nonempty] / counts[nonempty].unsqueeze(1)

        empty_idx = (~nonempty).nonzero(as_tuple=False).squeeze(1)
        if empty_idx.numel() > 0:
            rnd = torch.randint(
                0,
                n,
                (empty_idx.numel(),),
                generator=cpu_g,
            ).to(device)
            centroids[empty_idx] = x[rnd]
            counts[empty_idx] = 1.0

    dist = (
        x.pow(2).sum(dim=1, keepdim=True)
        + centroids.pow(2).sum(dim=1)
        - 2 * x @ centroids.t()
    )
    assign = dist.argmin(dim=1)
    final_counts = torch.bincount(assign, minlength=k).to(dtype=dtype)
    final_counts = final_counts.clamp(min=1.0)

    sums = torch.zeros(k, d, device=device, dtype=dtype)
    sums.scatter_add_(0, assign.unsqueeze(1).expand(n, d), x)
    centroids = sums / final_counts.unsqueeze(1)

    return centroids, final_counts


def _flatten_spatial(z: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B*H*W, C)"""
    return z.permute(0, 2, 3, 1).reshape(-1, z.shape[1])


@torch.no_grad()
def _collect_z_coarse(model: VQ_VAE_2Layer, x: torch.Tensor) -> torch.Tensor:
    f_fine = model._encoder_fine(x)
    f_coarse = model._encoder_coarse(f_fine)
    z_coarse = model._pre_vq_conv_coarse(f_coarse)
    return _flatten_spatial(z_coarse)


@torch.no_grad()
def _collect_z_fine(model: VQ_VAE_2Layer, x: torch.Tensor) -> torch.Tensor:
    f_fine = model._encoder_fine(x)
    f_coarse = model._encoder_coarse(f_fine)
    z_coarse = model._pre_vq_conv_coarse(f_coarse)
    _, quantized_coarse, _, _ = model._vq_coarse(z_coarse)
    decoded_coarse = model._decoder_coarse(quantized_coarse)
    feat_fine = torch.cat([f_fine, decoded_coarse], dim=1)
    z_fine = model._pre_vq_conv_fine(feat_fine)
    return _flatten_spatial(z_fine)


def init_vqvae_codebooks_from_loader(
    model: VQ_VAE_2Layer,
    loader: DataLoader,
    device: torch.device | str,
    *,
    max_batches: int | None = None,
    max_samples: int = 500_000,
    kmeans_iters: int = 15,
    seed: int = 0,
) -> None:
    """
    Two-stage k-means: coarse latents first, then fine latents using the updated coarse VQ.

    Runs ``model.eval()`` internally; restore ``model.train()`` yourself if needed.

    Args:
        model: VQ_VAE_2Layer instance.
        loader: Training DataLoader (same batch content as stage1).
        device: Target device.
        max_batches: Optional cap on batches scanned for collection.
        max_samples: Cap total vectors per codebook before k-means (memory / speed).
        kmeans_iters: Lloyd iterations.
        seed: RNG seed for subsampling and k-means init.
    """
    device_t = torch.device(device) if isinstance(device, str) else device
    subsample_g = torch.Generator(device="cpu")
    subsample_g.manual_seed(int(seed))

    was_training = model.training
    model.eval()

    def gather_rows(collect_fn) -> torch.Tensor:
        chunks: list[torch.Tensor] = []
        total = 0
        for bi, batch in enumerate(loader):
            if max_batches is not None and bi >= max_batches:
                break
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            x = x.to(device_t, non_blocking=True)
            rows = collect_fn(model, x).float()
            chunks.append(rows.cpu())
            total += rows.shape[0]
            if total >= max_samples:
                break
        if not chunks:
            raise RuntimeError("no data collected for k-means init")
        x_all = torch.cat(chunks, dim=0)
        if x_all.shape[0] > max_samples:
            idx = torch.randperm(
                x_all.shape[0], generator=subsample_g
            )[:max_samples]
            x_all = x_all[idx]
        return x_all.to(device_t)

    # 1) Coarse codebook
    Xc = gather_rows(_collect_z_coarse)
    k_c = model._vq_coarse._num_embeddings
    centroids_c, counts_c = kmeans_lloyd(
        Xc, k_c, n_iter=kmeans_iters, seed=int(seed)
    )
    model._vq_coarse.init_from_centroids(centroids_c, counts_c)

    # 2) Fine codebook (uses new coarse quantization path)
    Xf = gather_rows(_collect_z_fine)
    k_f = model._vq_fine._num_embeddings
    centroids_f, counts_f = kmeans_lloyd(
        Xf, k_f, n_iter=kmeans_iters, seed=int(seed) + 10_007
    )
    model._vq_fine.init_from_centroids(centroids_f, counts_f)

    if was_training:
        model.train()


def init_single_quantizer_from_tensor(
    vq: VectorQuantizerEMA,
    latents_flat: torch.Tensor,
    *,
    kmeans_iters: int = 15,
    seed: int = 0,
) -> None:
    """
    K-means on pre-collected latent rows (N, D) matching ``vq`` embedding dim.

    Convenience when you already have tensors (e.g. saved activations).
    """
    k = vq._num_embeddings
    centroids, counts = kmeans_lloyd(
        latents_flat.float(), k, n_iter=kmeans_iters, seed=int(seed)
    )
    vq.init_from_centroids(centroids, counts)
