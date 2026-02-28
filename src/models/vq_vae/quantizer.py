"""
Vector quantizers for VQ-VAE.

- VectorQuantizerEMA_v2: EMA with commitment loss, used by VQ_VAE_2Layer
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast
from typing import Tuple

class VectorQuantizerEMA(nn.Module):
    """
    EMA vector quantizer with commitment cost (VQ-VAE style).
    Returns (loss, quantized, perplexity, encodings).
    Reference: https://github.com/zalandoresearch/pytorch-vq-vae
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float,
        decay: float,
        epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._commitment_cost = commitment_cost
        self._decay = decay
        self._epsilon = epsilon

        self._embedding = nn.Embedding(num_embeddings, embedding_dim)
        self._embedding.weight.data.normal_()

        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self._ema_w.data.normal_()

    def forward(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            loss: commitment loss (scalar)
            quantized: quantized tensor (B, C, H, W)
            perplexity: codebook utilization metric
            encodings: one-hot encodings (N, num_embeddings)

        Runs in float32 under AMP to avoid NaN from distance computation and EMA updates.
        """
        input_dtype = inputs.dtype
        # Run entire quantizer in float32 for numerical stability (AMP-safe)
        with autocast(device_type=inputs.device.type, enabled=False):
            inputs = inputs.float()
            inputs = inputs.permute(0, 2, 3, 1).contiguous()
            input_shape = inputs.shape
            flat_input = inputs.view(-1, self._embedding_dim)

            # Embedding in float32 for stable distances and EMA
            emb = self._embedding.weight.float()
            distances = (
                flat_input.pow(2).sum(dim=1, keepdim=True)
                + emb.pow(2).sum(dim=1)
                - 2 * flat_input @ emb.t()
            )

            encoding_indices = distances.argmin(dim=1).unsqueeze(1)
            encodings = torch.zeros(
                encoding_indices.shape[0], self._num_embeddings,
                device=inputs.device, dtype=torch.float32,
            )
            encodings.scatter_(1, encoding_indices, 1.0)

            quantized = (encodings @ emb).view(input_shape)

            if self.training:
                self._ema_cluster_size.mul_(self._decay).add_(
                    encodings.sum(dim=0), alpha=1.0 - self._decay
                )
                n = self._ema_cluster_size.sum()
                self._ema_cluster_size.data = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self._num_embeddings * self._epsilon) * n
                )

                dw = encodings.t() @ flat_input
                self._ema_w.data.mul_(self._decay).add_(dw, alpha=1.0 - self._decay)
                self._embedding.weight.data.copy_(
                    self._ema_w / self._ema_cluster_size.unsqueeze(1).clamp(min=1e-5)
                )

            e_latent_loss = F.mse_loss(quantized.detach(), inputs)
            loss = self._commitment_cost * e_latent_loss

            quantized = inputs + (quantized - inputs).detach()
            avg_probs = encodings.mean(dim=0)
            perplexity = torch.exp(-(avg_probs * (avg_probs + 1e-10).log()).sum())

            quantized = quantized.permute(0, 3, 1, 2).contiguous()
            if input_dtype != torch.float32:
                quantized = quantized.to(input_dtype)
            loss = loss.to(input_dtype)
            perplexity = perplexity.to(input_dtype)

        return (
            loss,
            quantized,
            perplexity,
            encodings,
        )
