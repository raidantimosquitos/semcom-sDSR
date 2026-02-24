"""
sDSR-specific anomaly generation: augment VQ-VAE quantized features.

Wires generic generate_fake_anomalies_distant (or generate_fake_anomalies_uniform)
with VQ codebooks (vq_bot, vq_top).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.anomalies import (
    generate_fake_anomalies_distant,
    generate_fake_anomalies_uniform,
)


class AnomalyGeneration(nn.Module):
    """
    Augment (q_bot, q_top) with synthetic anomalies using VQ codebook sampling.

    sampling: "distant" = similarity-ordered subset (skip closest 5%, strength
    controls distance); "uniform" = draw uniformly from full codebook (no z/strength).
    """

    def __init__(self, sampling: str = "distant") -> None:
        super().__init__()
        self.sampling = sampling

    def forward(
        self,
        q_bot: torch.Tensor,
        q_top: torch.Tensor,
        M: torch.Tensor,
        vq_bot: nn.Module,
        vq_top: nn.Module,
        z_bot: torch.Tensor | None = None,
        z_top: torch.Tensor | None = None,
        strength_bot: torch.Tensor | float = 0.5,
        strength_top: torch.Tensor | float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Augment quantized features in mask regions.

        Args:
            q_bot, q_top: quantized features (B, emb_dim, H, W) each
            M: (B, 1, H_q_bot, W_q_bot) anomaly mask; will be resized per level
            vq_bot, vq_top: VectorQuantizerEMA_v2 modules (for codebook access)
            z_bot, z_top: optional pre-quantize features (used only for "distant")
            strength_bot, strength_top: anomaly strength [0.2, 1.0] (used only for "distant")

        Returns:
            (q_bot_a, q_top_a): augmented quantized features
        """
        cb_bot = vq_bot._embedding.weight
        cb_top = vq_top._embedding.weight
        M_top = F.interpolate(M.float(), size=q_top.shape[-2:], mode="nearest")
        M_bot = F.interpolate(M.float(), size=q_bot.shape[-2:], mode="nearest")

        if self.sampling == "uniform":
            q_top_a = generate_fake_anomalies_uniform(q_top, cb_top, M_top)
            q_bot_a = generate_fake_anomalies_uniform(q_bot, cb_bot, M_bot)
        else:
            z_bot = z_bot if z_bot is not None else q_bot
            z_top = z_top if z_top is not None else q_top
            q_top_a = generate_fake_anomalies_distant(
                z_top, q_top, cb_top, M_top, strength_top
            )
            q_bot_a = generate_fake_anomalies_distant(
                z_bot, q_bot, cb_bot, M_bot, strength_bot
            )
        return q_bot_a, q_top_a
