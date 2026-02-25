"""
sDSR: Spectrogram Dual Subspace Re-projection for anomaly detection.

Top-level model composing VQ-VAE (encoder + general decoder), object-specific
decoder, anomaly map generation, anomaly generation, and anomaly detection module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..vq_vae.autoencoders import VQ_VAE_2Layer

from .object_specific_decoder import ObjectSpecificDecoder
from .anomaly_generation import AnomalyGeneration
from .anomaly_detection import AnomalyDetectionModule


@dataclass
class sDSRConfig:
    """Configuration for sDSR model."""

    embedding_dim: int = 128
    num_hiddens: int = 128
    num_residual_layers: int = 2
    num_residual_hiddens: int = 64
    n_mels: int = 128
    T: int = 320
    anomaly_sampling: Literal["distant", "uniform"] = "uniform"
    anomaly_strength_min: float = 0.2
    anomaly_strength_max: float = 1.0
    use_subspace_restriction: bool = True


class sDSR(nn.Module):
    """
    sDSR: Spectrogram Dual Subspace Re-projection for anomaly detection.

    Inference path (no anomaly simulation):
      X -> vq_vae.encode() -> q_bot, q_top
      q_bot, q_top -> vq_vae.decode_general() -> X_G
      q_bot, q_top -> ObjectSpecificDecoder -> X_S
      [X_G, X_S] -> AnomalyDetectionModule -> M_out

    Training path (stage 2): uses AnomalyMapGenerator and AnomalyGeneration
    to augment q_bot, q_top -> q_bot_a, q_top_a, then X_S is reconstructed
    from augmented features.
    """

    def __init__(
        self,
        vq_vae: VQ_VAE_2Layer,
        config: sDSRConfig | None = None,
    ) -> None:
        """
        Args:
            vq_vae: Pre-trained VQ_VAE_2Layer (encoder/quantizers/general decoder)
            config: sDSRConfig; if None, uses defaults
        """
        super().__init__()
        self.config = config or sDSRConfig()
        cfg = self.config

        self._vq_vae = vq_vae
        self._object_decoder = ObjectSpecificDecoder(
            embedding_dim=cfg.embedding_dim,
            num_hiddens=cfg.num_hiddens,
            num_residual_layers=cfg.num_residual_layers,
            num_residual_hiddens=cfg.num_residual_hiddens,
            use_subspace_restriction=cfg.use_subspace_restriction,
        )
        self._anomaly_detection = AnomalyDetectionModule(
            in_channels=2,
            out_channels=2,
            base_width=64,
        )

        # Anomaly generation (training only): codebook replacement using dataset-provided mask
        self._anomaly_generation = AnomalyGeneration(sampling=cfg.anomaly_sampling)

        self._freeze_stage1()

    def _get_q_shape(self, n_mels: int, T: int) -> tuple[int, int]:
        """Infer q_bot spatial shape from spectrogram shape (VQ-VAE 4x + 2x downsampling)."""
        H = n_mels // 4  # 2x then 2x
        W = T // 4
        return (max(1, H), max(1, W))

    def _freeze_stage1(self) -> None:
        """Freeze VQ-VAE (encoder, quantizers, general decoder) in stage 2."""
        for p in self._vq_vae.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True) -> sDSR:
        """When training mode is on, keep VQ-VAE in eval so its EMA codebook is not updated."""
        super().train(mode)
        if mode:
            self._vq_vae.eval()
        return self

    def forward(
        self,
        x: torch.Tensor,
        return_intermediates: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Inference forward pass.

        Uses the standard path: X -> q -> X_G, X_S -> M_out.
        No anomaly simulation; deployable for evaluation.

        Args:
            x: (B, 1, n_mels, T) Mel spectrogram
            return_intermediates: if True, return (M_out, X_G, X_S, M_out)

        Returns:
            M_out: (B, 2, n_mels, T) segmentation logits
            If return_intermediates: (M_out, X_G, X_S, M_out) — M_out repeated for convenience
        """
        q_bot, q_top = self._vq_vae.encode(x)
        x_g = self._vq_vae.decode_general(q_bot, q_top)
        x_s = self._object_decoder(
            q_top, q_bot,
            self._vq_vae._vq_top, self._vq_vae._vq_bot,
            return_aux=False,
        )
        m_out = self._anomaly_detection(x_g, x_s.detach())
        if return_intermediates:
            return m_out, x_g, x_s
        return m_out

    def forward_train(
        self,
        x: torch.Tensor,
        M_gt: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Training forward pass (stage 2).

        Uses the dataset-provided anomaly map M_gt to augment q with synthetic
        anomalies (codebook replacement), then returns outputs for loss computation.

        Args:
            x: (B, 1, n_mels, T) Mel spectrogram (normal)
            M_gt: (B, 1, n_mels, T) ground-truth anomaly map from dataset

        Returns:
            dict with:
                m_out: (B, 2, n_mels, T) segmentation logits
                x_g: (B, 1, n_mels, T) general reconstruction
                x_s: (B, 1, n_mels, T) object-specific reconstruction
                M: (B, 1, n_mels, T) anomaly map for focal loss target
                x: input (for L2 target)
        """
        batch_size = x.shape[0]
        device = x.device
        q_shape = self._get_q_shape(
            self.config.n_mels if x.shape[-2] == self.config.n_mels else x.shape[-2],
            x.shape[-1],
        )
        M = F.interpolate(M_gt.float(), size=q_shape, mode="nearest")

        q_bot, q_top, z_bot, z_top = self._vq_vae.encode_with_prequant(x)
        vq_bot = self._vq_vae._vq_bot
        vq_top = self._vq_vae._vq_top

        strength_bot = (
            torch.rand(batch_size, device=device) * (self.config.anomaly_strength_max - self.config.anomaly_strength_min)
            + self.config.anomaly_strength_min
        )
        strength_top = (
            torch.rand(batch_size, device=device) * (self.config.anomaly_strength_max - self.config.anomaly_strength_min)
            + self.config.anomaly_strength_min
        )
        q_bot_a, q_top_a = self._anomaly_generation(
            q_bot, q_top, M, vq_bot, vq_top,
            z_bot=z_bot, z_top=z_top,
            strength_bot=strength_bot, strength_top=strength_top,
        )

        # Per-sample randomization: anomaly on both levels, only top, or only bottom (original DSR)
        use_both = torch.randint(
            0, 2, (batch_size,), device=device
        ).float().view(batch_size, 1, 1, 1)
        use_lo = torch.randint(
            0, 2, (batch_size,), device=device
        ).float().view(batch_size, 1, 1, 1)
        # use_both=1 -> both anomalous; use_both=0, use_lo=1 -> top only; use_both=0, use_lo=0 -> bottom only
        q_bot_final = (
            use_both * q_bot_a
            + (1 - use_both) * (use_lo * q_bot + (1 - use_lo) * q_bot_a)
        )
        q_top_final = (
            use_both * q_top_a
            + (1 - use_both) * (use_lo * q_top_a + (1 - use_lo) * q_top)
        )

        has_anomaly = (M.sum(dim=(1, 2, 3)) > 0).view(batch_size, 1, 1, 1).float()
        q_bot_used = has_anomaly * q_bot_final + (1 - has_anomaly) * q_bot
        q_top_used = has_anomaly * q_top_final + (1 - has_anomaly) * q_top

        with torch.no_grad():
            x_g = self._vq_vae.decode_general(q_bot_used, q_top_used)

        q_top_batch = q_top_used
        q_bot_batch = q_bot_used
        out_dec = self._object_decoder(
            q_top_batch, q_bot_batch,
            vq_top, vq_bot,
            return_aux=True,
        )
        x_s, aux = out_dec
        m_out = self._anomaly_detection(x_g, x_s)

        # GT mask for focal loss: per-level resize then blend with use_both/use_lo (original DSR)
        M_top = F.interpolate(M.float(), size=q_top.shape[-2:], mode="nearest")
        M_bot = F.interpolate(M.float(), size=q_bot.shape[-2:], mode="nearest")
        M_top_for_loss = F.interpolate(M_top, size=x.shape[-2:], mode="nearest")
        M_bot_for_loss = F.interpolate(M_bot, size=x.shape[-2:], mode="nearest")
        M_for_loss = (
            use_both * M_top_for_loss
            + (1 - use_both) * (use_lo * M_top_for_loss + (1 - use_lo) * M_bot_for_loss)
        )

        result = {
            "m_out": m_out,
            "x_g": x_g,
            "x_s": x_s,
            "M": M_for_loss,
            "x": x,
        }
        if "recon_feat_top" in aux and "recon_feat_bot" in aux:
            result["recon_feat_top"] = aux["recon_feat_top"]
            result["recon_feat_bot"] = aux["recon_feat_bot"]
            result["q_top"] = q_top
            result["q_bot"] = q_bot
        return result


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _print_shape(name: str, t: torch.Tensor) -> None:
    print(f"  {name}: {tuple(t.shape)}")


if __name__ == "__main__":
    import torch as _torch

    device = "cuda" if _torch.cuda.is_available() else "cpu"
    vq = VQ_VAE_2Layer(
        num_hiddens=128,
        num_residual_layers=2,
        num_residual_hiddens=64,
        num_embeddings=(4096, 4096),
        embedding_dim=128,
        commitment_cost=0.25,
        decay=0.99,
    )
    cfg = sDSRConfig(n_mels=128, T=320, embedding_dim=128, num_hiddens=128)
    model = sDSR(vq, cfg).to(device)
    x = _torch.randn(2, 1, 128, 320, device=device)

    print("sDSR pipeline shapes (inference):")
    print("-" * 50)
    _print_shape("input x (spectrogram)", x)

    q_bot, q_top = model._vq_vae.encode(x)
    _print_shape("q_bot (quantized bottom / fine)", q_bot)
    _print_shape("q_top (quantized top / coarse)", q_top)

    x_g = model._vq_vae.decode_general(q_bot, q_top)
    _print_shape("x_g (general reconstruction)", x_g)

    x_s = model._object_decoder(
        q_top, q_bot,
        model._vq_vae._vq_top, model._vq_vae._vq_bot,
        return_aux=False,
    )
    _print_shape("x_s (object-specific reconstruction)", x_s)

    m_out = model._anomaly_detection(x_g, x_s.detach())
    _print_shape("m_out (segmentation logits)", m_out)
    print("-" * 50)

    m_out = model(x)
    assert m_out.shape == (2, 2, 128, 320)
    m_out, x_g, x_s = model(x, return_intermediates=True)
    assert x_g.shape == (2, 1, 128, 320)
    assert x_s.shape == (2, 1, 128, 320)

    print("Training path shapes:")
    print("-" * 50)
    model.train()
    M_gt = _torch.zeros(2, 1, 128, 320, device=device)
    M_gt[0] = 1.0  # one sample with anomaly mask for testing
    out = model.forward_train(x, M_gt=M_gt)
    _print_shape("forward_train -> m_out", out["m_out"])
    _print_shape("forward_train -> x_g", out["x_g"])
    _print_shape("forward_train -> x_s", out["x_s"])
    _print_shape("forward_train -> M (anomaly map)", out["M"])
    _print_shape("forward_train -> x (input)", out["x"])
    print("-" * 50)

    assert out["m_out"].shape == (2, 2, 128, 320)
    assert out["x_s"].shape == (2, 1, 128, 320)
    print("sDSR smoke test passed.")
