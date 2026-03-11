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

    embedding_dim: int = 64
    hidden_channels: int = 64
    num_residual_layers: int = 2
    n_mels: int = 128
    T: int = 320
    anomaly_sampling: Literal["distant", "uniform"] = "uniform"
    anomaly_strength_min: float = 0.2
    anomaly_strength_max: float = 1.0
    use_subspace_restriction: bool = False


class sDSR(nn.Module):
    """
    sDSR: Spectrogram Dual Subspace Re-projection for anomaly detection.

    Inference path (no anomaly simulation):
      X -> vq_vae.encode() -> q_fine, q_coarse
      q_fine, q_coarse -> vq_vae.decode_general() -> X_G
      q_fine, q_coarse -> ObjectSpecificDecoder -> X_S
      [X_G, X_S] -> AnomalyDetectionModule -> M_out

    Training path (stage 2): uses AnomalyMapGenerator and AnomalyGeneration
    to augment q_fine, q_coarse -> q_fine_a, q_coarse_a (both levels always);
    then X_S is reconstructed from augmented features.
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
            hidden_channels=cfg.hidden_channels,
            num_residual_layers=cfg.num_residual_layers,
            use_subspace_restriction=cfg.use_subspace_restriction,
        )
        self._anomaly_detection = AnomalyDetectionModule(
            in_channels=2,
            out_channels=2,
            base_width=cfg.hidden_channels//2,
        )

        # Anomaly generation (training only): codebook replacement using dataset-provided mask
        self._anomaly_generation = AnomalyGeneration(sampling=cfg.anomaly_sampling)

        self._freeze_stage1()

    def _get_q_shape(self, n_mels: int, T: int) -> tuple[int, int]:
        """Infer q_fine spatial shape from spectrogram shape (encoder fine: 4x4 symmetric down -> 32x80 for 128x320)."""
        H = n_mels // 4
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
        q_fine, q_coarse = self._vq_vae.encode(x)
        x_g = self._vq_vae.decode_general(q_fine, q_coarse)
        x_s = self._object_decoder(
            q_coarse, q_fine,
            self._vq_vae._vq_coarse, self._vq_vae._vq_fine,
            return_aux=False,
        )
        m_out = self._anomaly_detection(x_g, x_s.detach())
        if return_intermediates:
            return m_out, x_g, x_s
        return m_out

    def forward_from_quantized(
        self,
        q_fine: torch.Tensor,
        q_coarse: torch.Tensor,
        return_intermediates: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decoder-only path: run general decoder, object-specific decoder, and anomaly
        detection from quantized features (e.g. after receiving indices over the channel).

        Args:
            q_fine: (B, emb_dim, H_fine, W_fine)
            q_coarse: (B, emb_dim, H_coarse, W_coarse)
            return_intermediates: if True, return (M_out, X_G, X_S)

        Returns:
            M_out: (B, 2, n_mels, T) segmentation logits
        """
        x_g = self._vq_vae.decode_general(q_fine, q_coarse)
        x_s = self._object_decoder(
            q_coarse, q_fine,
            self._vq_vae._vq_coarse, self._vq_vae._vq_fine,
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
                x: input (for L2 reconstruction target)
                When subspace restriction is enabled, also:
                recon_feat_fine, recon_feat_coarse, q_fine, q_coarse — for full-batch
                subspace loss (all samples: normal and anomalous).
        """
        batch_size = x.shape[0]
        device = x.device

        q_fine, q_coarse, z_fine, z_coarse = self._vq_vae.encode_with_prequant(x)
        vq_fine = self._vq_vae._vq_fine
        vq_coarse = self._vq_vae._vq_coarse

        strength_fine = (
            torch.rand(batch_size, device=device) * (self.config.anomaly_strength_max - self.config.anomaly_strength_min)
            + self.config.anomaly_strength_min
        )
        strength_coarse = (
            torch.rand(batch_size, device=device) * (self.config.anomaly_strength_max - self.config.anomaly_strength_min)
            + self.config.anomaly_strength_min
        )
        q_fine_a, q_coarse_a = self._anomaly_generation(
            q_fine, q_coarse, M_gt, vq_fine, vq_coarse,
            z_fine=z_fine, z_coarse=z_coarse,
            strength_fine=strength_fine, strength_coarse=strength_coarse,
        )

        # Inject anomalies randomly fine level, coarse level, or both levels
        has_anomaly = (M_gt.sum(dim=(1, 2, 3)) > 0).view(batch_size, 1, 1, 1).float()
        inject_location = torch.randint(0, 3, (batch_size,))
        use_fine = ((inject_location == 0) | (inject_location == 2)).float().view(batch_size, 1, 1, 1)
        use_coarse = ((inject_location == 1) | (inject_location == 2)).float().view(batch_size, 1, 1, 1)
        
        q_fine_used = has_anomaly * (use_fine * q_fine_a + (1 - use_fine) * q_fine) + (1 - has_anomaly) * use_fine * q_fine
        q_coarse_used = has_anomaly * (use_coarse * q_coarse_a + (1 - use_coarse) * q_coarse) + (1 - has_anomaly) * use_coarse * q_coarse

        with torch.no_grad():
            x_g = self._vq_vae.decode_general(q_fine_used, q_coarse_used)

        out_dec = self._object_decoder(
            q_coarse_used, q_fine_used,
            vq_coarse, vq_fine,
            return_aux=True,
        )
        x_s, aux = out_dec
        m_out = self._anomaly_detection(x_g, x_s.detach())

        # GT mask for focal loss: M_gt at spectrogram shape (same as m_out spatial dims)
        result = {
            "m_out": m_out,
            "x_g": x_g,
            "x_s": x_s,
            "M": M_gt.float(),
            "x": x,
        }
        if "recon_feat_coarse" in aux and "recon_feat_fine" in aux:
            result["recon_feat_coarse"] = aux["recon_feat_coarse"]
            result["recon_feat_fine"] = aux["recon_feat_fine"]
            result["q_coarse"] = q_coarse
            result["q_fine"] = q_fine
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
        hidden_channels=128,
        num_residual_layers=2,
        num_embeddings=(4096, 4096),
        embedding_dim=128,
        commitment_cost=0.25,
        decay=0.99,
    )
    cfg = sDSRConfig(n_mels=128, T=320, embedding_dim=128, hidden_channels=128)
    model = sDSR(vq, cfg).to(device)
    x = _torch.randn(2, 1, 128, 320, device=device)

    print("sDSR pipeline shapes (inference):")
    print("-" * 50)
    _print_shape("input x (spectrogram)", x)

    q_fine, q_coarse = model._vq_vae.encode(x)
    _print_shape("q_fine (quantized fine)", q_fine)
    _print_shape("q_coarse (quantized coarse)", q_coarse)

    x_g = model._vq_vae.decode_general(q_fine, q_coarse)
    _print_shape("x_g (general reconstruction)", x_g)

    x_s = model._object_decoder(
        q_coarse, q_fine,
        model._vq_vae._vq_coarse, model._vq_vae._vq_fine,
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
    assert out["M"].shape == (2, 1, 128, 320), "M (loss target) must be spectrogram shape"
    print("sDSR smoke test passed.")
