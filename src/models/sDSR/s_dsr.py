"""
sDSR: Spectrogram Dual Subspace Re-projection for anomaly detection.

Top-level model composing VQ-VAE (encoder + general decoder), object-specific
decoder, anomaly map generation, anomaly generation, and anomaly detection module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..vq_vae.autoencoders import VQ_VAE_2Layer

from .object_specific_decoder import ObjectSpecificDecoder
from .anomaly_generation import (
    AnomalyGeneration,
    project_spec_mask_to_latent_binary,
    upsample_latent_mask_to_spec,
)
from .anomaly_detection import AnomalyDetectionModule


# Per-machine-type codebook sampling presets (derived from L2 distance maps)
SAMPLING_PRESETS: dict[str, dict] = {
    "pump":         {"neighbor_prob": 0.05, "anomaly_strength_min": 0.05, "anomaly_strength_max": 1.0},
    "slider":       {"neighbor_prob": 0.05, "anomaly_strength_min": 0.05, "anomaly_strength_max": 1.0},
    "valve":        {"neighbor_prob": 0.05, "anomaly_strength_min": 0.05, "anomaly_strength_max": 1.0},
    "ToyCar":       {"neighbor_prob": 0.05, "anomaly_strength_min": 0.05, "anomaly_strength_max": 1.0},
    "ToyConveyor":  {"neighbor_prob": 0.05, "anomaly_strength_min": 0.05, "anomaly_strength_max": 1.0},
    "fan":          {"neighbor_prob": 0.05, "anomaly_strength_min": 0.05, "anomaly_strength_max": 1.0},
}


@dataclass
class sDSRConfig:
    """Configuration for sDSR model."""

    embedding_dim: Tuple[int, int] = (128, 128)
    hidden_channels: Tuple[int, int] = (128, 128)
    num_residual_layers: int = 2
    n_mels: int = 128
    T: int = 320
    anomaly_sampling: Literal["distant", "uniform"] = "distant"
    anomaly_strength_min: float = 0.2
    anomaly_strength_max: float = 1.0
    neighbor_prob: float = 0.05
    use_subspace_restriction: bool = True
    # Stage-2 latent injection: "uniform" = P(fine-only)=P(coarse-only)=P(both)=1/3;
    # "dsr" = P(both)=0.5, P(fine-only)=P(coarse-only)=0.25 (same tree as DSR use_both then use_hi/use_lo).
    anomaly_inj_distribution: Literal["uniform", "dsr"] = "dsr"
    machine_type: str | None = None

    def __post_init__(self) -> None:
        if self.machine_type is not None and self.machine_type in SAMPLING_PRESETS:
            preset = SAMPLING_PRESETS[self.machine_type]
            self.neighbor_prob = preset.get("neighbor_prob", self.neighbor_prob)
            self.anomaly_strength_min = preset.get("anomaly_strength_min", self.anomaly_strength_min)
            self.anomaly_strength_max = preset.get("anomaly_strength_max", self.anomaly_strength_max)


class sDSR(nn.Module):
    """
    sDSR: Spectrogram Dual Subspace Re-projection for anomaly detection.

    Inference path (no anomaly simulation):
      X -> vq_vae.encode() -> q_fine, q_coarse
      q_fine, q_coarse -> vq_vae.decode_general() -> X_general
      q_fine, q_coarse -> ObjectSpecificDecoder -> X_specific
      [X_specific, X_general] -> AnomalyDetectionModule -> M_out

    Training path (stage 2): AnomalyGeneration builds augmented codes from the
    spectrogram mask projected to fine and coarse; per anomaly sample we then
    randomly inject at fine only, coarse only, or both levels.
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
            base_width=128,
        )

        self._anomaly_generation = AnomalyGeneration(
            sampling=cfg.anomaly_sampling,
            neighbor_prob=cfg.neighbor_prob,
        )

        self._freeze_stage1()

    @staticmethod
    def _sample_inj_mode(
        batch_size: int,
        device: torch.device,
        distribution: Literal["uniform", "dsr"],
    ) -> torch.Tensor:
        """
        Per-sample injection mode for synthetic anomalies (used only when M_gt is non-empty).

        Returns:
            Long tensor (B,) with 0 = fine only, 1 = coarse only, 2 = both.
        """
        if distribution == "uniform":
            return torch.randint(0, 3, (batch_size,), device=device)
        u = torch.rand(batch_size, device=device)
        sub = torch.rand(batch_size, device=device)
        return torch.where(
            u >= 0.5,
            torch.full((batch_size,), 2, device=device, dtype=torch.long),
            torch.where(
                sub < 0.5,
                torch.zeros(batch_size, device=device, dtype=torch.long),
                torch.ones(batch_size, device=device, dtype=torch.long),
            ),
        )

    def _get_q_shape(self, n_mels: int, T: int) -> tuple[int, int]:
        """Infer q_fine spatial shape from spectrogram (fine x2/x4 down -> 64x80 for 128x320)."""
        H = n_mels // 2
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

        Uses the standard path: X -> q -> X_general, X_specific -> M_out.
        No anomaly simulation; deployable for evaluation.

        Args:
            x: (B, 1, n_mels, T) Mel spectrogram
            return_intermediates: if True, return (M_out, X_general, X_specific)

        Returns:
            M_out: (B, 2, n_mels, T) segmentation logits
            If return_intermediates: (M_out, X_general, X_specific)
        """
        q_fine, q_coarse = self._vq_vae.encode(x)
        x_general = self._vq_vae.decode_general(q_fine, q_coarse)
        x_specific = self._object_decoder(
            q_coarse, q_fine,
            self._vq_vae._vq_coarse, self._vq_vae._vq_fine,
            return_aux=False,
        )
        m_out = self._anomaly_detection(x_specific.detach(), x_general.detach())
        if return_intermediates:
            return m_out, x_general, x_specific
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
            return_intermediates: if True, return (M_out, X_general, X_specific)

        Returns:
            M_out: (B, 2, n_mels, T) segmentation logits
        """
        x_general = self._vq_vae.decode_general(q_fine, q_coarse)
        x_specific = self._object_decoder(
            q_coarse, q_fine,
            self._vq_vae._vq_coarse, self._vq_vae._vq_fine,
            return_aux=False,
        )
        m_out = self._anomaly_detection(x_specific.detach(), x_general.detach())
        if return_intermediates:
            return m_out, x_general, x_specific
        return m_out

    def forward_train(
        self,
        x: torch.Tensor,
        M_gt: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Training forward pass (stage 2).

        Uses the dataset-provided anomaly map M_gt to augment q with synthetic
        anomalies (codebook replacement). For each sample with a non-zero mask,
        injection mode is sampled per ``anomaly_inj_distribution`` (uniform 1/3 each,
        or DSR-style: 0.5 both, 0.25 fine-only, 0.25 coarse-only). Coarse or both
        modes recompute fine latents from ``f_fine`` and
        ``q_coarse_used`` so conditioning matches the hierarchy. Normal samples
        (empty mask) use clean (q_fine, q_coarse).

        Args:
            x: (B, 3, n_mels, T) Mel spectrogram (normal)
            M_gt: (B, 1, n_mels, T) ground-truth anomaly map from dataset

        Returns:
            dict with:
                m_out: (B, 2, n_mels, T) segmentation logits
                x_g: (B, 3, n_mels, T) general reconstruction
                x_s: (B, 3, n_mels, T) object-specific reconstruction
                M: (B, 1, n_mels, T) DSR-style focal target (latent-snapped mask
                    upsampled to spectrogram resolution; see ``M_gt`` for raw mask)
                M_gt: (B, 1, n_mels, T) original dataset mask (debugging / logging)
                x: input (for L2 reconstruction target)
                When subspace restriction is enabled, also:
                recon_feat_fine, recon_feat_coarse, q_fine, q_coarse — for full-batch
                subspace loss (all samples: normal and anomalous).
        """
        batch_size = x.shape[0]
        device = x.device

        f_fine, _, q_fine, q_coarse, z_fine, z_coarse = self._vq_vae.encode_with_prequant(x)
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
        # Single spectrogram mask M_gt → projected to fine and coarse in AnomalyGeneration (spatially coherent)
        q_fine_a, q_coarse_a = self._anomaly_generation(
            q_fine, q_coarse, M_gt, vq_fine, vq_coarse,
            z_fine=z_fine, z_coarse=z_coarse,
            strength_fine=strength_fine, strength_coarse=strength_coarse,
        )

        has_anomaly = (M_gt.sum(dim=(1, 2, 3)) > 0).view(batch_size, 1, 1, 1).float()
        # Per sample: 0 = fine only, 1 = coarse only (recompute fine), 2 = both.
        inj_mode = self._sample_inj_mode(
            batch_size, device, self.config.anomaly_inj_distribution
        )
        is_fine_only = has_anomaly * (inj_mode == 0).float().view(batch_size, 1, 1, 1)
        is_coarse_only = has_anomaly * (inj_mode == 1).float().view(batch_size, 1, 1, 1)
        is_both = has_anomaly * (inj_mode == 2).float().view(batch_size, 1, 1, 1)

        use_coarse_a = is_coarse_only + is_both
        q_coarse_used = use_coarse_a * q_coarse_a + (1.0 - use_coarse_a) * q_coarse

        with torch.no_grad():
            q_fine_recomp, z_fine_recomp = self._vq_vae.recompute_fine_from_coarse(
                f_fine, q_coarse_used
            )
            q_fine_a2, _ = self._anomaly_generation(
                q_fine_recomp,
                q_coarse_used,
                M_gt,
                vq_fine,
                vq_coarse,
                z_fine=z_fine_recomp,
                z_coarse=z_coarse,
                strength_fine=strength_fine,
                strength_coarse=strength_coarse,
                augment_coarse=False,
            )
            q_fine_used = (
                (1.0 - has_anomaly) * q_fine
                + is_fine_only * q_fine_a
                + is_coarse_only * q_fine_recomp
                + is_both * q_fine_a2
            )
            x_general = self._vq_vae.decode_general(q_fine_used, q_coarse_used)

        out_dec = self._object_decoder(
            q_coarse_used, q_fine_used,
            vq_coarse, vq_fine,
            return_aux=True,
        )
        x_specific, aux = out_dec
        m_out = self._anomaly_detection(x_specific.detach(), x_general.detach())

        # DSR-style focal target: pool M_gt to fine/coarse latent grids (same as
        # injection), nearest-upsample to spectrogram resolution, mix by inj_mode.
        _, _, n_mels, T_spec = M_gt.shape
        H_fine, W_fine = q_fine.shape[2], q_fine.shape[3]
        H_coarse, W_coarse = q_coarse.shape[2], q_coarse.shape[3]
        M_hi_lat = project_spec_mask_to_latent_binary(M_gt, H_fine, W_fine)
        M_lo_lat = project_spec_mask_to_latent_binary(M_gt, H_coarse, W_coarse)
        M_hi = upsample_latent_mask_to_spec(M_hi_lat, n_mels, T_spec)
        M_lo = upsample_latent_mask_to_spec(M_lo_lat, n_mels, T_spec)
        M_focal = has_anomaly * (
            is_fine_only * M_hi + (is_coarse_only + is_both) * M_lo
        )

        result = {
            "m_out": m_out,
            "x_general": x_general,
            "x_specific": x_specific,
            "M": M_focal,
            "M_gt": M_gt.float(),
            "x": x,
        }
        if "recon_feat_coarse" in aux and "recon_feat_fine" in aux:
            result["recon_feat_coarse"] = aux["recon_feat_coarse"]
            result["recon_feat_fine"] = aux["recon_feat_fine"]
            # L_sub targets clean quantized codes from encode (not augmented).
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
        hidden_channels=(256, 64),
        num_residual_layers=2,
        num_embeddings=(4096, 4096),
        embedding_dim=(256, 64),
        commitment_cost=0.25,
        decay=0.99,
    )
    cfg = sDSRConfig(n_mels=128, T=320, embedding_dim=(256, 64), hidden_channels=(256, 64))
    model = sDSR(vq, cfg).to(device)
    x = _torch.randn(2, 1, 128, 320, device=device)

    print("sDSR pipeline shapes (inference):")
    print("-" * 50)
    _print_shape("input x (spectrogram)", x)

    q_fine, q_coarse = model._vq_vae.encode(x)
    _print_shape("q_fine (quantized fine)", q_fine)
    _print_shape("q_coarse (quantized coarse)", q_coarse)

    x_general = model._vq_vae.decode_general(q_fine, q_coarse)
    _print_shape("x_general (general reconstruction)", x_general)

    x_specific = model._object_decoder(
        q_coarse, q_fine,
        model._vq_vae._vq_coarse, model._vq_vae._vq_fine,
        return_aux=False,
    )
    _print_shape("x_specific (object-specific reconstruction)", x_specific)

    m_out = model._anomaly_detection(x_specific, x_general.detach())
    _print_shape("m_out (segmentation logits)", m_out)
    print("-" * 50)

    m_out = model(x)
    assert m_out.shape == (2, 2, 128, 320)
    m_out, x_general, x_specific = model(x, return_intermediates=True)
    assert x_general.shape == (2, 1, 128, 320)
    assert x_specific.shape == (2, 1, 128, 320)

    print("Training path shapes:")
    print("-" * 50)
    model.train()
    M_gt = _torch.zeros(2, 1, 128, 320, device=device)
    M_gt[0] = 1.0  # one sample with anomaly mask for testing
    out = model.forward_train(x, M_gt=M_gt)
    _print_shape("forward_train -> m_out", out["m_out"])
    _print_shape("forward_train -> x_general", out["x_general"])
    _print_shape("forward_train -> x_specific", out["x_specific"])
    _print_shape("forward_train -> M (DSR-style focal target)", out["M"])
    _print_shape("forward_train -> M_gt (raw mask)", out["M_gt"])
    _print_shape("forward_train -> x (input)", out["x"])
    print("-" * 50)

    assert out["m_out"].shape == (2, 2, 128, 320)
    assert out["x_specific"].shape == (2, 1, 128, 320)
    assert out["M"].shape == (2, 1, 128, 320), "M (focal target) must be spectrogram shape"
    assert out["M_gt"].shape == (2, 1, 128, 320)
    uniq = _torch.unique(out["M"])
    assert bool((_torch.isin(uniq, _torch.tensor([0.0, 1.0], device=device))).all()), (
        "M focal target should be binary for binary M_gt"
    )
    print("sDSR smoke test passed.")
