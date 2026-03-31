"""
sDSR - Spectrogram Dual Subspace Reconstruction

Discrete Autoencoder (VQ-VAE-2 style) for spectrogram anomaly detection.
Architecture aligned with reference VQ-VAE-2: generic Encoder/Decoder with
downscale/upscale factors, ReZero residuals, BatchNorm, and 2-level hierarchy.

Input spectrograms are 2-D: (B, C, n_mels, T)
  B       – batch size
  C       – channels (1 for mono mel-spectrogram / 3 for RGB spectrogram)
  n_mels  – frequency bins (e.g. 128)
  T       – time frames (e.g. 256)

2-level flow (fine x4/x4, coarse x8/x8 from input):
  Encoder fine   : X -> f_fine        (x4/x4 down, 32x80 for 128x320, hidden_channels)
  Encoder coarse : f_fine -> f_coarse (x4 down -> x8/x8 from input, 16x40, hidden_channels)
  VQ coarse      : f_coarse -> z_coarse (1x1 conv to embed_dim) -> Q_coarse
  Decoder coarse : Q_coarse -> decoded_coarse (2x up to 32x80, hidden_channels)
  Fine input     : [f_fine, decoded_coarse] -> 1x1 conv -> z_fine -> Q_fine
  Upscaler       : Q_coarse (16x40) -> (32x80) in embed space (2x from coarse grid)
  Decoder fine   : [Q_coarse_up, Q_fine] -> X_out (x4/x4 up to 128x320)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .quantizer import VectorQuantizerEMA
from .encoders import EncoderFine, EncoderCoarse
from .decoders import DecoderFine, DecoderCoarse


# ---------------------------------------------------------------------------
# Discrete Model
# ---------------------------------------------------------------------------

class VQ_VAE_2Layer(nn.Module):
    """
    Two-layer VQ-VAE for spectrograms (reference VQ-VAE-2 structure).

    - Fine latent: x4/x4 down from input (e.g. 128x320 -> 32x80).
    - Coarse latent: x8/x8 down from input (e.g. 128x320 -> 16x40).
    - Coarse codebook: encoder_coarse -> 1x1 conv -> embed_dim -> VQ.
    - Coarse decoder: quantized_coarse -> Decoder (4x up) -> fine grid for conditioning.
    - Fine codebook: concat(encoder_fine, decoded_coarse) -> 1x1 conv -> embed_dim -> VQ.
    - Fine decoder: concat(upscaled_coarse, quantized_fine) -> Decoder -> image.
    """

    def __init__(
        self,
        hidden_channels: Tuple[int, int],
        num_residual_layers: int,
        num_embeddings: Tuple[int, int],
        embedding_dim: Tuple[int, int],
        commitment_cost: float,
        decay: float = 0.99,
        test: bool = False,
    ):
        super().__init__()
        self.test = test

        hidden_channels_coarse, hidden_channels_fine = hidden_channels
        self.hidden_channels_coarse = hidden_channels_coarse
        self.hidden_channels_fine = hidden_channels_fine
        self.res_channels_coarse = hidden_channels_coarse // 2
        self.res_channels_fine = hidden_channels_fine // 2


        embedding_dim_coarse, embedding_dim_fine = embedding_dim
        self.embedding_dim_coarse = embedding_dim_coarse
        self.embedding_dim_fine = embedding_dim_fine

        self.num_residual_layers = num_residual_layers

        num_embeddings_coarse, num_embeddings_fine = num_embeddings
        self.num_embeddings_fine = num_embeddings_fine
        self.num_embeddings_coarse = num_embeddings_coarse

        # Encoders (reference: first from image, rest from previous level output)
        self._encoder_fine = EncoderFine(
            in_channels=1,
            num_hiddens=self.hidden_channels_fine,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=self.res_channels_fine
        )
        self._encoder_coarse = EncoderCoarse(
            in_channels=self.hidden_channels_fine,
            num_hiddens=self.hidden_channels_coarse,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=self.res_channels_coarse
        )

        # Codebook input channels: coarse = hidden_channels; fine = hidden_channels + embed_dim (conditioned on decoded coarse)
        self._pre_vq_conv_coarse = nn.Conv2d(self.hidden_channels_coarse, self.embedding_dim_coarse, kernel_size=1, stride=1)

        self._pre_vq_conv_fine = nn.Conv2d(
            self.hidden_channels_fine + self.hidden_channels_fine, self.embedding_dim_fine, kernel_size=1, stride=1
        )

        # Vector quantizers
        self._vq_coarse = VectorQuantizerEMA(
            num_embeddings_coarse, self.embedding_dim_coarse, commitment_cost, decay
        )
        self._vq_fine = VectorQuantizerEMA(
            num_embeddings_fine, self.embedding_dim_fine, commitment_cost, decay
        )

        # Coarse decoder: quantized (embed_dim) at coarse res -> decoded at fine res (embed_dim). Upscale must match coarse downscale (scaling_rates[1]) so output matches f_fine grid.
        self._decoder_coarse = DecoderCoarse(
            in_channels=self.embedding_dim_coarse,
            num_hiddens=self.hidden_channels_coarse,
            out_channels=self.hidden_channels_fine,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=self.res_channels_coarse
        )
        # Fine decoder: concat(upscaled_coarse, quantized_fine) -> image
        self._decoder_fine = DecoderFine(
            in_channels=self.embedding_dim_fine + self.embedding_dim_coarse,
            num_hiddens=self.hidden_channels_fine,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=self.res_channels_fine
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        # Encode: fine (high-res) and coarse (low-res)
        f_fine = self._encoder_fine(x)
        f_coarse = self._encoder_coarse(f_fine)

        # Project and quantize coarse
        z_coarse = self._pre_vq_conv_coarse(f_coarse)
        loss_coarse, quantized_coarse, perplexity_coarse, _ = self._vq_coarse(z_coarse)

        # Decode coarse to embed_dim at fine spatial res (for conditioning fine codebook)
        decoded_coarse = self._decoder_coarse(quantized_coarse)
        feat_fine = torch.cat([f_fine, decoded_coarse], dim=1)

        # Project and quantize fine
        z_fine = self._pre_vq_conv_fine(feat_fine)
        loss_fine, quantized_fine, perplexity_fine, _ = self._vq_fine(z_fine)

        # Upscale coarse quantized to fine grid and decode jointly (reference: Upscaler then concat)
        quantized_coarse_up = F.interpolate(quantized_coarse, size=quantized_fine.shape[-2:], mode="bilinear", align_corners=False)
        quant_joined = torch.cat([quantized_coarse_up, quantized_fine], dim=1)
        recon = self._decoder_fine(quant_joined)

        return (
            loss_fine, loss_coarse, recon,
            quantized_coarse, quantized_fine,
            perplexity_coarse, perplexity_fine,
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode and quantize input spectrogram (for AudDSR inference/training).

        Args:
            x: (B, 3, n_mels, T) Mel spectrogram

        Returns:
            q_fine: (B, emb_dim, H_q, W_q) fine quantized features
            q_coarse: (B, emb_dim, H_q_coarse, W_q_coarse) coarse quantized features
        """
        q_fine, q_coarse, _, _ = self.encode_with_prequant(x)
        return q_fine, q_coarse

    def encode_with_prequant(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode with pre-quantize features (for AudDSR anomaly generation).

        Returns:
            q_fine, q_coarse, z_fine, z_coarse (z_* are pre-quantize continuous features)
        """
        f_fine = self._encoder_fine(x)
        f_coarse = self._encoder_coarse(f_fine)
        z_coarse = self._pre_vq_conv_coarse(f_coarse)
        _, quantized_coarse, _, _ = self._vq_coarse(z_coarse)
        decoded_coarse = self._decoder_coarse(quantized_coarse)
        feat_fine = torch.cat([f_fine, decoded_coarse], dim=1)
        z_fine = self._pre_vq_conv_fine(feat_fine)
        _, quantized_fine, _, _ = self._vq_fine(z_fine)
        return quantized_fine, quantized_coarse, z_fine, z_coarse

    def encode_to_indices(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode spectrogram to codebook indices only (for transmitter bitstream).

        Args:
            x: (B, 1, n_mels, T) Mel spectrogram

        Returns:
            indices_coarse: (B, H_coarse, W_coarse) long
            indices_fine: (B, H_fine, W_fine) long
        """
        f_fine = self._encoder_fine(x)
        f_coarse = self._encoder_coarse(f_fine)
        z_coarse = self._pre_vq_conv_coarse(f_coarse)
        idx_coarse_flat = self._vq_coarse.get_indices(z_coarse)
        _, quantized_coarse, _, _ = self._vq_coarse(z_coarse)
        decoded_coarse = self._decoder_coarse(quantized_coarse)
        feat_fine = torch.cat([f_fine, decoded_coarse], dim=1)
        z_fine = self._pre_vq_conv_fine(feat_fine)
        idx_fine_flat = self._vq_fine.get_indices(z_fine)
        B, _, H_coarse, W_coarse = z_coarse.shape
        _, _, H_fine, W_fine = z_fine.shape
        indices_coarse = idx_coarse_flat.view(B, H_coarse, W_coarse)
        indices_fine = idx_fine_flat.view(B, H_fine, W_fine)
        return indices_coarse, indices_fine

    def indices_to_quantized(
        self, indices_coarse: torch.Tensor, indices_fine: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Map codebook indices to quantized feature tensors (for receiver).

        Args:
            indices_coarse: (B, H_coarse, W_coarse) long
            indices_fine: (B, H_fine, W_fine) long

        Returns:
            q_fine: (B, emb_dim, H_fine, W_fine)
            q_coarse: (B, emb_dim, H_coarse, W_coarse)
        """
        emb_dim_c = int(self._vq_coarse._embedding_dim)
        emb_dim_f = int(self._vq_fine._embedding_dim)

        q_coarse = self._vq_coarse._embedding(indices_coarse.flatten())
        q_coarse = q_coarse.view(
            indices_coarse.shape[0],
            indices_coarse.shape[1],
            indices_coarse.shape[2],
            emb_dim_c,
        ).permute(0, 3, 1, 2)

        q_fine = self._vq_fine._embedding(indices_fine.flatten())
        q_fine = q_fine.view(
            indices_fine.shape[0],
            indices_fine.shape[1],
            indices_fine.shape[2],
            emb_dim_f,
        ).permute(0, 3, 1, 2)
        return q_fine, q_coarse

    def decode_general(
        self, q_fine: torch.Tensor, q_coarse: torch.Tensor
    ) -> torch.Tensor:
        """
        General object decoder: reconstruct spectrogram from quantized features.
        Preserves anomalies (for AudDSR X_G path). Frozen during stage 2.

        Args:
            q_fine: (B, emb_dim, H_q, W_q)
            q_coarse: (B, emb_dim, H_q_coarse, W_q_coarse)

        Returns:
            X_G: (B, 1, n_mels, T) reconstructed spectrogram
        """
        quantized_coarse_up = F.interpolate(q_coarse, size=q_fine.shape[-2:], mode="bilinear", align_corners=False)
        quant_joined = torch.cat([quantized_coarse_up, q_fine], dim=1)
        return self._decoder_fine(quant_joined)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Different codebook sizes: coarse smaller, fine larger
    m = VQ_VAE_2Layer(
        hidden_channels=(256,64),
        num_residual_layers=2,
        num_embeddings=(1024, 4096),
        embedding_dim=(256,64),
        commitment_cost=0.25,
        decay=0.99,
    ).to(device)

    x = torch.randn(1, 1, 128, 320, device=device)

    loss_fine, loss_coarse, recon, q_coarse, q_fine, perp_coarse, perp_fine = m(x)
    assert q_fine.shape[-2:] == (32, 80), f"q_fine shape {q_fine.shape}"
    assert q_coarse.shape[-2:] == (16, 40), f"q_coarse shape {q_coarse.shape}"
    assert recon.shape == (1, 1, 128, 320), f"recon shape {recon.shape}"
    n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"[VQ-VAE-2Layer] params={n_params:,}  loss_fine={loss_fine.item():.4f}  loss_coarse={loss_coarse.item():.4f}  recon={recon.shape}")
    print(f"Quantized coarse: {q_coarse.shape}")
    print(f"Quantized fine: {q_fine.shape}")

    print("Smoke test passed: fine 32x80, coarse 16x40, recon 128x320.")
