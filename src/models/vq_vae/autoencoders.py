"""
sDSR - Spectrogram Dual Subspace Reconstruction

Discrete Autoencoder (VQ-VAE-2 style) for spectrogram anomaly detection.

Input spectrograms are 2-D: (B, C, n_mels, T)
  B       – batch size
  C       – channels (1 for mono mel-spectrogram)
  n_mels  – frequency bins (e.g. 128)
  T       – time frames (e.g. 256)

Architecture (Figure 1 of the paper):
  Encoder 1  : X → f1          (downsampled in both freq and time by s1)
  Encoder 2  : f1 → f2         (further downsampled by s2)
  VQ1        : f2 → Q1         (coarse codebook, top)
  Decoder 1  : Q1 → fU         (upsample coarse back to f1 resolution)
  Concatenate: [f1, fU] → VQ2 → Q2  (fine codebook, bottom)
  Decoder 2  : [Q1_up, Q2] → X_out  (upsample to original resolution)

Loss (Equation 1):
  L_ae = lambda_x * MSE(X, X_out)
       + lambda_K * MSE(f2, sg[Q1])   <- VQ1 (top)
       + lambda_K * MSE(f1, sg[Q2])   <- VQ2 (bottom)
  where sg[.] is stop-gradient (.detach()), lambda_K = 0.25.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

from .quantizer import VectorQuantizerEMA
from .encoders import EncoderBot, EncoderTop
from .decoders import DecoderBot, DecoderTop


# ---------------------------------------------------------------------------
# Discrete Model
# ---------------------------------------------------------------------------

class VQ_VAE_2Layer(nn.Module):
    """
    Two-layer VQ-VAE for spectrograms.

    The top (coarse) and bottom (fine) quantizers can have different codebook sizes,
    allowing finer control over representation capacity at each level.
    """

    def __init__(
        self,
        num_hiddens: int,
        num_residual_layers: int,
        num_residual_hiddens: int,
        num_embeddings: Union[int, tuple[int, int]],
        embedding_dim: int,
        commitment_cost: float,
        decay: float = 0.0,
        test: bool = False,
    ):
        super().__init__()
        self.test = test

        # Resolve codebook sizes: int -> (top, bot) same; tuple -> (top, bot) explicit
        if isinstance(num_embeddings, int):
            num_embeddings_top = num_embeddings_bot = num_embeddings
        else:
            num_embeddings_top, num_embeddings_bot = num_embeddings

        # Encoders
        self._encoder_bot = EncoderBot(
            1, num_hiddens,
            num_residual_layers,
            num_residual_hiddens,
        )
        self._encoder_top = EncoderTop(
            num_hiddens, num_hiddens,
            num_residual_layers,
            num_residual_hiddens,
        )

        # Projection to embedding space before VQ
        self._pre_vq_conv_top = nn.Conv2d(num_hiddens, embedding_dim, kernel_size=1, stride=1)
        self._pre_vq_conv_bot = nn.Conv2d(
            num_hiddens + embedding_dim, embedding_dim, kernel_size=1, stride=1
        )

        # Vector quantizers (different codebook sizes allowed)
        self._vq_top = VectorQuantizerEMA(
            num_embeddings_top, embedding_dim, commitment_cost, decay
        )
        self._vq_bot = VectorQuantizerEMA(
            num_embeddings_bot, embedding_dim, commitment_cost, decay
        )

        # Decoders
        self._decoder_top = DecoderTop(
            embedding_dim, num_hiddens,
            num_residual_layers,
            num_residual_hiddens,
        )
        self._decoder_bot = DecoderBot(
            embedding_dim * 2, num_hiddens,
            num_residual_layers,
            num_residual_hiddens,
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        # Encode: bottom (high-res) and top (low-res)
        enc_bot = self._encoder_bot(x)
        enc_top = self._encoder_top(enc_bot)

        # Project and quantize top (coarse)
        z_top = self._pre_vq_conv_top(enc_top)
        loss_top, quantized_top, perplexity_top, _ = self._vq_top(z_top)

        # Decode top and concatenate with enc_bot for bottom quantizer input
        decoded_top = self._decoder_top(quantized_top)
        feat_bot = torch.cat([enc_bot, decoded_top], dim=1)

        # Project and quantize bottom (fine)
        z_bot = self._pre_vq_conv_bot(feat_bot)
        loss_bot, quantized_bot, perplexity_bot, _ = self._vq_bot(z_bot)

        # Align top quantized to bottom spatial size and decode jointly
        quantized_top_upsampled = F.interpolate(
            quantized_top, size=quantized_bot.shape[-2:],
            mode="bilinear", align_corners=False
        )
        quant_joined = torch.cat([quantized_top_upsampled, quantized_bot], dim=1)
        recon = self._decoder_bot(quant_joined)

        return (
            loss_bot, loss_top, recon,
            quantized_top, quantized_bot,
            perplexity_top, perplexity_bot,
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode and quantize input spectrogram (for AudDSR inference/training).

        Args:
            x: (B, 1, n_mels, T) Mel spectrogram

        Returns:
            q_bot: (B, emb_dim, H_q, W_q) bottom (fine) quantized features
            q_top: (B, emb_dim, H_q_top, W_q_top) top (coarse) quantized features
        """
        q_bot, q_top, _, _ = self.encode_with_prequant(x)
        return q_bot, q_top

    def encode_with_prequant(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode with pre-quantize features (for AudDSR anomaly generation).

        Returns:
            q_bot, q_top, z_bot, z_top (z_* are pre-quantize continuous features)
        """
        enc_bot = self._encoder_bot(x)
        enc_top = self._encoder_top(enc_bot)
        z_top = self._pre_vq_conv_top(enc_top)
        _, quantized_top, _, _ = self._vq_top(z_top)
        decoded_top = self._decoder_top(quantized_top)
        feat_bot = torch.cat([enc_bot, decoded_top], dim=1)
        z_bot = self._pre_vq_conv_bot(feat_bot)
        _, quantized_bot, _, _ = self._vq_bot(z_bot)
        return quantized_bot, quantized_top, z_bot, z_top

    def decode_general(
        self, q_bot: torch.Tensor, q_top: torch.Tensor
    ) -> torch.Tensor:
        """
        General object decoder: reconstruct spectrogram from quantized features.
        Preserves anomalies (for AudDSR X_G path). Frozen during stage 2.

        Args:
            q_bot: (B, emb_dim, H_q, W_q)
            q_top: (B, emb_dim, H_q_top, W_q_top)

        Returns:
            X_G: (B, 1, n_mels, T) reconstructed spectrogram
        """
        quantized_top_upsampled = F.interpolate(
            q_top, size=q_bot.shape[-2:], mode="bilinear", align_corners=False
        )
        quant_joined = torch.cat([quantized_top_upsampled, q_bot], dim=1)
        return self._decoder_bot(quant_joined)

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Same codebook size for both (backward compatible)
    model_same = VQ_VAE_2Layer(
        num_hiddens=128,
        num_residual_layers=2,
        num_residual_hiddens=64,
        num_embeddings=4096,
        embedding_dim=128,
        commitment_cost=0.25,
        decay=0.99,
    ).to(device)

    # Different codebook sizes: top (coarse) smaller, bottom (fine) larger
    model_diff = VQ_VAE_2Layer(
        num_hiddens=128,
        num_residual_layers=2,
        num_residual_hiddens=64,
        num_embeddings=(1024, 4096),  # top=1024, bot=4096
        embedding_dim=128,
        commitment_cost=0.25,
        decay=0.99,
    ).to(device)

    x = torch.randn(1, 1, 128, 320, device=device)

    for name, m in [("same", model_same), ("diff", model_diff)]:
        loss_b, loss_t, recon, q_t, q_b, perp_t, perp_b = m(x)
        n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"[{name}] params={n_params:,}  loss_b={loss_b.item():.4f}  loss_t={loss_t.item():.4f}  recon={recon.shape}")
        print(f"Quantized top: {q_t.shape}")
        print(f"Quantized bot: {q_b.shape}")
