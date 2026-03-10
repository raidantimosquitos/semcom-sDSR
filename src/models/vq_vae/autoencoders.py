"""
sDSR - Spectrogram Dual Subspace Reconstruction

Discrete Autoencoder (VQ-VAE-2 style) for spectrogram anomaly detection.

Input spectrograms are 2-D: (B, C, n_mels, T)
  B       – batch size
  C       – channels (1 for mono mel-spectrogram)
  n_mels  – frequency bins (e.g. 128)
  T       – time frames (e.g. 256)

Architecture (Figure 1 of the paper), symmetric 4x4 down/up:
  Encoder fine   : X -> f_fine        (4x4 down, 32x80 for 128x320, hidden_channels)
  Encoder coarse : f_fine -> f_coarse (4x4 down, 8x20, 4*hidden_channels)
  VQ coarse      : f_coarse -> Q_coarse (coarse codebook)
  Decoder coarse : Q_coarse -> f_coarse_up (4x upsample to 32x80, 4*hidden_channels)
  Concatenate    : [f_fine, f_coarse_up] -> VQ_fine -> Q_fine (fine codebook; 5*hidden_channels -> emb)
  Decoder fine   : [Q_coarse_up, Q_fine] -> X_out (4x symmetric upsample to original resolution)

Loss (Equation 1):
  L_ae = lambda_x * MSE(X, X_out)
       + lambda_K * MSE(f_top, sg[Q_top])   <- VQ_top (top)
       + lambda_K * MSE(f_bot, sg[Q_bot])   <- VQ_bot (bottom)
  where sg[.] is stop-gradient (.detach()), lambda_K = 0.25.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

from .quantizer import VectorQuantizerEMA
from .encoders import EncoderFine, EncoderCoarse
from .decoders import DecoderFine, DecoderCoarse


# ---------------------------------------------------------------------------
# Discrete Model
# ---------------------------------------------------------------------------

class VQ_VAE_2Layer(nn.Module):
    """
    Two-layer VQ-VAE for spectrograms.

    The coarse and fine quantizers can have different codebook sizes,
    allowing finer control over representation capacity at each level.
    """

    def __init__(
        self,
        hidden_channels: int,
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
        self.embedding_dim = embedding_dim
        self.hidden_channels = hidden_channels
        self.num_residual_layers = num_residual_layers
        self.num_residual_hiddens = num_residual_hiddens

        # Resolve codebook sizes: int -> (coarse, fine) same; tuple -> (coarse, fine) explicit
        if isinstance(num_embeddings, int):
            num_embeddings_coarse = num_embeddings_fine = num_embeddings
            self.num_embeddings_fine = num_embeddings_fine
            self.num_embeddings_coarse = num_embeddings_coarse
        else:
            num_embeddings_coarse, num_embeddings_fine = num_embeddings
            self.num_embeddings_fine = num_embeddings_fine
            self.num_embeddings_coarse = num_embeddings_coarse

        # Encoders
        self._encoder_fine = EncoderFine(
            1, hidden_channels,
            num_residual_layers,
            num_residual_hiddens,
        )
        self._encoder_coarse = EncoderCoarse(
            hidden_channels, hidden_channels,
            num_residual_layers,
            num_residual_hiddens,
        )

        # Projection to embedding space before VQ (f_coarse has 4*hidden, feat_fine = f_fine + decoded_coarse = 5*hidden)
        self._pre_vq_conv_coarse = nn.Conv2d(hidden_channels * 4, embedding_dim, kernel_size=1, stride=1)
        self._pre_vq_conv_fine = nn.Conv2d(
            hidden_channels * 5, embedding_dim, kernel_size=1, stride=1
        )

        # Vector quantizers (different codebook sizes allowed)
        self._vq_coarse = VectorQuantizerEMA(
            num_embeddings_coarse, embedding_dim, commitment_cost, decay
        )
        self._vq_fine = VectorQuantizerEMA(
            num_embeddings_fine, embedding_dim, commitment_cost, decay
        )

        # Decoders
        self._decoder_coarse = DecoderCoarse(
            embedding_dim, hidden_channels,
            num_residual_layers,
            num_residual_hiddens,
        )
        self._decoder_fine = DecoderFine(
            embedding_dim * 2, hidden_channels,
            num_residual_layers,
            num_residual_hiddens,
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

        # Decode coarse and concatenate with enc_fine for fine quantizer input
        decoded_coarse = self._decoder_coarse(quantized_coarse)
        feat_fine = torch.cat([f_fine, decoded_coarse], dim=1)

        # Project and quantize fine
        z_fine = self._pre_vq_conv_fine(feat_fine)
        loss_fine, quantized_fine, perplexity_fine, _ = self._vq_fine(z_fine)

        # Align coarse quantized to fine spatial size and decode jointly
        quantized_coarse_upsampled = F.interpolate(
            quantized_coarse, size=quantized_fine.shape[-2:],
            mode="bilinear", align_corners=False
        )
        quant_joined = torch.cat([quantized_coarse_upsampled, quantized_fine], dim=1)
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
            x: (B, 1, n_mels, T) Mel spectrogram

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
        emb_dim = self._vq_coarse._embedding_dim
        q_coarse = self._vq_coarse._embedding(indices_coarse.flatten())
        q_coarse = q_coarse.view(
            indices_coarse.shape[0], indices_coarse.shape[1], indices_coarse.shape[2], emb_dim
        ).permute(0, 3, 1, 2)
        q_fine = self._vq_fine._embedding(indices_fine.flatten())
        q_fine = q_fine.view(
            indices_fine.shape[0], indices_fine.shape[1], indices_fine.shape[2], emb_dim
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
        quantized_coarse_upsampled = F.interpolate(
            q_coarse, size=q_fine.shape[-2:], mode="bilinear", align_corners=False
        )
        quant_joined = torch.cat([quantized_coarse_upsampled, q_fine], dim=1)
        return self._decoder_fine(quant_joined)

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Same codebook size for both (backward compatible)
    model_same = VQ_VAE_2Layer(
        hidden_channels=128,
        num_residual_layers=2,
        num_residual_hiddens=64,
        num_embeddings=4096,
        embedding_dim=128,
        commitment_cost=0.25,
        decay=0.99,
    ).to(device)

    # Different codebook sizes: top (coarse) smaller, bottom (fine) larger
    model_diff = VQ_VAE_2Layer(
        hidden_channels=128,
        num_residual_layers=2,
        num_residual_hiddens=64,
        num_embeddings=(1024, 4096),  # top=1024, bot=4096
        embedding_dim=128,
        commitment_cost=0.25,
        decay=0.99,
    ).to(device)

    x = torch.randn(1, 1, 128, 320, device=device)

    for name, m in [("same", model_same), ("diff", model_diff)]:
        loss_fine, loss_coarse, recon, q_coarse, q_fine, perp_coarse, perp_fine = m(x)
        n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"[{name}] params={n_params:,}  loss_fine={loss_fine.item():.4f}  loss_coarse={loss_coarse.item():.4f}  recon={recon.shape}")
        print(f"Quantized coarse: {q_coarse.shape}")
        print(f"Quantized fine: {q_fine.shape}")
