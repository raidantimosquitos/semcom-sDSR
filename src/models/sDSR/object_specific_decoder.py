"""
Object Specific Decoder for sDSR.

Reconstructs anomaly-free spectrogram X_S from (q_coarse, q_fine).
Integrates subspace restriction (F̃ + frozen VQ) then spectrogram reconstruction.
Trained with L2 loss during stage 2.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..vq_vae.res_blocks_2d import ResidualStack
from .subspace_restriction import SubspaceRestrictionModule


class ObjectSpecificDecoder(nn.Module):
    """
    Object Specific Decoder: Q or Q_A -> (optional) subspace restriction -> quantize -> spectrogram -> X_S.

    Input: Q_coarse / Q_coarse_a (B, emb_dim, H_q, W_q), Q_fine / Q_fine_a (B, emb_dim, H_q, W_q)
    Output: X_S (B, 1, n_mels, T); when return_aux=True also (recon_feat_coarse, recon_feat_fine) for L_sub.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_channels: int,
        num_residual_layers: int,
        use_subspace_restriction: bool = True,
    ) -> None:
        super().__init__()
        self._embedding_dim = embedding_dim
        self._use_subspace_restriction = use_subspace_restriction

        if use_subspace_restriction:
            self._subspace_coarse = SubspaceRestrictionModule(embedding_size=embedding_dim)
            self._subspace_fine = SubspaceRestrictionModule(embedding_size=embedding_dim)
        else:
            self._subspace_coarse = None
            self._subspace_fine = None

        self.spectrogram_reconstruction_network = SpectrogramReconstructionNetwork(
            in_channels=2 * embedding_dim,
            hidden_channels=hidden_channels,
            num_residual_layers=num_residual_layers,
        )

    def forward(
        self,
        q_coarse: torch.Tensor,
        q_fine: torch.Tensor,
        vq_coarse: nn.Module | None = None,
        vq_fine: nn.Module | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        """
        Decode quantized features to anomaly-free spectrogram X_S.

        Args:
            q_coarse: (B, emb_dim, H_q_coarse, W_q_coarse)
            q_fine: (B, emb_dim, H_q, W_q)
            vq_coarse: frozen VQ module for coarse level (required when use_subspace_restriction)
            vq_fine: frozen VQ module for fine level (required when use_subspace_restriction)
            return_aux: if True, return (x_s, aux_dict) with recon_feat_coarse/fine for L_sub

        Returns:
            x_s: (B, 1, n_mels, T)
            When return_aux=True: (x_s, {"recon_feat_coarse": ..., "recon_feat_fine": ...}) or (x_s, {})
        """
        use_sub = (
            self._use_subspace_restriction
            and self._subspace_coarse is not None
            and self._subspace_fine is not None
            and vq_coarse is not None
            and vq_fine is not None
        )
        aux: dict[str, Any] = {}

        if use_sub:
            assert self._subspace_coarse is not None and self._subspace_fine is not None
            recon_feat_coarse, q_coarse_r, _ = self._subspace_coarse(q_coarse, vq_coarse)
            recon_feat_fine, q_fine_r, _ = self._subspace_fine(q_fine, vq_fine)
            x_s = self.spectrogram_reconstruction_network(q_coarse_r, q_fine_r)
            if return_aux:
                aux["recon_feat_coarse"] = recon_feat_coarse
                aux["recon_feat_fine"] = recon_feat_fine
        else:
            x_s = self.spectrogram_reconstruction_network(q_coarse, q_fine)

        if return_aux:
            return x_s, aux
        return x_s


class SpectrogramReconstructionNetwork(nn.Module):
    """
    UNet-style decoder: [Q_coarse_upsampled, Q_fine] -> spectrogram (1 ch).

    No re-downsampling: two encoder blocks at feature resolution (H_fine, W_fine),
    then bottleneck (ResidualStack), then symmetric 4x upsample: two 2x2 transposed
    convs (32x80 -> 64x160 -> 128x320), with skip connections from b1 and b2.

    Input: (B, 2 * embedding_dim, H_q, W_q) after concat; (H_q, W_q) = (n_mels/4, T/4) = (32, 80).
    Output: (B, 1, n_mels, T)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_residual_layers: int,
    ) -> None:
        super().__init__()
        norm_layer = nn.InstanceNorm2d
        # Encoder at feature resolution (no downsampling)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            norm_layer(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            norm_layer(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            norm_layer(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            norm_layer(hidden_channels),
            nn.ReLU(inplace=True),
        )
        # Bottleneck at feature resolution (middle channels = hidden_channels // 2)
        self._bottleneck = ResidualStack(hidden_channels, hidden_channels, num_residual_layers)
        # Decoder: symmetric 4x upsample (2x then 2x), with skips from b2 and b1
        self._conv_trans1 = nn.ConvTranspose2d(
            hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1
        )
        self._conv_after_skip2 = nn.Conv2d(
            hidden_channels * 2, hidden_channels, kernel_size=3, stride=1, padding=1
        )
        self._conv_trans2 = nn.ConvTranspose2d(
            hidden_channels, hidden_channels // 2, kernel_size=4, stride=2, padding=1
        )
        self._conv_after_skip1 = nn.Conv2d(
            hidden_channels // 2 + hidden_channels, hidden_channels // 2, kernel_size=3, stride=1, padding=1
        )
        self._conv_out = nn.Conv2d(hidden_channels // 2, 1, kernel_size=3, stride=1, padding=1)

    def forward(
        self,
        q_coarse: torch.Tensor,
        q_fine: torch.Tensor,
    ) -> torch.Tensor:
        q_coarse_up = F.interpolate(
            q_coarse, size=q_fine.shape[-2:], mode="bilinear", align_corners=False
        )
        x = torch.cat([q_coarse_up, q_fine], dim=1)
        b1 = self.block1(x)
        b2 = self.block2(b1)
        x = self._bottleneck(b2)
        # First up: 2x both dims, then concat skip from b2
        x = self._conv_trans1(x)
        x = F.relu(x)
        b2_up = F.interpolate(b2, scale_factor=2, mode="bilinear", align_corners=False)
        x = torch.cat([x, b2_up], dim=1)
        x = self._conv_after_skip2(x)
        x = F.relu(x)
        # Second up: 2x both dims to full resolution, then concat skip from b1 (4x from feature res)
        x = self._conv_trans2(x)
        x = F.relu(x)
        b1_up = F.interpolate(b1, scale_factor=4, mode="bilinear", align_corners=False)
        x = torch.cat([x, b1_up], dim=1)
        x = self._conv_after_skip1(x)
        x = F.relu(x)
        return self._conv_out(x)
