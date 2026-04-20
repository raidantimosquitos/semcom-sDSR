"""
Object Specific Decoder for sDSR.

Reconstructs anomaly-free spectrogram X_S from (q_coarse, q_fine).
Integrates subspace restriction (F̃ + frozen VQ) then spectrogram reconstruction.
Trained with L2 loss during stage 2.
"""

from __future__ import annotations

from typing import Any, Tuple

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
        embedding_dim: Tuple[int, int],
        hidden_channels: Tuple[int, int],
        num_residual_layers: int,
        use_subspace_restriction: bool = True,
    ) -> None:
        super().__init__()
        self._embedding_dim_coarse, self._embedding_dim_fine = embedding_dim
        self._hidden_channels_coarse, self._hidden_channels_fine = hidden_channels
        self._use_subspace_restriction = use_subspace_restriction

        if use_subspace_restriction:
            self._subspace_coarse = SubspaceRestrictionModule(embedding_size=self._embedding_dim_coarse)
            self._subspace_fine = SubspaceRestrictionModule(embedding_size=self._embedding_dim_fine)
        else:
            self._subspace_coarse = None
            self._subspace_fine = None

        self.spectrogram_reconstruction_network = SpectrogramReconstructionNetwork(
            in_channels=self._embedding_dim_coarse + self._embedding_dim_fine,
            hidden_channels=self._hidden_channels_coarse + self._hidden_channels_fine,
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
    Feedforward encoder-decoder: [Q_coarse_upsampled, Q_fine] -> spectrogram (1 ch).

    Encoder: two conv blocks with stride-2 conv downsampling (2x each) -> 4x down total.
    Bottleneck compresses to ``in_channels`` channels.
    Decoder: two ConvTranspose2d stages (no skip connections -- forces an
    information bottleneck so the network cannot leak anomaly detail),
    followed by a ResidualStack and final 4x bilinear upsample to spectrogram.

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
        self._in_channels = in_channels
        norm_layer = nn.InstanceNorm2d
        half = max(1, hidden_channels // 2)

        self._block1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            norm_layer(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, padding=1),
            norm_layer(in_channels * 2),
            nn.ReLU(inplace=True),
        )
        self._mp1 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3, stride=2, padding=1),
            norm_layer(in_channels * 2),
            nn.ReLU(inplace=True),
        )

        self._block2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3, padding=1),
            norm_layer(in_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 2, in_channels * 4, kernel_size=3, padding=1),
            norm_layer(in_channels * 4),
            nn.ReLU(inplace=True),
        )
        self._mp2 = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size=3, stride=2, padding=1),
            norm_layer(in_channels * 4),
            nn.ReLU(inplace=True),
        )

        self._bottleneck_conv = nn.Conv2d(in_channels * 4, in_channels, kernel_size=1)

        self._upblock1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self._upblock2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self._conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            norm_layer(hidden_channels),
            nn.ReLU(inplace=True),
        )

        self._residual_stack = ResidualStack(
            hidden_channels, hidden_channels, num_residual_layers, half
        )
        self._conv_trans_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
        )
        self._conv_trans_2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1),
        )

    def forward(
        self,
        q_coarse: torch.Tensor,
        q_fine: torch.Tensor,
    ) -> torch.Tensor:
        q_coarse_up = F.interpolate(
            q_coarse, size=q_fine.shape[-2:], mode="bilinear", align_corners=False
        )
        x = torch.cat([q_coarse_up, q_fine], dim=1)

        x = self._block1(x)
        x = self._mp1(x)
        x = self._block2(x)
        x = self._mp2(x)
        x = self._bottleneck_conv(x)

        x = self._upblock1(x)
        x = F.relu(x)
        x = self._upblock2(x)
        x = F.relu(x)
        x = self._conv_1(x)

        x = self._residual_stack(x)

        x = self._conv_trans_1(x)
        x = F.relu(x)

        return self._conv_trans_2(x)

