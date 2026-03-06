"""
Object Specific Decoder for sDSR.

Reconstructs anomaly-free spectrogram X_S from (q_top, q_bot).
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

    Input: Q_top / Q_top_a (B, emb_dim, H_q, W_q), Q_bot / Q_bot_a (B, emb_dim, H_q, W_q)
    Output: X_S (B, 1, n_mels, T); when return_aux=True also (recon_feat_top, recon_feat_bot) for L_sub.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_channels: int,
        num_residual_layers: int,
        num_residual_hiddens: int,
        use_subspace_restriction: bool = True,
    ) -> None:
        super().__init__()
        self._embedding_dim = embedding_dim
        self._use_subspace_restriction = use_subspace_restriction

        if use_subspace_restriction:
            self._subspace_top = SubspaceRestrictionModule(embedding_size=embedding_dim)
            self._subspace_bot = SubspaceRestrictionModule(embedding_size=embedding_dim)
        else:
            self._subspace_top = None
            self._subspace_bot = None

        self.spectrogram_reconstruction_network = SpectrogramReconstructionNetwork(
            in_channels=2 * embedding_dim,
            hidden_channels=hidden_channels,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
        )

    def forward(
        self,
        q_top: torch.Tensor,
        q_bot: torch.Tensor,
        vq_top: nn.Module | None = None,
        vq_bot: nn.Module | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        """
        Decode quantized features to anomaly-free spectrogram X_S.

        Args:
            q_top: (B, emb_dim, H_q_top, W_q_top)
            q_bot: (B, emb_dim, H_q, W_q)
            vq_top: frozen VQ module for top level (required when use_subspace_restriction)
            vq_bot: frozen VQ module for bot level (required when use_subspace_restriction)
            return_aux: if True, return (x_s, aux_dict) with recon_feat_top/bot for L_sub

        Returns:
            x_s: (B, 1, n_mels, T)
            When return_aux=True: (x_s, {"recon_feat_top": ..., "recon_feat_bot": ...}) or (x_s, {})
        """
        use_sub = (
            self._use_subspace_restriction
            and self._subspace_top is not None
            and self._subspace_bot is not None
            and vq_top is not None
            and vq_bot is not None
        )
        aux: dict[str, Any] = {}

        if use_sub:
            assert self._subspace_top is not None and self._subspace_bot is not None
            recon_feat_top, q_top_r, _ = self._subspace_top(q_top, vq_top)
            recon_feat_bot, q_bot_r, _ = self._subspace_bot(q_bot, vq_bot)
            x_s = self.spectrogram_reconstruction_network(q_top_r, q_bot_r)
            if return_aux:
                aux["recon_feat_top"] = recon_feat_top
                aux["recon_feat_bot"] = recon_feat_bot
        else:
            x_s = self.spectrogram_reconstruction_network(q_top, q_bot)

        if return_aux:
            return x_s, aux
        return x_s


class SpectrogramReconstructionNetwork(nn.Module):
    """
    UNet-style decoder: [Q_top_upsampled, Q_bot] -> spectrogram (1 ch).

    No re-downsampling: two encoder blocks at feature resolution (H_bot, W_bot),
    then bottleneck (ResidualStack), then asymmetric upsample: first 2x both dims,
    then 1x freq 2x time (total 2x freq, 4x time to full resolution), with skip
    connections from b1 and b2.

    Input: (B, 2 * embedding_dim, H_q, W_q) after concat; (H_q, W_q) = (n_mels/2, T/4).
    Output: (B, 1, n_mels, T)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_residual_layers: int,
        num_residual_hiddens: int,
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
        # Bottleneck at feature resolution (no 1024->64 squeeze; use num_hiddens)
        self._bottleneck = ResidualStack(
            hidden_channels,
            hidden_channels,
            num_residual_layers,
            num_residual_hiddens,
        )
        # Decoder: up 2x both + skip(b2), then 1x freq 2x time + skip(b1) (asymmetric: 2x freq, 4x time total)
        self._conv_trans1 = nn.ConvTranspose2d(
            hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1
        )
        self._conv_after_skip2 = nn.Conv2d(
            hidden_channels * 2, hidden_channels, kernel_size=3, stride=1, padding=1
        )
        self._conv_trans2 = nn.ConvTranspose2d(
            hidden_channels, hidden_channels // 2, kernel_size=(3, 4), stride=(1, 2), padding=(1, 1)
        )
        self._conv_after_skip1 = nn.Conv2d(
            hidden_channels // 2 + hidden_channels, hidden_channels // 2, kernel_size=3, stride=1, padding=1
        )
        self._conv_out = nn.Conv2d(hidden_channels // 2, 1, kernel_size=3, stride=1, padding=1)

    def forward(
        self,
        q_top: torch.Tensor,
        q_bot: torch.Tensor,
    ) -> torch.Tensor:
        q_top_up = F.interpolate(
            q_top, size=q_bot.shape[-2:], mode="bilinear", align_corners=False
        )
        x = torch.cat([q_top_up, q_bot], dim=1)
        b1 = self.block1(x)
        b2 = self.block2(b1)
        x = self._bottleneck(b2)
        # First up: 2x, then concat skip from b2
        x = self._conv_trans1(x)
        x = F.relu(x)
        b2_up = F.interpolate(b2, scale_factor=2, mode="bilinear", align_corners=False)
        x = torch.cat([x, b2_up], dim=1)
        x = self._conv_after_skip2(x)
        x = F.relu(x)
        # Second up: 1x freq, 2x time to full resolution, then concat skip from b1
        x = self._conv_trans2(x)
        x = F.relu(x)
        b1_up = F.interpolate(b1, scale_factor=(2, 4), mode="bilinear", align_corners=False)
        x = torch.cat([x, b1_up], dim=1)
        x = self._conv_after_skip1(x)
        x = F.relu(x)
        return self._conv_out(x)
