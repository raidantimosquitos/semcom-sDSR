"""
Anomaly map generation for sDSR training.

Two strategies (same interface as original DSR mask generation):
1. PerlinNoiseStrategy: threshold/binarize Perlin noise (DSR-style, from
   VitjanZ/DSR_anomaly_detection perlin.py)
2. AudioSpecificStrategy: choose frequency band + time segments (audio-domain
   analogue; same output format: __call__(batch_size, device) -> (B, 1, H, W))
"""

from __future__ import annotations

import random
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import rotate as ndimage_rotate

from .perlin import rand_perlin_2d_np


class PerlinNoiseStrategy:
    """
    Generate anomaly map by thresholding Perlin noise (sDSR / MVTec style).
    Produces blob-like anomaly regions. Optionally rotates the noise before
    thresholding (matching original DSR) for more varied blob orientations.
    """

    def __init__(
        self,
        spectrogram_shape: tuple[int, int],
        q_shape: tuple[int, int],
        threshold: float = 0.5,
        perlin_scale_range: tuple[int, int] = (0, 6),
        rotate: bool = True,
        rotation_range: tuple[float, float] = (-90.0, 90.0),
    ) -> None:
        """
        Args:
            spectrogram_shape: (n_mels, T) spectrogram spatial dimensions
            q_shape: (H_q, W_q) quantized feature map spatial dimensions
            threshold: binarization threshold
            perlin_scale_range: (min_exp, max_exp) for res = 2^randint(min_exp, max_exp)
            rotate: if True, apply random 2D rotation to noise before thresholding
            rotation_range: (min_deg, max_deg) for rotation angle in degrees
        """
        self.spectrogram_shape = spectrogram_shape
        self.q_shape = q_shape
        self.threshold = threshold
        self.perlin_scale_range = perlin_scale_range
        self.rotate = rotate
        self.rotation_range = rotation_range

    def __call__(
        self,
        batch_size: int,
        device: torch.device | str,
    ) -> torch.Tensor:
        """
        Generate anomaly map M. Each mask uses Perlin noise (optionally rotated)
        then binarized and resized to q_shape.

        Returns:
            M: (B, 1, H_q, W_q) binary mask
        """
        masks = []
        for _ in range(batch_size):
            res_y = 2 ** random.randint(*self.perlin_scale_range)
            res_x = 2 ** random.randint(*self.perlin_scale_range)
            res = (res_y, res_x)
            noise = rand_perlin_2d_np(self.spectrogram_shape, res)
            if self.rotate:
                angle = random.uniform(*self.rotation_range)
                noise = ndimage_rotate(
                    noise, angle, reshape=False, order=1, mode="constant", cval=0
                )
            binary = (noise > self.threshold).astype(np.float32)
            mask = torch.from_numpy(binary).unsqueeze(0).unsqueeze(0)
            masks.append(mask)
        M = torch.cat(masks, dim=0).to(device)
        M = F.interpolate(M, size=self.q_shape, mode="nearest")
        return M


class AudioSpecificStrategy:
    """
    Generate anomaly map for spectrograms (sDSR-defined).

    Choose a frequency band and several time segments; mark their intersection.
    Models anomalies that span full duration or specific frequency bands.
    """

    def __init__(
        self,
        spectrogram_shape: tuple[int, int],
        q_shape: tuple[int, int],
        n_mels: int,
        T: int,
        min_band_fraction: float = 0.1,
        max_band_fraction: float = 0.5,
        min_segments: int = 10,
        max_segments: int = 160,
    ) -> None:
        """
        Args:
            spectrogram_shape: (n_mels, T)
            q_shape: (H_q, W_q) quantized feature spatial dims
            n_mels, T: spectrogram dimensions
            min_band_fraction, max_band_fraction: band width as fraction of n_mels
            min_segments, max_segments: number of time segments to augment
        """
        self.spectrogram_shape = spectrogram_shape
        self.q_shape = q_shape
        self.n_mels = n_mels
        self.T = T
        self.min_band_fraction = min_band_fraction
        self.max_band_fraction = max_band_fraction
        self.min_segments = min_segments
        self.max_segments = max_segments

    def __call__(
        self,
        batch_size: int,
        device: torch.device | str,
    ) -> torch.Tensor:
        """
        Generate anomaly map M.

        Returns:
            M: (B, 1, H_q, W_q) binary mask
        """
        masks = []
        for _ in range(batch_size):
            M = np.zeros(self.spectrogram_shape, dtype=np.float32)
            band_width = int(
                self.n_mels
                * random.uniform(self.min_band_fraction, self.max_band_fraction)
            )
            band_width = max(1, band_width)
            f_low = random.randint(0, self.n_mels - band_width)
            f_high = min(f_low + band_width, self.n_mels)
            n_seg = random.randint(self.min_segments, self.max_segments)
            for _ in range(n_seg):
                seg_len = random.randint(
                    1, max(1, self.T // 4)
                )
                t_start = random.randint(0, max(0, self.T - seg_len))
                t_end = min(t_start + seg_len, self.T)
                M[f_low:f_high, t_start:t_end] = 1.0
            mask = torch.from_numpy(M).unsqueeze(0).unsqueeze(0)
            masks.append(mask)
        M = torch.cat(masks, dim=0).to(device)
        M = F.interpolate(M, size=self.q_shape, mode="nearest")
        return M


class AnomalyMapGenerator:
    """
    Generate anomaly map M for training using one of two strategies.
    When force_anomaly=False, each sample gets an independent draw: with
    probability zero_mask_prob a zero mask (no anomaly), else a generated mask.
    """

    def __init__(
        self,
        strategy: Literal["perlin", "audio_specific", "both"],
        spectrogram_shape: tuple[int, int],
        q_shape: tuple[int, int],
        n_mels: int | None = None,
        T: int | None = None,
        zero_mask_prob: float = 0.5,
    ) -> None:
        """
        Args:
            strategy: 'perlin', 'audio_specific', or 'both'
            spectrogram_shape: (n_mels, T)
            q_shape: (H_q, W_q)
            n_mels, T: required for audio_specific
            zero_mask_prob: per-sample probability of returning a zero mask (no anomaly)
        """
        self.strategy_name = strategy
        self.zero_mask_prob = zero_mask_prob
        self.q_shape = q_shape
        self.perlin = (
            PerlinNoiseStrategy(spectrogram_shape, q_shape)
            if strategy in ("perlin", "both")
            else None
        )
        self.audio_specific = (
            AudioSpecificStrategy(spectrogram_shape, q_shape, n_mels, T)
            if strategy in ("audio_specific", "both") and n_mels is not None and T is not None
            else None
        )

    def _generate_one(self, device: torch.device | str) -> torch.Tensor:
        """Generate a single non-zero mask (1, 1, H, W)."""
        if self.strategy_name == "perlin":
            assert self.perlin is not None
            return self.perlin(1, device)
        if self.strategy_name == "audio_specific":
            assert self.audio_specific is not None
            return self.audio_specific(1, device)
        if random.random() < 0.3:
            assert self.perlin is not None
            return self.perlin(1, device)
        assert self.audio_specific is not None
        return self.audio_specific(1, device)

    def generate(
        self,
        batch_size: int,
        device: torch.device | str,
        force_anomaly: bool = False,
    ) -> torch.Tensor:
        """
        Generate anomaly map M.

        When force_anomaly=False, each sample is decided independently: with
        probability zero_mask_prob the mask is all zeros, else one mask from
        the strategy (per-sample 50% no-anomaly, matching original DSR).

        Args:
            batch_size: number of masks to generate
            device: torch device
            force_anomaly: if True, skip zero_mask_prob and always generate real masks

        Returns:
            M: (B, 1, H_q, W_q) binary mask
        """
        if force_anomaly:
            if self.strategy_name == "perlin":
                assert self.perlin is not None
                return self.perlin(batch_size, device)
            if self.strategy_name == "audio_specific":
                assert self.audio_specific is not None
                return self.audio_specific(batch_size, device)
            if random.random() < 0.3:
                assert self.perlin is not None
                return self.perlin(batch_size, device)
            assert self.audio_specific is not None
            return self.audio_specific(batch_size, device)

        masks = []
        for _ in range(batch_size):
            if random.random() < self.zero_mask_prob:
                masks.append(
                    torch.zeros(
                        1, 1, self.q_shape[0], self.q_shape[1], device=device
                    )
                )
            else:
                masks.append(self._generate_one(device))
        return torch.cat(masks, dim=0)
