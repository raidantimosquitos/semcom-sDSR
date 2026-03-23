"""
Anomaly map generation for sDSR training.

Strategies (same interface: __call__(batch_size, device) -> (B, 1, H, W)):
1. PerlinNoiseStrategy: threshold/binarize Perlin noise (DSR-style)
2. AudioSpecificStrategy: random frequency band + 3–7 time segments within that band
"""

from __future__ import annotations

import random
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import rotate as ndimage_rotate

from .perlin import rand_perlin_2d_np

# Minimum mask extent so regions survive max-pool to coarse latent (8×8 cells)
MIN_FREQ_BINS = 4
MIN_TIME_FRAMES = 4


class PerlinNoiseStrategy:
    """
    Generate anomaly map by thresholding Perlin noise (sDSR / MVTec style).
    Produces blob-like anomaly regions. Optionally rotates the noise before
    thresholding (matching original DSR) for more varied blob orientations.
    """

    def __init__(
        self,
        spectrogram_shape: tuple[int, int],
        threshold: float = 0.5,
        perlin_scale_range: tuple[int, int] = (1, 5),
        rotate: bool = True,
        rotation_range: tuple[float, float] = (-120.0, 120.0),
    ) -> None:
        """
        Args:
            spectrogram_shape: (n_mels, T) spectrogram spatial dimensions
            threshold: binarization threshold
            perlin_scale_range: (min_exp, max_exp) for res = 2^randint(min_exp, max_exp)
            rotate: if True, apply random 2D rotation to noise before thresholding
            rotation_range: (min_deg, max_deg) for rotation angle in degrees
        """
        self.spectrogram_shape = spectrogram_shape
        self.threshold = threshold
        n_mels, T = spectrogram_shape
        # Cap max exponent so Perlin wavelength >= MIN_* (blobs survive max-pool)
        max_exp_freq = int(np.log2(max(1, n_mels // MIN_FREQ_BINS)))
        max_exp_time = int(np.log2(max(1, T // MIN_TIME_FRAMES)))
        effective_max = min(max_exp_freq, max_exp_time, perlin_scale_range[1])
        effective_min = min(perlin_scale_range[0], effective_max)
        self.perlin_scale_range = (effective_min, effective_max)
        self.rotate = rotate
        self.rotation_range = rotation_range

    def __call__(
        self,
        batch_size: int,
        device: torch.device | str,
    ) -> torch.Tensor:
        """
        Generate anomaly map M. Each mask uses Perlin noise (optionally rotated)
        then binarized at spectrogram shape.

        Returns:
            M: (B, 1, n_mels, T) binary mask
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
        return M


class AudioSpecificStrategy:
    """
    General spectrogram-space mask: pick a random frequency band (random bandwidth),
    then set mask to 1 on several (3–7) disjoint time segments within that band.
    """

    def __init__(
        self,
        spectrogram_shape: tuple[int, int],
        n_mels: int,
        T: int,
        min_segments: int = 3,
        max_segments: int = 7,
    ) -> None:
        self.spectrogram_shape = spectrogram_shape
        self.n_mels = n_mels
        self.T = T
        self.min_segments = min_segments
        self.max_segments = max_segments

    def _single_mask_numpy(self) -> np.ndarray:
        n_mels, T = self.n_mels, self.T
        M = np.zeros((n_mels, T), dtype=np.float32)
        bandwidth = self._sample_bandwidth(n_mels)
        f_low = random.randint(0, max(0, n_mels - bandwidth))
        f_high = f_low + bandwidth
        n_seg = random.randint(self.min_segments, self.max_segments)
        for _ in range(n_seg):
            if T <= MIN_TIME_FRAMES:
                t_start, t_end = 0, T
            else:
                seg_len = self._sample_segment_len(T)
                max_start = max(0, T - seg_len)
                t_start = random.randint(0, max_start)
                t_end = t_start + seg_len
            M[f_low:f_high, t_start:t_end] = 1.0
        return M

    def single_mask(self, device: torch.device | str) -> torch.Tensor:
        """One (1, 1, n_mels, T) mask."""
        arr = self._single_mask_numpy()
        return torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

    def _sample_bandwidth(self, n_mels: int) -> int:
        u = random.random()
        if n_mels <= MIN_FREQ_BINS:
            return n_mels
        
        # Renormalize if upper tiers are empty
        if n_mels <= 30:
            return random.randint(MIN_FREQ_BINS, n_mels)
        if n_mels <= 70:
            # only narrow vs medium
            if u < 0.7 / (0.7 + 0.2):
                return random.randint(MIN_FREQ_BINS, min(30, n_mels))
            return random.randint(31, n_mels)
        if u < 0.7:
            return random.randint(MIN_FREQ_BINS, 30)
        if u < 0.9:
            return random.randint(31, min(70, n_mels))
        # wide
        lo = max(71, MIN_FREQ_BINS)
        hi = n_mels
        if lo > hi:
            return random.randint(31, min(70, n_mels))
        return random.randint(lo, hi)
        
    def _sample_segment_len(self, T: int) -> int:
        if T <= 1:
            return T
        if T < 10:
            return random.randint(MIN_TIME_FRAMES, T)
        u = random.random()
        def clamp_len(x: int) -> int:
            return max(MIN_TIME_FRAMES, min(x, T))
        if T <= 40:
            return clamp_len(random.randint(10, T))
        if T <= 120:
            # short vs medium only; split 60:30 within [0,1)
            p_short = 0.6 / (0.6 + 0.3)
            if u < p_short:
                return clamp_len(random.randint(10, 40))
            return clamp_len(random.randint(41, T))
        # T >= 121
        if u < 0.6:
            return clamp_len(random.randint(10, 40))
        if u < 0.9:
            return clamp_len(random.randint(41, 120))
        return clamp_len(random.randint(121, T))


    def __call__(
        self,
        batch_size: int,
        device: torch.device | str,
    ) -> torch.Tensor:
        masks = []
        for _ in range(batch_size):
            arr = self._single_mask_numpy()
            mask = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            masks.append(mask)
        return torch.cat(masks, dim=0).to(device)


class AnomalyMapGenerator:
    """
    Generate anomaly map M for training using one of several strategies.
    When force_anomaly=False, each sample gets an independent draw: with
    probability zero_mask_prob a zero mask (no anomaly), else a generated mask.
    """

    def __init__(
        self,
        strategy: Literal["perlin", "audio_specific", "both"],
        spectrogram_shape: tuple[int, int],
        n_mels: int | None = None,
        T: int | None = None,
        zero_mask_prob: float = 0.5,
    ) -> None:
        """
        Args:
            strategy: 'perlin', 'audio_specific', or 'both' (Perlin vs audio 50/50)
            spectrogram_shape: (n_mels, T)
            n_mels, T: required for audio_specific and both
            zero_mask_prob: per-sample probability of returning a zero mask (no anomaly)
        """
        self.strategy_name = strategy
        self.spectrogram_shape = spectrogram_shape
        self.zero_mask_prob = zero_mask_prob
        self.perlin = (
            PerlinNoiseStrategy(spectrogram_shape)
            if strategy in ("perlin", "both")
            else None
        )
        self.audio_specific = (
            AudioSpecificStrategy(spectrogram_shape, n_mels, T)
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
        if self.strategy_name == "both":
            if random.random() < 0.5:
                assert self.perlin is not None
                return self.perlin(1, device)
            assert self.audio_specific is not None
            return self.audio_specific(1, device)
        raise RuntimeError(f"Unknown strategy: {self.strategy_name}")

    def generate_for_training_sample(
        self,
        device: torch.device | str,
        force_anomaly: bool = True,
    ) -> torch.Tensor:
        """Generate one mask for a single training sample."""
        if force_anomaly:
            if self.strategy_name == "perlin":
                assert self.perlin is not None
                return self.perlin(1, device)
            if self.strategy_name == "audio_specific":
                assert self.audio_specific is not None
                return self.audio_specific.single_mask(device)
            if self.strategy_name == "both":
                if random.random() < 0.5:
                    assert self.perlin is not None
                    return self.perlin(1, device)
                assert self.audio_specific is not None
                return self.audio_specific.single_mask(device)
            raise RuntimeError(f"Unknown strategy: {self.strategy_name}")
        return self.generate(1, device, force_anomaly=False)

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
            M: (B, 1, n_mels, T) binary mask
        """
        if force_anomaly:
            if self.strategy_name == "perlin":
                assert self.perlin is not None
                return self.perlin(batch_size, device)
            if self.strategy_name == "audio_specific":
                assert self.audio_specific is not None
                return self.audio_specific(batch_size, device)
            if self.strategy_name == "both":
                if random.random() < 0.5:
                    assert self.perlin is not None
                    return self.perlin(batch_size, device)
                assert self.audio_specific is not None
                return self.audio_specific(batch_size, device)
            raise RuntimeError(f"Unknown strategy: {self.strategy_name}")

        masks = []
        for _ in range(batch_size):
            if random.random() < self.zero_mask_prob:
                masks.append(
                    torch.zeros(
                        1, 1, *self.spectrogram_shape,
                        device=device, dtype=torch.float32
                    )
                )
            else:
                masks.append(self._generate_one(device))
        return torch.cat(masks, dim=0)
