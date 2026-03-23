"""
Anomaly map generation for sDSR training.

Strategies (same interface: __call__(batch_size, device) -> (B, 1, H, W)):
1. PerlinNoiseStrategy: binarized Perlin noise on the mel–time grid (ICASSP 2024
   AudDSR Sec. 3.3), with axis-aligned variants that resemble band-like or
   duration-spanning defects—without rotating noise as in vision DSR.
2. AudioSpecificStrategy: one random frequency band + several disjoint time
   segments within that band (same section).

Machine-type / ID agnostic: all choices are uniform (or Perlin-scale–only)
functions of (n_mels, T), suitable for unsupervised training on the full
DCASE-style mixture.
"""

from __future__ import annotations

import random
from typing import Literal

import numpy as np
import torch

from .perlin import rand_perlin_2d_np

# Minimum mask extent so regions survive pooling to coarse latent (8×8 from input)
MIN_FREQ_BINS = 4
MIN_TIME_FRAMES = 4


class PerlinNoiseStrategy:
    """
    Binarized Perlin noise for diverse anomaly regions (AudDSR / DSR-style).

    Pure 2D Perlin (optionally rotated in vision DSR) does not match typical
    log-mel defects well: anomalies often cover **most time** in a narrow band
    or **most mel bins** for a short interval (ICASSP 2024 Sec. 3.3). We keep
    **axis-aligned** noise only and mix:

    - **2d**: full (n_mels, T) Perlin;
    - **time_ridge**: 1×T Perlin, broadcast across mel (duration-wide structure);
    - **freq_ridge**: n_mels×1 Perlin, broadcast across time (band-like structure).
    """

    def __init__(
        self,
        spectrogram_shape: tuple[int, int],
        threshold: float = 0.5,
        perlin_scale_range: tuple[int, int] = (1, 5),
    ) -> None:
        """
        Args:
            spectrogram_shape: (n_mels, T)
            threshold: binarization threshold (noise roughly in [-1, 1])
            perlin_scale_range: inclusive exponent range; grid scale is 2**k per axis
        """
        self.spectrogram_shape = spectrogram_shape
        self.threshold = threshold
        n_mels, T = spectrogram_shape
        max_exp_freq = int(np.log2(max(1, n_mels // MIN_FREQ_BINS)))
        max_exp_time = int(np.log2(max(1, T // MIN_TIME_FRAMES)))
        effective_max = min(max_exp_freq, max_exp_time, perlin_scale_range[1])
        effective_min = min(perlin_scale_range[0], effective_max)
        self.perlin_scale_range = (effective_min, effective_max)

    def _rand_res_1d(self) -> int:
        return 2 ** random.randint(*self.perlin_scale_range)

    def _noise_2d(self) -> np.ndarray:
        n_mels, T = self.spectrogram_shape
        ry = self._rand_res_1d()
        rx = self._rand_res_1d()
        return rand_perlin_2d_np((n_mels, T), (ry, rx))

    def _noise_time_ridge(self) -> np.ndarray:
        """Correlated across all mel bins (single temporal Perlin profile)."""
        n_mels, T = self.spectrogram_shape
        rt = min(self._rand_res_1d(), max(1, T))
        noise = rand_perlin_2d_np((1, T), (1, rt))
        return np.repeat(noise, n_mels, axis=0)

    def _noise_freq_ridge(self) -> np.ndarray:
        """Correlated across all frames (single spectral Perlin profile)."""
        n_mels, T = self.spectrogram_shape
        rf = min(self._rand_res_1d(), max(1, n_mels))
        noise = rand_perlin_2d_np((n_mels, 1), (rf, 1))
        return np.repeat(noise, T, axis=1)

    def _one_mask_numpy(self) -> np.ndarray:
        # Equal mix: no hyperparameters tied to a specific machine type
        choice = random.randrange(3)
        if choice == 0:
            noise = self._noise_2d()
        elif choice == 1:
            noise = self._noise_time_ridge()
        else:
            noise = self._noise_freq_ridge()
        return (noise > self.threshold).astype(np.float32)

    def __call__(
        self,
        batch_size: int,
        device: torch.device | str,
    ) -> torch.Tensor:
        masks = []
        for _ in range(batch_size):
            binary = self._one_mask_numpy()
            masks.append(torch.from_numpy(binary).unsqueeze(0).unsqueeze(0))
        return torch.cat(masks, dim=0).to(device)


class AudioSpecificStrategy:
    """
    One random frequency band, then several disjoint time segments inside it.

    Matches the AudDSR description (ICASSP 2024 Sec. 3.3): choose where in
    frequency the defect lives, then mark a few contiguous time intervals.
    Sampling is uniform over valid bandwidths and segment lengths so it
    generalizes across all mel lengths and time horizons used in training.
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
        self.min_segments = max(1, min_segments)
        self.max_segments = max(self.min_segments, max_segments)

    def _single_mask_numpy(self) -> np.ndarray:
        n_mels, T = self.n_mels, self.T
        M = np.zeros((n_mels, T), dtype=np.float32)

        # 50/50: (A) almost full mel range, very short in time  vs  (B) thin band, long in time
        wide_band_short_time = random.random() < 0.5

        if wide_band_short_time:
            # --- Type A: high bandwidth (most of the spectrogram), short time columns ---
            bandwidth = random.randint(max(MIN_FREQ_BINS, n_mels * 2 // 3), n_mels)
            f_low = random.randint(0, n_mels - bandwidth)
            f_high = f_low + bandwidth
            seg_len_hi = max(MIN_TIME_FRAMES, min(T, 24))  # cap "short" (tune 8–32)
            seg_len_lo = MIN_TIME_FRAMES
        else:
            # --- Type B: low bandwidth (1–8 mel bins), longer segments along time ---
            bw_max = min(8, n_mels)
            bandwidth = random.randint(1, bw_max)
            f_low = random.randint(0, n_mels - bandwidth)
            f_high = f_low + bandwidth
            seg_len_lo = max(MIN_TIME_FRAMES, T // 16)  # tune: fraction of T
            seg_len_hi = T  # full clip possible; cap if you want milder:
            # seg_len_hi = min(T, max(seg_len_lo + 1, T * 3 // 4))

        n_seg = random.randint(self.min_segments, self.max_segments)
        for _ in range(n_seg):
            if T <= MIN_TIME_FRAMES:
                t_start, t_end = 0, T
            else:
                lo = min(seg_len_lo, T)
                hi = min(seg_len_hi, T)
                if hi < lo:
                    lo, hi = hi, lo  # safety
                seg_len = random.randint(lo, hi)
                max_start = max(0, T - seg_len)
                t_start = random.randint(0, max_start)
                t_end = t_start + seg_len
            M[f_low:f_high, t_start:t_end] = 1.0

        return M

    def single_mask(self, device: torch.device | str) -> torch.Tensor:
        """One (1, 1, n_mels, T) mask."""
        arr = self._single_mask_numpy()
        return torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

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
