"""
Anomaly map generation for sDSR training.

Strategies (same interface: __call__(batch_size, device) -> (B, 1, H, W)):
1. PerlinNoiseStrategy: binarized true 2D Perlin noise on the mel–time grid
   (ICASSP 2024 AudDSR-style), then reduced to a small number of connected
   blobs so anomalies stay sparse and spectrogram-shaped.
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
from scipy import ndimage

from .perlin import rand_perlin_2d_np

# Minimum mask extent so regions survive pooling to coarse latent (8×8 from input)
MIN_FREQ_BINS = 4
MIN_TIME_FRAMES = 4

# Helper functions
def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    """True if [a0,a1) and [b0,b1) intersect (half-open)."""
    return not (a1 <= b0 or b1 <= a0)


def _sample_disjoint_time_segments(
    n_seg: int,
    T: int,
    seg_len_lo: int,
    seg_len_hi: int,
    max_tries_per_seg: int = 400,
) -> list[tuple[int, int]]:
    """
    Sample up to n_seg non-overlapping half-open intervals [t_start, t_end) in [0, T).
    Stops early if a segment cannot be placed after max_tries_per_seg attempts.
    """
    if T <= MIN_TIME_FRAMES:
        return [(0, T)]
    lo = max(1, min(seg_len_lo, T))
    hi = min(seg_len_hi, T)
    if hi < lo:
        lo, hi = hi, lo
    if lo > T:
        lo = min(MIN_TIME_FRAMES, T)
    if hi < lo:
        return [(0, T)]

    intervals: list[tuple[int, int]] = []
    for _ in range(n_seg):
        placed = False
        for _ in range(max_tries_per_seg):
            seg_len = random.randint(lo, hi)
            if seg_len > T:
                continue
            max_start = T - seg_len
            t_start = random.randint(0, max_start)
            t_end = t_start + seg_len
            if any(
                _intervals_overlap(t_start, t_end, s, e) for s, e in intervals
            ):
                continue
            intervals.append((t_start, t_end))
            placed = True
            break
        if not placed:
            break

    if not intervals:
        seg_len = min(hi, T)
        return [(0, max(seg_len, min(lo, T)))]
    return intervals


class PerlinNoiseStrategy:
    """
    Binarized Perlin noise for diverse anomaly regions (AudDSR / DSR-style).
    This strategy generates a single true 2D Perlin field on the (n_mels, T)
    grid, binarizes it, then keeps only a few connected blobs so anomalies
    remain sparse and spectrogram-shaped.
    """

    def __init__(
        self,
        spectrogram_shape: tuple[int, int],
        threshold: float = 0.5,
        perlin_scale_range: tuple[int, int] = (1, 5),
        min_blobs: int = 2,
        max_blobs: int = 10,
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
        self.min_blobs = max(1, min_blobs)
        self.max_blobs = max(self.min_blobs, max_blobs)

    def _rand_res_1d(self) -> int:
        return 2 ** random.randint(*self.perlin_scale_range)

    def _noise_2d(self) -> np.ndarray:
        n_mels, T = self.spectrogram_shape
        ry = self._rand_res_1d()
        rx = self._rand_res_1d()
        return rand_perlin_2d_np((n_mels, T), (ry, rx))


    def _one_mask_numpy(self) -> np.ndarray:
        noise = self._noise_2d()
        binary = (noise > self.threshold)
        return binary

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
    One random frequency band, then several **pairwise disjoint** time segments
    inside it (AudDSR-style). Three mixture modes:
    - **A**: wide mel band, longer short-time support (raised masked area).
    - **B**: narrow band, medium–long time, capped at 75% of T (no full-timeline stripe).
    - **C**: medium bandwidth and medium duration (between A and B).
    """

    def __init__(
        self,
        spectrogram_shape: tuple[int, int],
        n_mels: int,
        T: int,
        min_segments: int = 4,
        max_segments: int = 8,
    ) -> None:
        self.spectrogram_shape = spectrogram_shape
        self.n_mels = n_mels
        self.T = T
        self.min_segments = max(1, min_segments)
        self.max_segments = max(self.min_segments, max_segments)

    def _single_mask_numpy(self) -> np.ndarray:
        n_mels, T = self.n_mels, self.T
        M = np.zeros((n_mels, T), dtype=np.float32)

        mode = random.randrange(3)  # A, B, or C

        if mode == 0:
            # --- Type A: high bandwidth, longer short-time columns (seg up to ~40–80) ---
            bandwidth = random.randint(max(MIN_FREQ_BINS, n_mels * 2 // 3), n_mels)
            f_low = random.randint(0, n_mels - bandwidth)
            f_high = f_low + bandwidth
            seg_len_lo = MIN_TIME_FRAMES
            if T >= 40:
                seg_len_hi = min(T, random.randint(MIN_TIME_FRAMES+1, min(40, T)))
            else:
                seg_len_hi = T
        elif mode == 1:
            # --- Type B: narrow band, capped timeline (no full-T stripe) ---
            bw_max = min(16, n_mels)
            bandwidth = random.randint(1, bw_max)
            f_low = random.randint(0, n_mels - bandwidth)
            f_high = f_low + bandwidth
            seg_len_lo = max(MIN_TIME_FRAMES, T // 16)
            seg_len_hi = min(T, (T * 3) // 4)
        else:
            # --- Type C: medium bandwidth + medium duration ---
            lo_bw = max(MIN_FREQ_BINS, n_mels // 4)
            hi_bw = max(lo_bw, (2 * n_mels) // 3)
            bandwidth = random.randint(lo_bw, hi_bw)
            f_low = random.randint(0, n_mels - bandwidth)
            f_high = f_low + bandwidth
            seg_len_lo = max(MIN_TIME_FRAMES, T // 20)
            seg_len_hi = min(T, max(seg_len_lo, T // 5))

        if seg_len_hi < seg_len_lo:
            seg_len_lo, seg_len_hi = seg_len_hi, seg_len_lo

        n_seg = random.randint(self.min_segments, self.max_segments)
        for t_start, t_end in _sample_disjoint_time_segments(
            n_seg, T, seg_len_lo, seg_len_hi
        ):
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
