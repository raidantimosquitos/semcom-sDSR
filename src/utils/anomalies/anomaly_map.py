"""
Anomaly map generation for sDSR training.

Strategies (same interface: __call__(batch_size, device) -> (B, 1, H, W)):
1. PerlinNoiseStrategy: binarized true 2D Perlin noise on the mel–time grid
   (ICASSP 2024 AudDSR-style), then reduced to a small number of connected
   blobs so anomalies stay sparse and spectrogram-shaped.
2. SliderSpecificStrategy (alias AudioSpecificStrategy): one random frequency
   band + several disjoint time segments (tuned for slider-like defects).
   3. MachineSpecificStrategy: dispatches per DCASE machine_type (slider, ToyCar,
   ToyConveyor, pump/valve-specific masks, placeholders for the rest).

When strategy is ``machine_specific`` or ``both``, masks can depend on the
training sample's ``machine_type`` (passed from the dataset).
"""

from __future__ import annotations

import random
from typing import Literal, Sequence

import numpy as np
import torch

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


def _sample_disjoint_time_segments_in_window(
    n_seg: int,
    t_lo: int,
    t_hi: int,
    seg_len_lo: int,
    seg_len_hi: int,
    max_tries_per_seg: int = 400,
) -> list[tuple[int, int]]:
    """
    Disjoint [t_start, t_end) inside [t_lo, t_hi) by sampling in T_sub = t_hi - t_lo
    then offsetting (reuses _sample_disjoint_time_segments on [0, T_sub)).
    """
    t_lo = max(0, min(t_lo, t_hi - 1))
    t_hi = max(t_lo + 1, t_hi)
    T_sub = t_hi - t_lo
    if T_sub <= MIN_TIME_FRAMES:
        return [(t_lo, t_hi)]
    raw = _sample_disjoint_time_segments(
        n_seg, T_sub, seg_len_lo, seg_len_hi, max_tries_per_seg
    )
    return [(t_lo + a, t_lo + b) for a, b in raw]


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


class SliderSpecificStrategy:
    """
    One random frequency band, then several **pairwise disjoint** time segments
    inside it (AudDSR-style). Three mixture modes:
    - **A**: wide mel band, longer short-time support (raised masked area).
    - **B**: narrow band, medium–long time, capped at 75% of T (no full-timeline stripe).
    - **C**: medium bandwidth and medium duration (between A and B).

    Tuned for slider-like defects; used as the ``slider`` branch of
    :class:`MachineSpecificStrategy`.
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


class ToyCarSpecificStrategy:
    """
    Mel-band mixture aligned with ToyCar normal-vs-anomaly difference maps
    (primary ~76–84, secondary ~38–52, high ~108–124), disjoint time segments
    inside an active time window (default 50–300 for T=320), optional cloud
    patches in mid–high mel.
    """

    def __init__(
        self,
        spectrogram_shape: tuple[int, int],
        n_mels: int,
        T: int,
        t_active_lo: int = 50,
        t_active_hi: int = 300,
        min_segments: int = 2,
        max_segments: int = 6,
    ) -> None:
        self.spectrogram_shape = spectrogram_shape
        self.n_mels = n_mels
        self.T = T
        self.t_active_lo = max(0, min(t_active_lo, T - 1))
        self.t_active_hi = max(self.t_active_lo + 1, min(t_active_hi, T))
        self.min_segments = max(1, min_segments)
        self.max_segments = max(self.min_segments, max_segments)

    def _pick_band(self) -> tuple[int, int]:
        n_mels = self.n_mels
        if n_mels <= MIN_FREQ_BINS + 2:
            bw = max(MIN_FREQ_BINS, min(n_mels, 8))
            f_low = random.randint(0, max(0, n_mels - bw))
            return f_low, min(n_mels, f_low + bw)
        u = random.random()

        def _band(center_lo: int, center_hi: int, bw_lo: int, bw_hi: int) -> tuple[int, int]:
            center_hi = min(center_hi, n_mels - 1)
            center_lo = max(0, min(center_lo, center_hi))
            if center_lo > center_hi:
                bw = random.randint(MIN_FREQ_BINS, min(32, n_mels))
                f0 = random.randint(0, max(0, n_mels - bw))
                return f0, min(n_mels, f0 + bw)
            c = random.randint(center_lo, center_hi)
            hi_bw = min(bw_hi, n_mels)
            lo_bw = max(MIN_FREQ_BINS, bw_lo)
            if lo_bw > hi_bw:
                bw = max(MIN_FREQ_BINS, min(n_mels, 8))
                f_low = random.randint(0, max(0, n_mels - bw))
                return f_low, min(n_mels, f_low + bw)
            bw = random.randint(lo_bw, hi_bw)
            bw = max(MIN_FREQ_BINS, min(bw, n_mels))
            f_low = max(0, min(c - bw // 2, n_mels - bw))
            f_high = min(n_mels, f_low + bw)
            if f_high - f_low < MIN_FREQ_BINS:
                f_high = min(n_mels, f_low + MIN_FREQ_BINS)
            return f_low, f_high

        if u < 0.52:
            return _band(76, 84, 6, 14)
        if u < 0.78:
            return _band(38, 52, 6, 14)
        if u < 0.92:
            return _band(108, 124, 8, 18)
        bw = random.randint(MIN_FREQ_BINS, min(32, n_mels))
        f_low = random.randint(0, max(0, n_mels - bw))
        return f_low, min(n_mels, f_low + bw)

    def _maybe_cloud_patches(self, M: np.ndarray) -> None:
        if random.random() >= 0.35:
            return
        n_mels, T = self.n_mels, self.T
        t_lo, t_hi = self.t_active_lo, self.t_active_hi
        for _ in range(random.randint(1, 3)):
            bw = random.randint(MIN_FREQ_BINS, min(24, n_mels))
            f0 = random.randint(0, max(0, n_mels - bw))
            f1 = min(n_mels, f0 + bw)
            seg_lo = max(MIN_TIME_FRAMES, (t_hi - t_lo) // 20)
            seg_hi = min(t_hi - t_lo, max(seg_lo, (t_hi - t_lo) // 4))
            for t_start, t_end in _sample_disjoint_time_segments_in_window(
                1, t_lo, t_hi, seg_lo, seg_hi
            ):
                M[f0:f1, t_start:t_end] = 1.0

    def _single_mask_numpy(self) -> np.ndarray:
        n_mels, T = self.n_mels, self.T
        M = np.zeros((n_mels, T), dtype=np.float32)
        t_lo, t_hi = self.t_active_lo, self.t_active_hi
        T_sub = t_hi - t_lo
        seg_len_lo = max(MIN_TIME_FRAMES, T_sub // 25)
        seg_len_hi = max(seg_len_lo, min(T_sub, T_sub // 3))

        n_seg = random.randint(self.min_segments, self.max_segments)
        f_low, f_high = self._pick_band()
        for t_start, t_end in _sample_disjoint_time_segments_in_window(
            n_seg, t_lo, t_hi, seg_len_lo, seg_len_hi
        ):
            M[f_low:f_high, t_start:t_end] = 1.0

        if random.random() < 0.22:
            f2_low, f2_high = self._pick_band()
            n2 = random.randint(1, max(1, n_seg // 2))
            for t_start, t_end in _sample_disjoint_time_segments_in_window(
                n2, t_lo, t_hi, seg_len_lo, seg_len_hi
            ):
                M[f2_low:f2_high, t_start:t_end] = 1.0

        self._maybe_cloud_patches(M)
        return M

    def single_mask(self, device: torch.device | str) -> torch.Tensor:
        arr = self._single_mask_numpy()
        return (
            torch.from_numpy(arr.astype(np.float32))
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )

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


class ToyConveyorSpecificStrategy:
    """
    Mel-band mixture aligned with ToyConveyor pooled |anom−norm| maps (see
    ``notebooks/ToyConveyor/ToyConveyor_anom_norm_comp``): strong mid-frequency
    horizontal blobs, secondary low-mid rumble and high-mid scrape, with
    occasional long-time stripes (conveyor-like) and small scattered patches.
    Disjoint time segments inside a wide active window (default ~35–305 for
    T=320); band centers, widths, and segment counts are randomized.
    """

    def __init__(
        self,
        spectrogram_shape: tuple[int, int],
        n_mels: int,
        T: int,
        t_active_lo: int = 35,
        t_active_hi: int = 305,
        min_segments: int = 2,
        max_segments: int = 7,
    ) -> None:
        self.spectrogram_shape = spectrogram_shape
        self.n_mels = n_mels
        self.T = T
        self.t_active_lo = max(0, min(t_active_lo, T - 1))
        self.t_active_hi = max(self.t_active_lo + 1, min(t_active_hi, T))
        self.min_segments = max(1, min_segments)
        self.max_segments = max(self.min_segments, max_segments)

    def _pick_band(self) -> tuple[int, int]:
        n_mels = self.n_mels
        if n_mels <= MIN_FREQ_BINS + 2:
            bw = max(MIN_FREQ_BINS, min(n_mels, 8))
            f_low = random.randint(0, max(0, n_mels - bw))
            return f_low, min(n_mels, f_low + bw)
        u = random.random()

        def _band(center_lo: int, center_hi: int, bw_lo: int, bw_hi: int) -> tuple[int, int]:
            center_hi = min(center_hi, n_mels - 1)
            center_lo = max(0, min(center_lo, center_hi))
            if center_lo > center_hi:
                bw = random.randint(MIN_FREQ_BINS, min(32, n_mels))
                f0 = random.randint(0, max(0, n_mels - bw))
                return f0, min(n_mels, f0 + bw)
            c = random.randint(center_lo, center_hi)
            hi_bw = min(bw_hi, n_mels)
            lo_bw = max(MIN_FREQ_BINS, bw_lo)
            if lo_bw > hi_bw:
                bw = max(MIN_FREQ_BINS, min(n_mels, 8))
                f_low = random.randint(0, max(0, n_mels - bw))
                return f_low, min(n_mels, f_low + bw)
            bw = random.randint(lo_bw, hi_bw)
            bw = max(MIN_FREQ_BINS, min(bw, n_mels))
            f_low = max(0, min(c - bw // 2, n_mels - bw))
            f_high = min(n_mels, f_low + bw)
            if f_high - f_low < MIN_FREQ_BINS:
                f_high = min(n_mels, f_low + MIN_FREQ_BINS)
            return f_low, f_high

        # Priors: mid-frequency emphasis (pooled diff map), low-mid rumble, high scrape.
        if u < 0.46:
            return _band(50, 72, 10, 26)
        if u < 0.74:
            return _band(18, 42, 6, 18)
        if u < 0.90:
            return _band(84, 118, 8, 22)
        bw = random.randint(MIN_FREQ_BINS, min(36, n_mels))
        f_low = random.randint(0, max(0, n_mels - bw))
        return f_low, min(n_mels, f_low + bw)

    def _maybe_cloud_patches(self, M: np.ndarray) -> None:
        if random.random() >= 0.38:
            return
        n_mels, T = self.n_mels, self.T
        t_lo, t_hi = self.t_active_lo, self.t_active_hi
        for _ in range(random.randint(1, 3)):
            bw = random.randint(MIN_FREQ_BINS, min(28, n_mels))
            f0 = random.randint(0, max(0, n_mels - bw))
            f1 = min(n_mels, f0 + bw)
            seg_lo = max(MIN_TIME_FRAMES, (t_hi - t_lo) // 20)
            seg_hi = min(t_hi - t_lo, max(seg_lo, (t_hi - t_lo) // 4))
            for t_start, t_end in _sample_disjoint_time_segments_in_window(
                1, t_lo, t_hi, seg_lo, seg_hi
            ):
                M[f0:f1, t_start:t_end] = 1.0

    def _single_mask_numpy(self) -> np.ndarray:
        n_mels, T = self.n_mels, self.T
        M = np.zeros((n_mels, T), dtype=np.float32)
        t_lo, t_hi = self.t_active_lo, self.t_active_hi
        T_sub = t_hi - t_lo

        # Occasional single long-time band (horizontal stripe in the spectrogram).
        if random.random() < 0.14:
            f_low, f_high = self._pick_band()
            span_lo = max(MIN_TIME_FRAMES, T_sub // 3)
            span_hi = min(T_sub, max(span_lo + 1, (T_sub * 4) // 5))
            for t_start, t_end in _sample_disjoint_time_segments_in_window(
                1, t_lo, t_hi, span_lo, span_hi
            ):
                M[f_low:f_high, t_start:t_end] = 1.0
            if random.random() < 0.35:
                f2_low, f2_high = self._pick_band()
                n2 = random.randint(1, 3)
                seg_lo = max(MIN_TIME_FRAMES, T_sub // 25)
                seg_hi = max(seg_lo, min(T_sub, T_sub // 4))
                for t_start, t_end in _sample_disjoint_time_segments_in_window(
                    n2, t_lo, t_hi, seg_lo, seg_hi
                ):
                    M[f2_low:f2_high, t_start:t_end] = 1.0
            self._maybe_cloud_patches(M)
            return M

        seg_len_lo = max(MIN_TIME_FRAMES, T_sub // 28)
        seg_len_hi = max(seg_len_lo, min(T_sub, T_sub // 2))

        n_seg = random.randint(self.min_segments, self.max_segments)
        f_low, f_high = self._pick_band()
        for t_start, t_end in _sample_disjoint_time_segments_in_window(
            n_seg, t_lo, t_hi, seg_len_lo, seg_len_hi
        ):
            M[f_low:f_high, t_start:t_end] = 1.0

        if random.random() < 0.24:
            f2_low, f2_high = self._pick_band()
            n2 = random.randint(1, max(1, n_seg // 2))
            for t_start, t_end in _sample_disjoint_time_segments_in_window(
                n2, t_lo, t_hi, seg_len_lo, seg_len_hi
            ):
                M[f2_low:f2_high, t_start:t_end] = 1.0

        self._maybe_cloud_patches(M)
        return M

    def single_mask(self, device: torch.device | str) -> torch.Tensor:
        arr = self._single_mask_numpy()
        return (
            torch.from_numpy(arr.astype(np.float32))
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )

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


class PumpSpecificStrategy:
    """
    Pump-like anomalies: mostly **vertical time stripes** on the mel-time grid.

    Rationale (from the repo's pump spectrogram demo):
    the simplest pump mask prior used there sets `mask[:, t0:t1] = 1`
    (full mel bandwidth over random time segments).

    We keep it sparse via a small number of disjoint time segments, and we add
    randomization via:
    - variable segment lengths (roughly 10–60 frames when T=320),
    - mostly full-band stripes, but occasionally restricted to a random
      frequency band,
    - optional extra "cloud" patches.
    """

    def __init__(
        self,
        spectrogram_shape: tuple[int, int],
        n_mels: int,
        T: int,
        t_active_lo: int = 20,
        t_active_hi: int = 300,
        min_segments: int = 2,
        max_segments: int = 6,
    ) -> None:
        self.spectrogram_shape = spectrogram_shape
        self.n_mels = n_mels
        self.T = T
        self.t_active_lo = max(0, min(t_active_lo, T - 1))
        self.t_active_hi = max(self.t_active_lo + 1, min(t_active_hi, T))
        self.min_segments = max(1, min_segments)
        self.max_segments = max(self.min_segments, max_segments)

    def _single_mask_numpy(self) -> np.ndarray:
        n_mels, T = self.n_mels, self.T
        M = np.zeros((n_mels, T), dtype=np.float32)

        t_lo, t_hi = self.t_active_lo, self.t_active_hi
        T_sub = t_hi - t_lo

        # Two modes: either fewer longer segments, or more short segments.
        if random.random() < 0.5:
            seg_len_lo = max(MIN_TIME_FRAMES, T_sub // 60)
            seg_len_hi = min(T_sub, max(seg_len_lo + 1, T_sub // 6))
        else:
            seg_len_lo = max(MIN_TIME_FRAMES, T_sub // 90)
            seg_len_hi = min(T_sub, max(seg_len_lo + 1, T_sub // 10))

        n_seg = random.randint(self.min_segments, self.max_segments)

        for t_start, t_end in _sample_disjoint_time_segments_in_window(
            n_seg, t_lo, t_hi, seg_len_lo, seg_len_hi
        ):
            # Mostly full-band vertical stripes, sometimes restricted to a freq band.
            if random.random() < 0.82:
                M[:, t_start:t_end] = 1.0
            else:
                bw = random.randint(MIN_FREQ_BINS, max(MIN_FREQ_BINS + 1, n_mels // 2))
                f_low = random.randint(0, max(0, n_mels - bw))
                M[f_low : f_low + bw, t_start:t_end] = 1.0

        # Occasionally add a couple of small "cloud" patches (still time-local).
        if random.random() < 0.30:
            for _ in range(random.randint(1, 3)):
                bw = random.randint(MIN_FREQ_BINS, min(24, n_mels))
                f0 = random.randint(0, max(0, n_mels - bw))
                seg_lo = max(MIN_TIME_FRAMES, T_sub // 40)
                seg_hi = min(T_sub, max(seg_lo + 1, T_sub // 8))
                seg_len = random.randint(seg_lo, seg_hi)
                t_start = random.randint(t_lo, max(t_lo, t_hi - seg_len))
                t_end = t_start + seg_len
                M[f0 : f0 + bw, t_start:t_end] = 1.0

        return M

    def single_mask(self, device: torch.device | str) -> torch.Tensor:
        arr = self._single_mask_numpy()
        return (
            torch.from_numpy(arr.astype(np.float32))
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )

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


class PlaceholderMachineSpecificStrategy:
    """
    TODO: Replace with frequency/time priors from normal-vs-anomaly analysis
    per machine type. Returns one valid rectangular band + disjoint segments
    so training does not crash.
    """

    def __init__(
        self,
        spectrogram_shape: tuple[int, int],
        n_mels: int,
        T: int,
        min_segments: int = 2,
        max_segments: int = 5,
    ) -> None:
        self.spectrogram_shape = spectrogram_shape
        self.n_mels = n_mels
        self.T = T
        self.min_segments = max(1, min_segments)
        self.max_segments = max(self.min_segments, max_segments)

    def _single_mask_numpy(self) -> np.ndarray:
        n_mels, T = self.n_mels, self.T
        M = np.zeros((n_mels, T), dtype=np.float32)
        bandwidth = random.randint(MIN_FREQ_BINS, max(MIN_FREQ_BINS, n_mels // 4))
        f_low = random.randint(0, max(0, n_mels - bandwidth))
        f_high = f_low + bandwidth
        seg_len_lo = MIN_TIME_FRAMES
        seg_len_hi = min(T, max(MIN_TIME_FRAMES + 1, T // 6))
        n_seg = random.randint(self.min_segments, self.max_segments)
        for t_start, t_end in _sample_disjoint_time_segments(
            n_seg, T, seg_len_lo, seg_len_hi
        ):
            M[f_low:f_high, t_start:t_end] = 1.0
        return M

    def single_mask(self, device: torch.device | str) -> torch.Tensor:
        arr = self._single_mask_numpy()
        return (
            torch.from_numpy(arr.astype(np.float32))
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )

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


class MachineSpecificStrategy:
    """
    Dispatches to per-machine mask strategies using exact DCASE ``machine_type``
    strings (e.g. ``\"slider\"``, ``\"ToyCar\"``, ``\"ToyConveyor\"``). ``None`` or
    unknown types use
    :class:`PlaceholderMachineSpecificStrategy` (not slider-shaped masks).
    """

    def __init__(
        self,
        spectrogram_shape: tuple[int, int],
        n_mels: int,
        T: int,
    ) -> None:
        self.spectrogram_shape = spectrogram_shape
        self.n_mels = n_mels
        self.T = T
        self._slider = SliderSpecificStrategy(spectrogram_shape, n_mels, T)
        self._toycar = ToyCarSpecificStrategy(spectrogram_shape, n_mels, T)
        self._toyconveyor = ToyConveyorSpecificStrategy(spectrogram_shape, n_mels, T)
        self._pump = PumpSpecificStrategy(spectrogram_shape, n_mels, T)
        self._placeholder = PlaceholderMachineSpecificStrategy(
            spectrogram_shape, n_mels, T
        )
        self._by_type: dict[
            str,
            SliderSpecificStrategy
            | ToyCarSpecificStrategy
            | ToyConveyorSpecificStrategy
            | PumpSpecificStrategy
            | PlaceholderMachineSpecificStrategy,
        ] = {
            "slider": self._slider,
            "ToyCar": self._toycar,
            "ToyConveyor": self._toyconveyor,
            "fan": self._placeholder,
            "pump": self._pump,
            "valve": self._placeholder,
        }

    def _strategy_for(self, machine_type: str | None):
        if machine_type is None:
            return self._placeholder
        key = machine_type.strip()
        return self._by_type.get(key, self._placeholder)

    def mask_for(
        self,
        machine_type: str | None,
        batch_size: int,
        device: torch.device | str,
    ) -> torch.Tensor:
        strat = self._strategy_for(machine_type)
        return strat(batch_size, device)

    def single_mask_for(
        self,
        machine_type: str | None,
        device: torch.device | str,
    ) -> torch.Tensor:
        strat = self._strategy_for(machine_type)
        return strat.single_mask(device)


# Deprecated name; use SliderSpecificStrategy.
AudioSpecificStrategy = SliderSpecificStrategy


def _normalize_anomaly_strategy(
    strategy: Literal["perlin", "machine_specific", "audio_specific", "both"],
) -> Literal["perlin", "machine_specific", "both"]:
    if strategy == "audio_specific":
        return "machine_specific"
    return strategy


class AnomalyMapGenerator:
    """
    Generate anomaly map M for training using one of several strategies.
    When force_anomaly=False, each sample gets an independent draw: with
    probability zero_mask_prob a zero mask (no anomaly), else a generated mask.
    """

    def __init__(
        self,
        strategy: Literal["perlin", "machine_specific", "audio_specific", "both"],
        spectrogram_shape: tuple[int, int],
        n_mels: int | None = None,
        T: int | None = None,
        zero_mask_prob: float = 0.5,
    ) -> None:
        """
        Args:
            strategy: ``perlin``, ``machine_specific`` (per-type masks; slider +
                ToyCar + placeholders), ``both`` (50% Perlin vs machine-specific),
                or deprecated alias ``audio_specific`` (same as ``machine_specific``).
            spectrogram_shape: (n_mels, T)
            n_mels, T: required for ``machine_specific`` and ``both``
            zero_mask_prob: per-sample probability of returning a zero mask (no anomaly)
        """
        self._strategy_raw = strategy
        self.strategy_name = _normalize_anomaly_strategy(strategy)
        self.spectrogram_shape = spectrogram_shape
        self.zero_mask_prob = zero_mask_prob
        self.perlin = (
            PerlinNoiseStrategy(spectrogram_shape)
            if self.strategy_name in ("perlin", "both")
            else None
        )
        self.machine_specific = (
            MachineSpecificStrategy(spectrogram_shape, n_mels, T)
            if self.strategy_name in ("machine_specific", "both")
            and n_mels is not None
            and T is not None
            else None
        )

    def _generate_one(
        self,
        device: torch.device | str,
        machine_type: str | None = None,
    ) -> torch.Tensor:
        """Generate a single non-zero mask (1, 1, H, W)."""
        if self.strategy_name == "perlin":
            assert self.perlin is not None
            return self.perlin(1, device)
        if self.strategy_name == "machine_specific":
            assert self.machine_specific is not None
            return self.machine_specific.single_mask_for(machine_type, device)
        if self.strategy_name == "both":
            if random.random() < 0.5:
                assert self.perlin is not None
                return self.perlin(1, device)
            assert self.machine_specific is not None
            return self.machine_specific.single_mask_for(machine_type, device)
        raise RuntimeError(f"Unknown strategy: {self.strategy_name}")

    def generate_for_training_sample(
        self,
        device: torch.device | str,
        force_anomaly: bool = True,
        machine_type: str | None = None,
    ) -> torch.Tensor:
        """Generate one mask for a single training sample."""
        if force_anomaly:
            if self.strategy_name == "perlin":
                assert self.perlin is not None
                return self.perlin(1, device)
            if self.strategy_name == "machine_specific":
                assert self.machine_specific is not None
                return self.machine_specific.single_mask_for(machine_type, device)
            if self.strategy_name == "both":
                if random.random() < 0.5:
                    assert self.perlin is not None
                    return self.perlin(1, device)
                assert self.machine_specific is not None
                return self.machine_specific.single_mask_for(machine_type, device)
            raise RuntimeError(f"Unknown strategy: {self.strategy_name}")
        return self.generate(1, device, force_anomaly=False, machine_types=None)

    def generate(
        self,
        batch_size: int,
        device: torch.device | str,
        force_anomaly: bool = False,
        machine_types: Sequence[str] | None = None,
    ) -> torch.Tensor:
        """
        Generate anomaly map M.

        When force_anomaly=False, each sample is decided independently: with
        probability zero_mask_prob the mask is all zeros, else one mask from
        the strategy (per-sample 50% no-anomaly, matching original DSR).

        For ``machine_specific`` / ``both`` with ``force_anomaly=True`` and
        ``batch_size > 1``, pass ``machine_types`` of length ``batch_size`` for
        per-row types; if omitted, masks use the placeholder strategy (unknown
        type) for every sample — training uses
        :meth:`generate_for_training_sample` with per-sample ``machine_type``.

        Args:
            batch_size: number of masks to generate
            device: torch device
            force_anomaly: if True, skip zero_mask_prob and always generate real masks
            machine_types: optional per-batch machine_type strings (DCASE keys)

        Returns:
            M: (B, 1, n_mels, T) binary mask
        """
        if force_anomaly:
            if self.strategy_name == "perlin":
                assert self.perlin is not None
                return self.perlin(batch_size, device)
            if self.strategy_name == "machine_specific":
                assert self.machine_specific is not None
                if machine_types is None:
                    return self.machine_specific.mask_for(None, batch_size, device)
                if len(machine_types) != batch_size:
                    raise ValueError(
                        f"machine_types length {len(machine_types)} != batch_size {batch_size}"
                    )
                parts = [
                    self.machine_specific.single_mask_for(mt, device)
                    for mt in machine_types
                ]
                return torch.cat(parts, dim=0)
            if self.strategy_name == "both":
                if random.random() < 0.5:
                    assert self.perlin is not None
                    return self.perlin(batch_size, device)
                assert self.machine_specific is not None
                if machine_types is None:
                    return self.machine_specific.mask_for(None, batch_size, device)
                if len(machine_types) != batch_size:
                    raise ValueError(
                        f"machine_types length {len(machine_types)} != batch_size {batch_size}"
                    )
                parts = [
                    self.machine_specific.single_mask_for(mt, device)
                    for mt in machine_types
                ]
                return torch.cat(parts, dim=0)
            raise RuntimeError(f"Unknown strategy: {self.strategy_name}")

        masks = []
        for _ in range(batch_size):
            if random.random() < self.zero_mask_prob:
                masks.append(
                    torch.zeros(
                        1,
                        1,
                        *self.spectrogram_shape,
                        device=device,
                        dtype=torch.float32,
                    )
                )
            else:
                masks.append(self._generate_one(device, machine_type=None))
        return torch.cat(masks, dim=0)
