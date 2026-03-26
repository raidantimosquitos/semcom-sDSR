"""
Anomaly map generation for sDSR training.

This project uses a single **audio_specific** synthetic anomaly mask generator
for all machine types, matching the paper description:

- Select a random frequency band (2–16 mel bins).
- Mark multiple disjoint time intervals inside that band as anomalous.
"""

from __future__ import annotations

import random
from typing import Sequence

import numpy as np
import torch

from .perlin import generate_perlin_noise_2d

# Minimum mask extent so regions survive pooling to coarse latent (8×8 from input).
# NOTE: For audio_specific we allow narrower (2–16) bands per paper.
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


class AudioSpecificStrategy:
    """
    Paper-style audio-specific anomaly masks (applies to all machine types).

    Procedure:
    - Pick a random frequency band width in [2, 16] mel bins.
    - Pick a random band location (f_low:f_high).
    - Sample several disjoint time intervals and mark them in that band.
    """

    def __init__(
        self,
        spectrogram_shape: tuple[int, int],
        n_mels: int,
        T: int,
        min_segments: int = 2,
        max_segments: int = 8,
        band_lo: int = 2,
        band_hi: int = 96,
    ) -> None:
        self.spectrogram_shape = spectrogram_shape
        self.n_mels = n_mels
        self.T = T
        self.min_segments = max(1, min_segments)
        self.max_segments = max(self.min_segments, max_segments)
        self.band_lo = max(1, band_lo)
        self.band_hi = max(self.band_lo, band_hi)

    def _single_mask_numpy(self) -> np.ndarray:
        n_mels, T = self.n_mels, self.T
        M = np.zeros((n_mels, T), dtype=np.float32)

        # Frequency band
        bw = random.randint(self.band_lo, min(self.band_hi, n_mels))
        bw = max(1, min(bw, n_mels))
        f_low = random.randint(0, max(0, n_mels - bw))
        f_high = min(n_mels, f_low + bw)

        # Disjoint time segments inside that band
        n_seg = random.randint(self.min_segments, self.max_segments)
        seg_len_lo = max(MIN_TIME_FRAMES, T // 40)
        seg_len_hi = max(seg_len_lo, min(T, T // 6))
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
            masks.append(torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).unsqueeze(0))
        return torch.cat(masks, dim=0).to(device)


class PerlinNoiseStrategy:
    """
    Simple Perlin-noise anomaly masks.

    Uses :func:`generate_perlin_noise_2d` to generate a (H, W) Perlin field,
    then binarizes with a threshold.
    """

    def __init__(
        self,
        spectrogram_shape: tuple[int, int],
        threshold: float = 0.5,
        res_exponent_lo: int = 0,
        res_exponent_hi: int = 4,
    ) -> None:
        self.spectrogram_shape = spectrogram_shape
        self.threshold = threshold
        self.res_exponent_lo = res_exponent_lo
        self.res_exponent_hi = max(res_exponent_lo, res_exponent_hi)

    def _rand_res(self) -> tuple[int, int]:
        """
        Pick a Perlin resolution (ry, rx) as powers of 2 that (best-effort)
        divide the target shape, to avoid degenerate artifacts.
        """
        H, W = self.spectrogram_shape
        exps = list(range(self.res_exponent_lo, self.res_exponent_hi + 1))
        candidates: list[tuple[int, int]] = []
        for ey in exps:
            ry = 2**ey
            if ry > H:
                continue
            for ex in exps:
                rx = 2**ex
                if rx > W:
                    continue
                if H % ry == 0 and W % rx == 0:
                    candidates.append((ry, rx))
        if candidates:
            return random.choice(candidates)
        # Fallback: any power-of-2 <= shape
        ry = min(2 ** random.choice(exps), max(1, H))
        rx = min(2 ** random.choice(exps), max(1, W))
        return max(1, ry), max(1, rx)

    def _single_mask_numpy(self) -> np.ndarray:
        H, W = self.spectrogram_shape
        ry, rx = self._rand_res()
        noise = generate_perlin_noise_2d((H, W), (ry, rx))  # ~[-1, 1]
        M = (noise > self.threshold).astype(np.float32)
        return M

    def __call__(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        masks = []
        for _ in range(batch_size):
            arr = self._single_mask_numpy()
            masks.append(torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).unsqueeze(0))
        return torch.cat(masks, dim=0).to(device)


class AnomalyMapGenerator:
    """
    Generate anomaly map M for training.

    This implementation always uses :class:`AudioSpecificStrategy` for all
    machine types.
    """

    def __init__(
        self,
        strategy: str,
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
        self.strategy_name = strategy
        self.spectrogram_shape = spectrogram_shape
        self.zero_mask_prob = zero_mask_prob
        if n_mels is None or T is None:
            n_mels, T = spectrogram_shape
        self.perlin = (
            PerlinNoiseStrategy(spectrogram_shape)
            if self.strategy_name in ("perlin", "both")
            else None
        )
        self.audio_specific = AudioSpecificStrategy(spectrogram_shape, n_mels, T)

    def _generate_one(
        self,
        device: torch.device | str,
        machine_type: str | None = None,
    ) -> torch.Tensor:
        """Generate a single non-zero mask (1, 1, H, W)."""
        _ = machine_type  # unused; same generator for all types
        if self.strategy_name == "perlin":
            assert self.perlin is not None
            return self.perlin(1, device)
        if self.strategy_name == "both":
            # Mix Perlin (20%) and audio_specific (80%) for now.
            if random.random() < 0.2:
                assert self.perlin is not None
                return self.perlin(1, device)
            return self.audio_specific(1, device)
        # Default: audio_specific
        return self.audio_specific(1, device)

    def generate_for_training_sample(
        self,
        device: torch.device | str,
        force_anomaly: bool = True,
        machine_type: str | None = None,
    ) -> torch.Tensor:
        """Generate one mask for a single training sample."""
        if force_anomaly:
            return self._generate_one(device, machine_type=machine_type)
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
            _ = machine_types  # unused
            if self.strategy_name == "perlin":
                assert self.perlin is not None
                return self.perlin(batch_size, device)
            if self.strategy_name == "both":
                # Batch-level mix (matches prior behavior of choosing one branch per call).
                if random.random() < 0.2:
                    assert self.perlin is not None
                    return self.perlin(batch_size, device)
                return self.audio_specific(batch_size, device)
            return self.audio_specific(batch_size, device)

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
