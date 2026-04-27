"""
Spectromorphic anomaly mask generation for sDSR training.

One strategy: pick a mel band (stratified Hz → mel), modulate over time
with alternating geometric renewal runs. Perlin noise as an optional
regularizer. Never produces a fully-filled time strip.
"""

from __future__ import annotations

import math
import random
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .perlin import rand_perlin_2d_np


# ---------------------------------------------------------------------------
# Frequency utilities
# ---------------------------------------------------------------------------

def _hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + max(hz, 0.0) / 700.0)


def _hz_band_to_mel_bins(
    f0_hz: float,
    bw_hz: float,
    n_mels: int,
    f_min_hz: float,
    f_max_hz: float,
) -> tuple[int, int]:
    """Map a linear-Hz band to half-open mel-bin indices [i0, i1)."""
    mel_min = _hz_to_mel(f_min_hz)
    mel_span = _hz_to_mel(f_max_hz) - mel_min
    if mel_span <= 0:
        return 0, 1

    def to_bin(hz: float) -> float:
        return (_hz_to_mel(hz) - mel_min) / mel_span * n_mels

    i0 = max(0, min(n_mels - 1, int(math.floor(to_bin(max(f_min_hz, f0_hz))))))
    i1 = max(i0 + 1, min(n_mels, int(math.ceil(to_bin(min(f_max_hz, f0_hz + bw_hz))))))
    return i0, i1


# ---------------------------------------------------------------------------
# Time modulation
# ---------------------------------------------------------------------------

def _alternating_renewal(T: int, p_on: float, p_off: float, start_on: bool) -> np.ndarray:
    """
    Binary (T,) vector from alternating geometric on/off runs.
    Guaranteed to have both 0s and 1s for typical p values.
    """
    out = np.zeros(T, dtype=np.float32)
    t, state = 0, start_on
    while t < T:
        p = p_on if state else p_off
        run = min(max(1, int(np.random.geometric(p))), T - t)
        if state:
            out[t : t + run] = 1.0
        t += run
        state = not state
    return out


def _renewal_params(T: int) -> tuple[float, float]:
    """Log-uniform mean run lengths in [10, T-1], returned as geometric p values."""
    lo, hi = math.log(10.0), math.log(max(11.0, T - 1.0))
    mu_on  = math.exp(random.uniform(lo, hi))
    mu_off = math.exp(random.uniform(lo, hi))
    eps = 1e-6
    p_on  = float(np.clip(1.0 / mu_on,  eps, 1 - eps))
    p_off = float(np.clip(1.0 / mu_off, eps, 1 - eps))
    return p_on, p_off


# ---------------------------------------------------------------------------
# Perlin regularizer
# ---------------------------------------------------------------------------

def _perlin_mask(n_mels: int, T: int) -> np.ndarray:
    """
    Thresholded 2-D Perlin noise mask (binary float32), aligned with the common
    """
    min_perlin_scale = 3
    perlin_scale = 6  # randint in [0, 5] -> scales in {1,2,4,8,16,32}
    perlin_scaley = 2 ** int(random.randint(min_perlin_scale, perlin_scale - 1))

    scaley_exp = int(math.log2(perlin_scaley))

    perlin_scalex = 2 ** int(random.randint(scaley_exp, perlin_scale - 1))
    perlin_scalex = max(1, min(perlin_scalex, T))
    perlin_scaley = max(1, min(perlin_scaley, n_mels))

    noise = rand_perlin_2d_np((n_mels, T), (perlin_scaley, perlin_scalex))

    # Optional flips (still dependency-free; helps mask variety)
    if random.random() < 0.5:
        noise = np.flip(noise, axis=0)  # freq flip
    if random.random() < 0.5:
        noise = np.flip(noise, axis=1)  # time flip

    threshold = float(np.percentile(noise, random.uniform(55, 80)))
    perlin_thr = (noise > threshold).astype(np.float32)
    return perlin_thr


# ---------------------------------------------------------------------------
# Frequency band sampler
# ---------------------------------------------------------------------------

# Stratified Hz bands (low / mid / high). Keeps coverage of the full spectrum.
_HZ_STRATA: list[tuple[float, float]] = [(0.0, 1000.0), (1000.0, 3000.0), (3000.0, 8000.0)]


def _sample_mel_band(
    n_mels: int,
    f_min_hz: float,
    f_max_hz: float,
    bw_min_hz: float,
    bw_max_hz: float,
    max_tries: int = 32,
) -> tuple[int, int] | None:
    """
    Sample one mel band via stratified Hz selection.
    Returns (i0, i1) or None if no valid band found within max_tries.
    """
    for _ in range(max_tries):
        b_lo, b_hi = random.choice(_HZ_STRATA)
        lo = max(f_min_hz, b_lo)
        hi = min(f_max_hz, b_hi)
        if hi - lo < 1.0:
            continue
            
        bw = random.uniform(
            min(bw_min_hz, hi - lo),
            min(bw_max_hz, hi - lo),
        )
        f0 = random.uniform(lo, hi - bw)
        i0, i1 = _hz_band_to_mel_bins(f0, bw, n_mels, f_min_hz, f_max_hz)
        if i1 > i0:
            return i0, i1
    return None


# ---------------------------------------------------------------------------
# Main strategy
# ---------------------------------------------------------------------------

class SpectromorphicMaskStrategy:
    """
    Spectromorphic anomaly masks: a mel band modulated over time.

    Each mask is one of:
      - **Band + renewal** (prob ``1 - perlin_prob``): a stratified Hz band
        mapped to mel bins, activated over time by alternating geometric runs.
        Never a fully continuous strip — the renewal process always creates gaps.
      - **Perlin** (prob ``perlin_prob``): thresholded 2-D Perlin noise for
        blob-shaped masks.

    Args:
        n_mels: mel bins in the spectrogram.
        T: time frames in the spectrogram.
        q_shape: output spatial shape; masks are interpolated if it differs from (n_mels, T).
        perlin_prob: probability of the Perlin branch per mask.
        f_min_hz, f_max_hz: mel filterbank frequency range (Hz).
        bw_min_hz, bw_max_hz: uniform range for band width (Hz).
    """

    # Hard fallback: a very narrow partial-time band used only if _sample_mel_band fails.
    _FALLBACK_BW_HZ = 40.0

    def __init__(
        self,
        n_mels: int = 128,
        T: int = 320,
        q_shape: tuple[int, int] | None = None,
        perlin_prob: float = 0.15,
        f_min_hz: float = 0.0,
        f_max_hz: float = 8_000.0,
        bw_min_hz: float = 40.0,
        bw_max_hz: float = 1_000.0,
        **_: object,
    ) -> None:
        self.n_mels = n_mels
        self.T = T
        self.q_shape = q_shape or (n_mels, T)
        self.perlin_prob = float(np.clip(perlin_prob, 0.0, 1.0))
        self.f_min_hz = f_min_hz
        self.f_max_hz = f_max_hz
        self.bw_min_hz = bw_min_hz
        self.bw_max_hz = bw_max_hz

    # -- mask builders -------------------------------------------------------

    def _band_mask(self) -> np.ndarray:
        """Mel band × renewal-modulated time vector."""
        mask = np.zeros((self.n_mels, self.T), dtype=np.float32)

        # band = _sample_mel_band(
        #     self.n_mels, self.f_min_hz, self.f_max_hz, self.bw_min_hz, self.bw_max_hz
        # )
        # if band is None:
        #     # Hard fallback: tiny band, partial time via a single renewal
        #     band = _hz_band_to_mel_bins(
        #         self.f_min_hz, self._FALLBACK_BW_HZ,
        #         self.n_mels, self.f_min_hz, self.f_max_hz,
        #     )

        # i0, i1 = band

        # rng = np.random.default_rng()
        # band_width = rng.integers(6, 50 + 1)
        # f_start = rng.integers(0, self.n_mels - band_width + 1)
        # f_end = f_start + band_width

        # i0 = f_start
        # i1 = f_end

        # p_on, p_off = _renewal_params(self.T)
        # mu_on = 1.0 / p_on
        # start_on = random.random() < mu_on / (mu_on + 1.0 / p_off)
        # time_track = _alternating_renewal(self.T, p_on, p_off, start_on)
        # mask[i0:i1, :] = time_track

        # Sample K independent contiguous time segments
        # n_segments = random.randint(1, 4)
        # for _ in range(n_segments):
        #     seg_len = random.randint(self.T // 20, self.T // 4 + 1)
        #     t_start = random.randint(0, max(0, self.T - seg_len))
        #     mask[i0:i1, t_start : t_start + seg_len] = 1.0

        rng = np.random.default_rng()

        # ── Coarse grid dimensions ────────────────────────────────────────────────
        C_F = self.n_mels // 8          # 16 freq cells
        C_T = self.T // 8               # 40 time cells
        cell_f = self.n_mels // C_F     # 8  mel bins per cell
        cell_t = self.T // C_T          # 8  time frames per cell

        # ── Step 1: frequency band in coarse cells ───────────────────────────────
        # Min 2 coarse cells (16 mel bins) so band survives both fine and coarse projection.
        # Max 6 coarse cells (48 mel bins, ~37% of axis) keeps anomaly localised.
        band_h_cells = int(rng.integers(2, 7))                          # [2, 6] cells
        band_lo_cell = int(rng.integers(0, C_F - band_h_cells + 1))
        band_hi_cell = band_lo_cell + band_h_cells                      # exclusive

        band_lo = band_lo_cell * cell_f
        band_hi = band_hi_cell * cell_f

        # ── Step 2: time segments in coarse cells ────────────────────────────────
        # Partition the time axis into N coarse-cell-aligned segments, then
        # activate a contiguous run within each. Min run = 2 cells so the activated
        # region spans ≥1 fine cell and ≥1 coarse cell unambiguously.
        num_segs = int(rng.integers(1, 5))                              # [1, 4] segments

        # Draw segment boundaries as coarse cell indices
        if num_segs == 1:
            cut_cells = []
        else:
            cut_cells = sorted(
                rng.choice(np.arange(1, C_T), size=min(num_segs - 1, C_T - 1), replace=False).tolist()
            )
        seg_boundaries = [0] + cut_cells + [C_T]
        segments_cells = [
            (seg_boundaries[i], seg_boundaries[i + 1])
            for i in range(len(seg_boundaries) - 1)
        ]

        # ── Step 3: activate a run within each segment ───────────────────────────
        # Run occupies [aug_lo, aug_hi] fraction of the segment, minimum 2 cells.
        # aug_lo/hi sampled once per mask for temporal coherence across segments.
        aug_lo_frac = rng.uniform(0.25, 0.55)
        aug_hi_frac = rng.uniform(aug_lo_frac + 0.15, min(aug_lo_frac + 0.55, 1.0))

        mask = np.zeros((self.n_mels, self.T), dtype=np.float32)

        for seg_lo_c, seg_hi_c in segments_cells:
            seg_len_c = seg_hi_c - seg_lo_c
            if seg_len_c < 2:
                # Segment too short for a meaningful run — skip rather than inject noise
                continue

            run_len_c = int(np.clip(
                rng.integers(
                    max(2, int(aug_lo_frac * seg_len_c)),
                    max(3, int(aug_hi_frac * seg_len_c) + 1),
                ),
                2, seg_len_c,
            ))
            run_start_c = int(rng.integers(0, seg_len_c - run_len_c + 1))

            t_lo = (seg_lo_c + run_start_c) * cell_t
            t_hi = t_lo + run_len_c * cell_t

            mask[band_lo:band_hi, t_lo:t_hi] = 1.0

        # Fallback: if all segments were skipped (e.g. very short T), activate
        # a single central 2×4 cell block as a minimal valid mask.
        if mask.sum() == 0:
            f_c = C_F // 2
            t_c = C_T // 2
            mask[f_c * cell_f:(f_c + 2) * cell_f, t_c * cell_t:(t_c + 4) * cell_t] = 1.0

        return mask

    def _perlin_mask(self) -> np.ndarray:
        return _perlin_mask(self.n_mels, self.T)

    # -- public interface ----------------------------------------------------

    def __call__(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        """Return ``(B, 1, *q_shape)`` binary float32 mask tensor."""
        masks = [
            torch.from_numpy(
                self._perlin_mask() if random.random() < self.perlin_prob else self._band_mask()
            ).unsqueeze(0).unsqueeze(0)
            for _ in range(batch_size)
        ]
        M = torch.cat(masks, dim=0).to(device)
        if self.q_shape != (self.n_mels, self.T):
            M = F.interpolate(M, size=self.q_shape, mode="nearest")
        return M


# ---------------------------------------------------------------------------
# AnomalyMapGenerator — training entry point
# ---------------------------------------------------------------------------

class AnomalyMapGenerator:
    """
    Wraps :class:`SpectromorphicMaskStrategy` for training-loop use.

    Args:
        spectrogram_shape: ``(n_mels, T)`` of the model input.
        q_shape: output shape (defaults to ``spectrogram_shape``).
        **strategy_kwargs: forwarded to :class:`SpectromorphicMaskStrategy`.
    """

    def __init__(
        self,
        spectrogram_shape: tuple[int, int],
        q_shape: tuple[int, int] | None = None,
        **strategy_kwargs,
    ) -> None:
        n_mels, T = spectrogram_shape
        self.spectrogram_shape = spectrogram_shape
        self.q_shape = q_shape or spectrogram_shape
        self._strategy = SpectromorphicMaskStrategy(
            n_mels=n_mels, T=T, q_shape=self.q_shape, **strategy_kwargs
        )

    def generate(
        self,
        batch_size: int,
        device: torch.device | str,
    ) -> torch.Tensor:
        """
        Generate ``(B, 1, *q_shape)`` anomaly masks.

        This generator **only** produces anomaly masks (non-zero). Sampling
        normal/zero masks is handled at the dataset level.
        """
        return self._strategy(batch_size, device)

    def generate_for_training_sample(
        self,
        device: torch.device | str,
    ) -> torch.Tensor:
        """Convenience wrapper for a single sample."""
        return self.generate(1, device)