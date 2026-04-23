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
    """Thresholded 2-D Perlin noise, binary output. Falls back to soft noise if empty."""
    res_y = max(2, min(2 ** random.randint(1, 4), n_mels))
    res_x = max(2, min(2 ** random.randint(1, 4), T))
    noise = rand_perlin_2d_np((n_mels, T), (res_y, res_x))
    mask = (noise > random.uniform(0.3, 0.6)).astype(np.float32)
    return mask if mask.sum() > 0 else noise.astype(np.float32)


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

        band = _sample_mel_band(
            self.n_mels, self.f_min_hz, self.f_max_hz, self.bw_min_hz, self.bw_max_hz
        )
        if band is None:
            # Hard fallback: tiny band, partial time via a single renewal
            band = _hz_band_to_mel_bins(
                self.f_min_hz, self._FALLBACK_BW_HZ,
                self.n_mels, self.f_min_hz, self.f_max_hz,
            )

        i0, i1 = band
        p_on, p_off = _renewal_params(self.T)
        mu_on = 1.0 / p_on
        start_on = random.random() < mu_on / (mu_on + 1.0 / p_off)
        time_track = _alternating_renewal(self.T, p_on, p_off, start_on)
        mask[i0:i1, :] = time_track
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
        zero_mask_prob: probability of an all-zero (normal) mask when
            ``force_anomaly=False``.
        **strategy_kwargs: forwarded to :class:`SpectromorphicMaskStrategy`.
    """

    def __init__(
        self,
        spectrogram_shape: tuple[int, int],
        q_shape: tuple[int, int] | None = None,
        zero_mask_prob: float = 0.5,
        **strategy_kwargs,
    ) -> None:
        n_mels, T = spectrogram_shape
        self.spectrogram_shape = spectrogram_shape
        self.q_shape = q_shape or spectrogram_shape
        self.zero_mask_prob = zero_mask_prob
        self._strategy = SpectromorphicMaskStrategy(
            n_mels=n_mels, T=T, q_shape=self.q_shape, **strategy_kwargs
        )

    def generate(
        self,
        batch_size: int,
        device: torch.device | str,
        force_anomaly: bool = False,
    ) -> torch.Tensor:
        """
        Generate ``(B, 1, *q_shape)`` anomaly masks.

        If ``force_anomaly=True``, all masks are non-zero.
        Otherwise each is zeroed with probability ``zero_mask_prob``.
        """
        if force_anomaly:
            return self._strategy(batch_size, device)

        masks = []
        for _ in range(batch_size):
            if random.random() < self.zero_mask_prob:
                masks.append(torch.zeros(1, 1, *self.spectrogram_shape, device=device))
            else:
                masks.append(self._strategy(1, device))
        return torch.cat(masks, dim=0)

    def generate_for_training_sample(
        self,
        device: torch.device | str,
        force_anomaly: bool = True,
    ) -> torch.Tensor:
        """Convenience wrapper for a single sample."""
        return self.generate(1, device, force_anomaly=force_anomaly)