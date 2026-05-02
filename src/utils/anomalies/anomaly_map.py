"""
Spectromorphic anomaly mask generation for sDSR training.

One strategy: pick a mel band (uniform Hz → mel), modulate over time
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
from scipy.ndimage import rotate as nd_rotate

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
    min_perlin_scale = 0
    perlin_scale = 6  # randint in [0, 5] -> scales in {1,2,4,8,16,32}
    angle_deg = random.uniform(-15.0, 15.0)
    
    perlin_scaley = 2 ** int(random.randint(min_perlin_scale, perlin_scale - 1))

    scaley_exp = int(math.log2(perlin_scaley))

    perlin_scalex = 2 ** int(random.randint(scaley_exp, perlin_scale - 1))
    perlin_scalex = max(1, min(perlin_scalex, T))
    perlin_scaley = max(1, min(perlin_scaley, n_mels))

    noise = rand_perlin_2d_np((n_mels, T), (perlin_scaley, perlin_scalex))
    noise = nd_rotate(noise, angle_deg, axes=(0, 1), reshape=False)

    threshold = 0.5
    perlin_thr = (noise > threshold).astype(np.float32)
    return perlin_thr


# ---------------------------------------------------------------------------
# Frequency band sampler
# ---------------------------------------------------------------------------

def _sample_mel_band(
    n_mels: int,
    f_min_hz: float,
    f_max_hz: float,
    bw_min_hz: float,
    bw_max_hz: float,
    max_tries: int = 32,
) -> tuple[int, int] | None:
    """
    Sample one mel band via uniform Hz selection.
    Returns (i0, i1) or None if no valid band found within max_tries.
    """
    for _ in range(max_tries):
        lo = float(f_min_hz)
        hi = float(f_max_hz)
        if hi - lo < 2.0:
            continue

        # Uniformly sample bandwidth, then uniformly sample its start frequency.
        max_bw = max(1.0, hi - lo)
        bw_lo = float(np.clip(min(bw_min_hz, bw_max_hz), 1.0, max_bw))
        bw_hi = float(np.clip(max(bw_min_hz, bw_max_hz), bw_lo, max_bw))
        bw = random.uniform(bw_lo, bw_hi)

        # Ensure f0 is valid even in edge cases where bw ≈ (hi - lo).
        f0_hi = hi - bw
        if f0_hi <= lo:
            f0 = lo
        else:
            f0 = random.uniform(lo, f0_hi)
        i0, i1 = _hz_band_to_mel_bins(f0, bw, n_mels, f_min_hz, f_max_hz)
        if i1 > i0:
            return i0, i1
    return None

def _smoothstep(t: np.ndarray) -> np.ndarray:
    """Perlin fade: 6t^5 - 15t^4 + 10t^3"""
    return t * t * t * (t * (t * 6 - 15) + 10)

def perlin_1d(n: int, n_octaves: int = 3, persistence: float = 0.5,
              lacunarity: float = 2.0, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Pure-numpy 1D Perlin noise of length n.
    Produces smooth values in approximately [-1, 1].

    n_octaves   : more octaves → finer detail on the freq axis
    persistence : amplitude decay per octave (< 1 → coarser dominates)
    lacunarity  : frequency multiplier per octave
    """
    if rng is None:
        rng = np.random.default_rng()

    result    = np.zeros(n, dtype=np.float64)
    amplitude = 1.0
    frequency = 1.0

    for _ in range(n_octaves):
        # Number of grid cells at this octave
        n_cells = max(2, int(np.ceil(n * frequency / n)))
        # Random gradient at each grid vertex: +1 or -1
        n_verts = n_cells + 1
        grads   = rng.choice([-1.0, 1.0], size=n_verts)

        # Sample positions in [0, n_cells]
        xs     = np.linspace(0, n_cells, n, endpoint=False)
        x0     = np.floor(xs).astype(int)
        x1     = x0 + 1
        t      = xs - x0                         # local coordinate in [0, 1)
        fade_t = _smoothstep(t)

        # Clamp to valid gradient indices
        x0 = np.clip(x0, 0, n_verts - 1)
        x1 = np.clip(x1, 0, n_verts - 1)

        # Dot products: gradient × distance to vertex
        d0 = grads[x0] * t           # distance from left vertex
        d1 = grads[x1] * (t - 1.0)  # distance from right vertex

        # Interpolate
        result += amplitude * (d0 + fade_t * (d1 - d0))

        amplitude *= persistence
        frequency *= lacunarity

    return result


# ---------------------------------------------------------------------------
# Main strategy
# ---------------------------------------------------------------------------

class SpectromorphicMaskStrategy:
    """
    Spectromorphic anomaly masks: a mel band modulated over time.

    Each mask is one of:
      - **Band + renewal** (prob ``1 - perlin_prob``): a uniform Hz band
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
        perlin_prob: float = 0.2,
        f_min_hz: float = 0.0,
        f_max_hz: float = 8_000.0,
        bw_min_hz: float = 40.0,
        bw_max_hz: float = 2_000.0,
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

        # ---------------------------------------------------------------------
        # Old band_mask implementation (kept for reference)
        # ---------------------------------------------------------------------
        # min_band_frac: float = 0.6
        # max_band_frac: float = 1.0
        
        # # Step 1: frequency band (domain-constrained bounds stay fixed)
        # band_h = random.randint(
        #     max(1, int(min_band_frac * self.n_mels)),
        #     max(1, int(max_band_frac * self.n_mels)),
        # )
        # band_lo = random.randint(0, self.n_mels - band_h)
        # band_hi = band_lo + band_h
    
        # i0, i1 = band_lo, band_hi
    
        # # ── Step 2: time segments in coarse cells ────────────────────────────
        # num_segs = int(random.randint(1, 6))
        # min_aug_frac = 0.01
        # max_aug_frac = 0.05
    
        # # Draw (num_segs - 1) unique interior cut points, then sort
        # cut_points = sorted(
        #     random.sample(range(1, self.T), min(num_segs - 1, self.T - 1))
        # )
        # boundaries = [0] + cut_points + [self.T]
        # segments = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
    
        # # ── Step 3: augment a random consecutive run within each segment ─────
        # for seg_start, seg_end in segments:
        #     seg_len = seg_end - seg_start
        #     if seg_len < 1:
        #         continue
    
        #     run_len = random.randint(
        #         max(1, int(min_aug_frac * seg_len)),
        #         max(1, int(max_aug_frac * seg_len)),
        #     )
        #     run_start = random.randint(0, seg_len - run_len)
        #     mask[i0:i1, seg_start + run_start : seg_start + run_start + run_len] = 1.0

        rng = np.random.default_rng(None)
        threshold    = 0.15   # fraction of freq rows to activate
        n_octaves    = 3      # smoothness of freq-axis noise
        persistence  = 0.5
        lacunarity   = 2.0
        # Time-axis structure
        time_width   = random.uniform(0.5, 1.0)    # 1.0 = full width; < 1.0 = partial band
        time_jitter  = random.uniform(0.0, 0.1)    # per-row random time offset (0 = aligned)
        # Random list of 
        n_harmonics = random.randint(1, 4)
        harmonic_rows =  random.sample(range(self.n_mels), n_harmonics)
        harmonic_boost = 0.4
        harmonic_sigma = 2.0

        # ── Step 1: smooth noise along frequency axis ──────────────────
        noise = perlin_1d(self.n_mels, n_octaves=n_octaves, persistence=persistence,
                        lacunarity=lacunarity, rng=rng)

        # Normalize to [0, 1]
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)

        # ── Step 2: harmonic row bias ──────────────────────────────────
        if harmonic_rows is not None:
            freq_axis = np.arange(self.n_mels, dtype=np.float32)
            bias      = np.zeros(self.n_mels, dtype=np.float32)
            for r in harmonic_rows:
                bias += harmonic_boost * np.exp(
                    -0.5 * ((freq_axis - r) / harmonic_sigma) ** 2
                )
            noise = np.clip(noise + bias, 0, None)
            noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)

        # ── Step 3: threshold → active frequency rows ─────────────────
        cutoff      = np.quantile(noise, 1.0 - threshold)
        active_rows = (noise >= cutoff)   # (self.n_mels,) bool

        # ── Step 4: expand rows to 2D with time structure ──────────────
        mask = np.zeros((self.n_mels, self.T), dtype=np.float32)

        for h in np.where(active_rows)[0]:
            if time_width >= 1.0:
                mask[h, :] = 1.0
            else:
                w      = max(1, int(self.T * time_width))
                if time_jitter > 0:
                    # Each row's band starts at a slightly different time offset
                    jitter = int(rng.uniform(-time_jitter * self.T,
                                            time_jitter * self.T))
                    t_start = int(np.clip(
                        rng.integers(0, self.T - w + 1) + jitter, 0, self.T - w
                    ))
                else:
                    t_start = rng.integers(0, self.T - w + 1)
                mask[h, t_start:t_start + w] = 1.0

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