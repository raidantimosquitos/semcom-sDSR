"""
Spectromorphic anomaly mask generation for sDSR training.

One strategy: pick a mel band (uniform Hz → mel), modulate over time
with alternating geometric renewal runs. Perlin noise as an optional
regularizer. Never produces a fully-filled time strip.
"""

from __future__ import annotations

import math
import random

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
# Perlin regularizer
# ---------------------------------------------------------------------------

def _shear_perlin_freq_axis(noise: np.ndarray, shear: float) -> np.ndarray:
    """
    Shear the Perlin field along the frequency (mel) axis as a function of time.
    shear ∈ [-0.3, 0.3] shifts each time column by shear * col_index rows.
    Implemented as a remap; out-of-bounds filled by reflection.
    """
    h, w = noise.shape
    col_idx = np.arange(w, dtype=np.float32)
    row_shift = shear * col_idx  # (W,) pixel shift per column
    # Build sampling grid
    rows = np.arange(h, dtype=np.float32)[:, None] - row_shift[None, :]  # (H, W)
    rows = rows % h  # reflection via modulo (periodic)
    cols = np.tile(np.arange(w, dtype=np.float32)[None, :], (h, 1))
    from scipy.ndimage import map_coordinates
    return map_coordinates(noise, [rows, cols], order=1, mode='wrap').astype(np.float32)


def _perlin_mask(
    n_mels: int,
    T: int,
    *,
    min_scale_exp: int = 0,
    max_scale_exp: int = 6,
    threshold_beta: float = 0.4,
    shear_range: tuple[float, float] | None = (-0.25, 0.25),
) -> np.ndarray:
    """
    Thresholded 2-D Perlin mask (binary float32).

    Matches 3DSR ``generate_perlin_noise`` (``_tmp_3dsr/data_loader.py``):
    randomized threshold ``τ ∈ [β, 2β)`` with default ``β=0.4``, and
    binarization ``|noise| > τ``. Optionally rotates the noise (uniform angle
    in ``rotate_deg_range``) via :func:`_rotate_perlin_reflect_torch`.
    """
    # perlin_scaley = 2 ** int(random.randint(min_scale_exp, max_scale_exp))
    # perlin_scalex = 2 ** int(random.randint(min_scale_exp, max_scale_exp))
    perlin_scaley = 2 ** int(random.randint(0, 2))
    perlin_scalex = 2 ** int(random.randint(3, 6))

    noise = rand_perlin_2d_np((n_mels, T), (perlin_scaley, perlin_scalex)).astype(np.float32)

    if random.random() < 0.5 and shear_range is not None:
        shear = random.uniform(*shear_range)
        noise = _shear_perlin_freq_axis(noise, shear)

    tau   = random.uniform(threshold_beta, 2 * threshold_beta)
    sign  = random.choice([1, -1])
    perlin_bin = (sign * noise > tau).astype(np.float32)

    if perlin_bin.sum() == 0:
        perlin_bin = np.zeros((n_mels, T), dtype=np.float32)
        bw_start = random.randint(0, n_mels - 1)
        bw_end = max(bw_start + random.randint(1, 64), n_mels - 1)
        t_start = random.randint(0, T - 1)
        t_end = max(t_start + random.randint(1, 160), T - 1)
        perlin_bin[bw_start:bw_end, t_start:t_end] = 1.0

    return perlin_bin


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

# ---------------------------------------------------------------------------
# Main strategy
# ---------------------------------------------------------------------------

class SpectromorphicMaskStrategy:
    """
    Spectromorphic anomaly masks: a mel band modulated over time.

    Each mask is one of:
      - **Band + renewal** (prob ``1 - perlin_prob``): one or more **disjoint**
        mel strips (log-uniform height each, non-overlapping rows; gaps allowed), each with its own Dirichlet time
        segments and Beta-distributed runs (same structure as the legacy single
        band). If the marked fraction exceeds ``band_mask_max_coverage``, ones
        are randomly thinned in-place until at or below that fraction.
      - **Perlin** (prob ``perlin_prob``): thresholded 2-D Perlin noise for
        blob-shaped masks.

    Args:
        n_mels: mel bins in the spectrogram.
        T: time frames in the spectrogram.
        q_shape: output spatial shape; masks are interpolated if it differs from (n_mels, T).
        perlin_prob: probability of the Perlin branch per mask.
        perlin_threshold_beta: threshold scale ``β``; each mask draws ``τ ∈ [β, 2β)`` and sets
            ones where ``|noise| > τ`` (same rule as 3DSR ``generate_perlin_noise``).
        perlin_rotate_deg_range: if not ``None``, rotate the noise by a uniform angle in this
            range (degrees) via ``grid_sample(..., padding_mode='reflection')`` before
            thresholding; set to ``None`` to disable rotation.
        f_min_hz, f_max_hz: mel filterbank frequency range (Hz).
        bw_min_hz, bw_max_hz: uniform range for band width (Hz).
        band_mask_max_bands: upper bound on how many disjoint mel bands to attempt (actual count is
            uniform in ``1..max_bands``, then each band is placed with rejection until no overlap;
            fewer bands remain if placement fails).
        band_mask_max_coverage: if the fraction of strictly positive entries exceeds this value,
            random marked pixels are cleared until the fraction is at most this value.
            Set to ``1.0`` to disable thinning.
    """

    # Hard fallback: a very narrow partial-time band used only if _sample_mel_band fails.
    _FALLBACK_BW_HZ = 40.0

    def __init__(
        self,
        n_mels: int = 128,
        T: int = 320,
        q_shape: tuple[int, int] | None = None,
        perlin_prob: float = 0.4,
        mixed_prob: float = 0.25,
        perlin_threshold_beta: float = 0.4,
        perlin_shear_range: tuple[float, float] | None = (-0.25, 0.25),
        n_bands: int | None = None,
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
        self.mixed_prob = float(np.clip(mixed_prob, 0.0, 1.0))
        self.perlin_threshold_beta = float(perlin_threshold_beta)
        self.perlin_shear_range = perlin_shear_range
        self.f_min_hz = f_min_hz
        self.n_bands = n_bands
        self.f_max_hz = f_max_hz
        self.bw_min_hz = bw_min_hz
        self.bw_max_hz = bw_max_hz


    def _band_mask(self, n_bands: int | None = None) -> np.ndarray:
        """Mel band × renewal-modulated time vector."""
        mask = np.zeros((self.n_mels, self.T), dtype=np.float32)
        nb = n_bands if n_bands is not None else random.randint(1, 3)
        for _ in range(nb):
            band_h = random.randint(2, max(3, self.n_mels // 8))   # 2..16 mel rows
            center  = random.randint(band_h // 2, self.n_mels - band_h // 2)
            r0, r1  = max(0, center - band_h // 2), min(self.n_mels, center + band_h // 2)
            if random.random() < 0.5:                          # time crop
                t0 = random.randint(0, self.T // 2)
                t1 = random.randint(self.T // 2, self.T)
            else:
                t0, t1 = 0, self.T
            mask[r0:r1, t0:t1] = 1.0
        return mask



        # ---------------------------------------------------------------------
        # Old band_mask implementation (kept for reference)
        # ---------------------------------------------------------------------
        min_band_frac: float = 0.01
        max_band_frac: float = 1.0 # 1.0
        
        # Step 1: frequency band (domain-constrained bounds stay fixed)
        band_h = random.randint(
            max(1, int(min_band_frac * self.n_mels)),
            max(1, int(max_band_frac * self.n_mels)),
        )
        band_lo = random.randint(0, self.n_mels - band_h)
        band_hi = band_lo + band_h
        
        i0, i1 = band_lo, band_hi

        # ── Step 2: time segments in coarse cells ────────────────────────────
        num_segs = int(random.randint(1, 5))

        if band_h > 64:
            min_aug_frac = 0.05 # 0.1
            max_aug_frac = 0.4 # 1.0
        else:
            min_aug_frac = 0.4
            max_aug_frac = 1.0

        # min_aug_frac = 0.05
        # max_aug_frac = 1.0

        # Draw (num_segs - 1) unique interior cut points, then sort
        # cut_points = sorted(
        #    random.sample(range(1, self.T), min(num_segs - 1, self.T - 1))
        #)
        #boundaries = [0] + cut_points + [self.T]
        #segments = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]

        # Evenly split [0, T) into num_segs segments (integer division spreads remainder).
        boundaries = [i * self.T // num_segs for i in range(num_segs + 1)]
        segments = [(boundaries[i], boundaries[i + 1]) for i in range(num_segs)]
    
        # ── Step 3: augment a random consecutive run within each segment ─────
        for seg_start, seg_end in segments:
            seg_len = seg_end - seg_start
            if seg_len < 1:
                continue
    
            run_len = random.randint(
                max(1, int(min_aug_frac * seg_len)),
                max(1, int(max_aug_frac * seg_len)),
            )
            run_start = random.randint(0, seg_len - run_len)
            mask[i0:i1, seg_start + run_start : seg_start + run_start + run_len] = 1.0

        return mask

    def _mixed_mask(self) -> np.ndarray:
        band = self._band_mask(n_bands=1)
        perlin = self._perlin_mask()
        mixed = (perlin * band).astype(np.float32)
        if mixed.sum() == 0:
            mixed = (perlin + band).astype(np.float32)
        return mixed


    def _perlin_mask(self) -> np.ndarray:
        return _perlin_mask(
            self.n_mels,
            self.T,
            threshold_beta=self.perlin_threshold_beta,
            shear_range=self.perlin_shear_range,
        )

    # -- public interface ----------------------------------------------------

    def __call__(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        """Return ``(B, 1, *q_shape)`` binary float32 mask tensor."""
        masks = [
            torch.from_numpy(
                self._mixed_mask() if random.random() < self.mixed_prob else self._perlin_mask() if random.random() < self.perlin_prob else self._band_mask()
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