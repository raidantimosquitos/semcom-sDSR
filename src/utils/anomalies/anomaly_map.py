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

def _rotate_perlin_reflect_torch(noise: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rotate 2-D Perlin field about the center using bilinear sampling with
    ``padding_mode='reflection'``.

    Compared to SciPy ``ndimage.rotate(..., reshape=False, cval=0)``, reflection
    padding avoids collapsing energy toward the center from zero-filled corners on
    non-square maps (e.g. 128×320). Does not depend on torchvision.
    """
    if abs(angle_deg) < 1e-6:
        return noise.astype(np.float32, copy=False)

    h, w = int(noise.shape[0]), int(noise.shape[1])
    t = torch.from_numpy(noise.astype(np.float32)).view(1, 1, h, w)
    rad = math.radians(angle_deg)
    cos_t, sin_t = math.cos(rad), math.sin(rad)

    ys = torch.linspace(-1.0, 1.0, h, dtype=t.dtype, device=t.device)
    xs = torch.linspace(-1.0, 1.0, w, dtype=t.dtype, device=t.device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")

    # Inverse warp: rotate sampling coordinates by +angle so the image rotates CCW.
    sx = gx * cos_t + gy * sin_t
    sy = -gx * sin_t + gy * cos_t
    grid = torch.stack((sx, sy), dim=-1).unsqueeze(0)

    out = F.grid_sample(
        t,
        grid,
        mode="bilinear",
        padding_mode="reflection",
        align_corners=True,
    )
    return out[0, 0].detach().numpy().astype(np.float32)


def _perlin_mask(
    n_mels: int,
    T: int,
    *,
    min_scale_exp: int = 0,
    max_scale_exp: int = 6,
    threshold_beta: float = 0.4,
    rotate_deg_range: tuple[float, float] | None = (-90.0, 90.0),
) -> np.ndarray:
    """
    Thresholded 2-D Perlin mask (binary float32).

    Matches 3DSR ``generate_perlin_noise`` (``_tmp_3dsr/data_loader.py``):
    randomized threshold ``τ ∈ [β, 2β)`` with default ``β=0.4``, and
    binarization ``|noise| > τ``. Optionally rotates the noise (uniform angle
    in ``rotate_deg_range``) via :func:`_rotate_perlin_reflect_torch`.
    """
    perlin_scaley = 2 ** int(random.randint(min_scale_exp, max_scale_exp))
    perlin_scalex = 2 ** int(random.randint(min_scale_exp, max_scale_exp))

    noise = rand_perlin_2d_np((n_mels, T), (perlin_scaley, perlin_scalex)).astype(np.float32)

    if rotate_deg_range is not None:
        lo, hi = rotate_deg_range
        angle_deg = random.uniform(lo, hi)
        noise = _rotate_perlin_reflect_torch(noise, angle_deg)

    beta = float(threshold_beta)
    threshold = random.random() * beta + beta
    return (np.abs(noise) > threshold).astype(np.float32)


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
        perlin_prob: float = 0.1,
        perlin_threshold_beta: float = 0.4,
        perlin_rotate_deg_range: tuple[float, float] | None = (-90.0, 90.0),
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
        self.perlin_threshold_beta = float(perlin_threshold_beta)
        self.perlin_rotate_deg_range = perlin_rotate_deg_range
        self.f_min_hz = f_min_hz
        self.f_max_hz = f_max_hz
        self.bw_min_hz = bw_min_hz
        self.bw_max_hz = bw_max_hz


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

        # # Log-uniform band width — one draw, mirrors 2^randint(min_scale, max_scale)
        # rng = np.random.default_rng()
        # num_bands_range = (1, 4)
        # bw_scale_range = (0, 6)
        # num_segs_range = (1, 5)
        # max_aug_frac = 1.0
        # min_aug_frac = 0.05

        # # Step 1: partition Y-axis into num_bands non-overlapping cells
        # num_bands = int(rng.integers(num_bands_range[0], num_bands_range[1] + 1))
        # y_boundaries = [i * self.n_mels // num_bands for i in range(num_bands + 1)]

        # for b in range(num_bands):
            # cell_y0  = y_boundaries[b]
            # cell_y1  = y_boundaries[b + 1]
            # cell_h   = cell_y1 - cell_y0
            # if cell_h < 1:
                # continue

            # # Step 2: within each Y-cell, sample an independent band
            # max_exp  = max(bw_scale_range[0], min(bw_scale_range[1],
                        # int(np.floor(np.log2(cell_h)))))
            # bw       = int(2 ** rng.integers(bw_scale_range[0], max_exp + 1))
            # bw       = min(bw, cell_h)
            # y0       = int(rng.integers(cell_y0, cell_y1 - bw + 1))

            # # Step 3: partition time axis into num_segs cells, independently per band
            # num_segs    = int(rng.integers(num_segs_range[0], num_segs_range[1] + 1))
            # x_boundaries = [i * self.T // num_segs for i in range(num_segs + 1)]

            # # Step 4: within each time cell, sample one contiguous run
            # for s in range(num_segs):
                # seg_start = x_boundaries[s]
                # seg_end   = x_boundaries[s + 1]
                # seg_len   = seg_end - seg_start
                # if seg_len < 1:
                    # continue

                # run_len   = int(rng.integers(
                    # max(1, int(min_aug_frac * seg_len)),
                    # max(1, int(max_aug_frac * seg_len)) + 1,
                # ))
                # run_start = int(rng.integers(0, max(1, seg_len - run_len + 1)))
                # mask[y0:y0 + bw, seg_start + run_start:seg_start + run_start + run_len] = 1.0
        
        # return mask

        # ---------------------------------------------------------------------
        # Old band_mask implementation (kept for reference)
        # ---------------------------------------------------------------------
        min_band_frac: float = 0.05
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



    def _perlin_mask(self) -> np.ndarray:
        return _perlin_mask(
            self.n_mels,
            self.T,
            threshold_beta=self.perlin_threshold_beta,
            rotate_deg_range=self.perlin_rotate_deg_range,
        )

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