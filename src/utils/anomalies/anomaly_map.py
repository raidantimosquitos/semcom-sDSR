"""
Spectromorphic anomaly mask generation for sDSR training.

Rectangular band masks (uniform mel-bin bandwidth, random time segments) and
Perlin noise as a low-probability regularizer.  Masks are solid rectangles
(no pixel thinning) so max-pool projection to latent space stays coherent.
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F

from .perlin import rand_perlin_2d_np


def _random_band_fallback_mask(
    n_mels: int,
    T: int,
    *,
    active_mel_top: int | None = None,
    mel_frac_range: tuple[float, float] = (0.03, 0.45),
    time_frac_range: tuple[float, float] = (0.04, 0.55),
) -> np.ndarray:
    """
    Single solid rectangle on ``(n_mels, T)``: random mel height and vertical
    offset within ``[0, active_mel_top)`` (or full height when ``None``), random
    time length and start.  Fraction ranges control typical coverage on e.g.
    ``128×320`` inputs; always returns a non-empty mask when ``n_mels, T >= 1``.
    """
    mask = np.zeros((n_mels, T), dtype=np.float32)
    if n_mels < 1 or T < 1:
        return mask

    top = int(np.clip(active_mel_top if active_mel_top is not None else n_mels, 1, n_mels))

    lo_m, hi_m = mel_frac_range
    lo_m, hi_m = float(np.clip(min(lo_m, hi_m), 1e-6, 1.0)), float(
        np.clip(max(lo_m, hi_m), 1e-6, 1.0)
    )
    lo_t, hi_t = time_frac_range
    lo_t, hi_t = float(np.clip(min(lo_t, hi_t), 1e-6, 1.0)), float(
        np.clip(max(lo_t, hi_t), 1e-6, 1.0)
    )

    if top == 1:
        band_h, r0 = 1, 0
    else:
        raw_h = max(1, round(top * random.uniform(lo_m, hi_m)))
        band_h = min(top, max(2, raw_h))
        r0 = random.randint(0, top - band_h)
    r1 = r0 + band_h

    run_len = max(1, min(T, round(T * random.uniform(lo_t, hi_t))))
    t0 = random.randint(0, max(0, T - run_len))
    mask[r0:r1, t0 : t0 + run_len] = 1.0
    return mask


# ---------------------------------------------------------------------------
# Perlin regularizer
# ---------------------------------------------------------------------------

def _perlin_mask(
    n_mels: int,
    T: int,
    *,
    perlin_scale_freq: int = 4,
    perlin_scale_time: int = 6,
    min_perlin_scale_freq: int = 1,
    min_perlin_scale_time: int = 2,
    beta: float = 0.5,
    active_mel_top: int | None = None,
    fallback_mel_frac_range: tuple[float, float] = (0.03, 0.45),
    fallback_time_frac_range: tuple[float, float] = (0.04, 0.55),
) -> np.ndarray:
    """Thresholded Perlin binary mask; empty threshold → random band fallback."""
    # Anisotropic scale: freq axis coarser, time axis finer
    exp_x = int(torch.randint(min_perlin_scale_freq, perlin_scale_freq, (1,)).item())
    exp_y = int(torch.randint(min_perlin_scale_time, perlin_scale_time, (1,)).item())
    perlin_scalex = int(2**exp_x)
    perlin_scaley = int(2**exp_y)

    perlin_noise = rand_perlin_2d_np(
        (n_mels, T),
        (perlin_scalex, perlin_scaley),
    )
    if random.random() < 0.5:
        perlin_noise = np.ascontiguousarray(np.fliplr(perlin_noise))
    if random.random() < 0.5:
        perlin_noise = np.ascontiguousarray(np.flipud(perlin_noise))

    threshold = torch.rand(1).item() * beta + beta  # [beta, 2*beta]

    perlin_thr = np.where(
        np.abs(perlin_noise) > threshold,
        np.ones_like(perlin_noise, dtype=np.float32),
        np.zeros_like(perlin_noise, dtype=np.float32),
    )
    if float(perlin_thr.sum()) == 0.0:
        return _random_band_fallback_mask(
            n_mels,
            T,
            active_mel_top=active_mel_top,
            mel_frac_range=fallback_mel_frac_range,
            time_frac_range=fallback_time_frac_range,
        )
    return perlin_thr


# ---------------------------------------------------------------------------
# Main strategy
# ---------------------------------------------------------------------------

class SpectromorphicMaskStrategy:
    """
    Spectromorphic anomaly masks for sDSR training on DCASE2020 Task 2.

    Each sample draws **Band** with probability ``1 - perlin_prob`` or **Perlin**
    with probability ``perlin_prob``.

    **Band:** one contiguous mel band: ``band_h`` uniform on ``{2, …, active_top}``
    (``band_h = 1`` only if ``active_top == 1``), with ``active_top`` =
    ``band_active_mel_top`` or ``n_mels``, placed at a random row offset.
    Time axis: ``n_segs`` from ``band_n_segs_range`` with random interior cut-points;
    within each segment, a consecutive run with fill fraction from
    ``band_aug_frac_range``.

    **Perlin:** thresholded anisotropic noise (regularizer against axis-aligned bias).

    Args:
        n_mels: mel bins in the spectrogram.
        T: time frames in the spectrogram.
        q_shape: output spatial shape; masks are up/down-sampled if different
            from ``(n_mels, T)``.
        perlin_prob: probability of the Perlin branch (recommended ~0.2–0.3).
        perlin_q_range, perlin_shear_range, perlin_active_mel_top: Perlin controls.
        band_n_segs_range: inclusive range for number of time segments.
        band_aug_frac_range: fill fraction for the run inside each segment.
        band_active_mel_top: clip band to mel rows ``[0, active_top)``; ``None`` = all.
    """

    def __init__(
        self,
        n_mels: int = 128,
        T: int = 320,
        q_shape: tuple[int, int] | None = None,
        perlin_prob: float = 0.2,
        perlin_active_mel_top: int | None = None,
        band_n_segs_range: tuple[int, int] = (1, 5),
        band_aug_frac_range: tuple[float, float] = (0.1, 1.0),
        band_active_mel_top: int | None = None,
        **_: object,
    ) -> None:
        self.n_mels = n_mels
        self.T = T
        self.q_shape = q_shape or (n_mels, T)
        self.perlin_prob = float(np.clip(perlin_prob, 0.0, 1.0))
        self.perlin_active_mel_top = perlin_active_mel_top
        self.band_n_segs_range = band_n_segs_range
        self.band_aug_frac_range = band_aug_frac_range
        self.band_active_mel_top = band_active_mel_top

    def _band_mask(self) -> np.ndarray:
        """Single contiguous mel band × segmented time runs (solid rectangles)."""
        mask = np.zeros((self.n_mels, self.T), dtype=np.float32)
        active_top = (
            self.band_active_mel_top
            if self.band_active_mel_top is not None
            else self.n_mels
        )
        active_top = int(np.clip(active_top, 1, self.n_mels))

        if active_top == 1:
            band_h, r0, r1 = 1, 0, 1
        else:
            band_h = random.randint(2, active_top)
            r0 = random.randint(0, active_top - band_h)
            r1 = r0 + band_h

        n_segs = random.randint(self.band_n_segs_range[0], self.band_n_segs_range[1])
        if n_segs <= 1 or self.T < 2:
            segments = [(0, self.T)]
        else:
            n_cuts = min(n_segs - 1, self.T - 1)
            cut_pts = sorted(random.sample(range(1, self.T), n_cuts))
            bounds = [0] + cut_pts + [self.T]
            segments = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]

        for seg_start, seg_end in segments:
            seg_len = seg_end - seg_start
            if seg_len < 1:
                continue
            fill = random.uniform(
                self.band_aug_frac_range[0], self.band_aug_frac_range[1]
            )
            run_len = max(1, min(seg_len, round(fill * seg_len)))
            run_start = random.randint(0, seg_len - run_len)
            t0 = seg_start + run_start
            t1 = t0 + run_len
            mask[r0:r1, t0:t1] = 1.0

        return mask

    def _perlin_mask(self) -> np.ndarray:
        m = _perlin_mask(
            self.n_mels,
            self.T,
            active_mel_top=self.perlin_active_mel_top,
        )
        top = self.perlin_active_mel_top
        if top is not None and int(top) < self.n_mels:
            m = m.copy()
            m[int(top):, :] = 0.0
        return m

    def _sample_mask_numpy(self) -> np.ndarray:
        return (
            self._perlin_mask()
            if random.random() < self.perlin_prob
            else self._band_mask()
        )

    def __call__(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        """Return ``(B, 1, *q_shape)`` binary float32 mask tensor."""
        masks = [
            torch.from_numpy(self._sample_mask_numpy()).unsqueeze(0).unsqueeze(0)
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