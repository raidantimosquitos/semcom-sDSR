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
    shear_range: tuple[float, float] | None = (-0.08, 0.08),
    q_range: tuple[float, float] = (0.80, 0.95),
    active_mel_top: int | None = None,
) -> np.ndarray:
    """
    Thresholded 2-D Perlin mask (binary float32) adapted for DCASE2020 Task 2.

    Scale: ``perlin_scaley`` ∈ {2, 4, 8} (≥2 frequency cells; scaley=1 is
    degenerate) × ``perlin_scalex`` ∈ {8, 16, 32, 64} (wide time extent).

    Threshold: quantile-based (``q ∈ q_range``) on the signed noise field —
    keeps the top ``1 - q`` fraction (≈5–20%) as one connected horizontal
    stripe rather than scattered blobs.  Coverage is bounded implicitly by the
    quantile: no pixel-level thinning is applied (it would fragment the blob
    without reducing latent-space coverage due to max-pool projection).

    Shear: optional mild shear along the frequency axis (default ±0.08,
    ≈25 mel-bin drift) models gradual harmonic drift.

    ``active_mel_top``: restrict mask to ``[0, active_mel_top)`` mel bins.
    """
    perlin_scaley = 2 ** int(random.randint(1, 3))   # {2, 4, 8}
    perlin_scalex = 2 ** int(random.randint(3, 6))   # {8, 16, 32, 64}

    noise = rand_perlin_2d_np((n_mels, T), (perlin_scaley, perlin_scalex)).astype(np.float32)

    if random.random() < 0.5 and shear_range is not None:
        shear = random.uniform(*shear_range)
        noise = _shear_perlin_freq_axis(noise, shear)

    sign = random.choice([1, -1])
    signed_noise = sign * noise
    q = random.uniform(q_range[0], q_range[1])
    tau = float(np.quantile(signed_noise, q))
    perlin_bin = (signed_noise > tau).astype(np.float32)

    top = active_mel_top if active_mel_top is not None else n_mels
    if top < n_mels:
        perlin_bin[top:, :] = 0.0

    if perlin_bin.sum() == 0:
        bw = random.randint(2, max(3, n_mels // 8))
        bw_start = random.randint(0, max(0, top - bw))
        bw_end = min(bw_start + bw, top)
        perlin_bin[bw_start:bw_end, :] = 1.0

    return perlin_bin


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
        perlin_prob: float = 0.25,
        perlin_q_range: tuple[float, float] = (0.80, 0.95),
        perlin_shear_range: tuple[float, float] | None = (-0.08, 0.08),
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
        self.perlin_q_range = perlin_q_range
        self.perlin_shear_range = perlin_shear_range
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
        return _perlin_mask(
            self.n_mels,
            self.T,
            q_range=self.perlin_q_range,
            shear_range=self.perlin_shear_range,
            active_mel_top=self.perlin_active_mel_top,
        )

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