"""
Spectromorphic anomaly mask generation for sDSR training.

Band masks: n non-overlapping frequency bands with unbiased width and position
(no low-frequency skew), each divided into independent time segments with a
contiguous masked run.  Perlin noise as a low-probability regularizer.
Masks are solid rectangles so max-pool projection to latent space stays coherent.
"""

from __future__ import annotations

import random
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .perlin import rand_perlin_2d_np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sample_bands(
    n: int,
    bw_min: int,
    bw_max: int,
    active_top: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    """
    Place *n* non-overlapping bands of uniform-random width in [bw_min, bw_max]
    at uniform-random positions within [0, active_top).

    Width and position are sampled independently (no frequency bias), and bands
    are shuffled before placement so wide bands are equally likely anywhere in
    the spectrum.

    Returns list of (mel_start, mel_end) sorted by mel_start, or raises
    RuntimeError when the constraints cannot be satisfied.
    """
    bw_max = min(bw_max, active_top)

    for _ in range(500):
        widths = [rng.randint(bw_min, bw_max) for _ in range(n)]
        if sum(widths) > active_top:
            continue

        rng.shuffle(widths)  # decouple width from position

        slack = active_top - sum(widths)
        # Sample n-1 dividers uniformly in [0, slack] — symmetric gap distribution
        dividers = sorted(rng.randint(0, slack) for _ in range(n - 1))
        gaps = [d for d in np.diff([0] + dividers + [slack]).tolist()]

        # Random global offset so bands don't always hug mel-bin 0
        offset = rng.randint(0, int(gaps[-1])) if gaps[-1] > 0 else 0
        cursor = offset
        bands: list[tuple[int, int]] = []
        ok = True
        for i, w in enumerate(widths):
            end = cursor + w
            if end > active_top:
                ok = False
                break
            bands.append((cursor, end))
            cursor = end + (int(gaps[i]) if i < n - 1 else 0)

        if ok and len(bands) == n:
            return sorted(bands)

    raise RuntimeError(
        f"Cannot place {n} band(s) of width [{bw_min}, {bw_max}] "
        f"in {active_top} mel bins after 500 attempts. "
        f"Reduce n_bands_max or increase bandwidth_max."
    )


def _make_time_segments(T: int, n_segs: int) -> list[tuple[int, int]]:
    """
    Divide [0, T) into *n_segs* segments using random interior cut-points
    (matches the existing SpectromorphicMaskStrategy logic).
    """
    if n_segs <= 1 or T < 2:
        return [(0, T)]
    n_cuts = min(n_segs - 1, T - 1)
    cut_pts = sorted(random.sample(range(1, T), n_cuts))
    bounds = [0] + cut_pts + [T]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


# ---------------------------------------------------------------------------
# Perlin regularizer  (unchanged from original)
# ---------------------------------------------------------------------------

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
    time length and start.  Always returns a non-empty mask when n_mels, T >= 1.
    """
    mask = np.zeros((n_mels, T), dtype=np.float32)
    if n_mels < 1 or T < 1:
        return mask

    top = int(np.clip(active_mel_top if active_mel_top is not None else n_mels, 1, n_mels))

    lo_m = float(np.clip(min(mel_frac_range), 1e-6, 1.0))
    hi_m = float(np.clip(max(mel_frac_range), 1e-6, 1.0))
    lo_t = float(np.clip(min(time_frac_range), 1e-6, 1.0))
    hi_t = float(np.clip(max(time_frac_range), 1e-6, 1.0))

    if top == 1:
        band_h, r0 = 1, 0
    else:
        raw_h = max(1, round(top * random.uniform(lo_m, hi_m)))
        band_h = min(top, max(2, raw_h))
        r0 = random.randint(0, top - band_h)

    run_len = max(1, min(T, round(T * random.uniform(lo_t, hi_t))))
    t0 = random.randint(0, max(0, T - run_len))
    mask[r0 : r0 + band_h, t0 : t0 + run_len] = 1.0
    return mask


def _perlin_mask(
    n_mels: int,
    T: int,
    *,
    perlin_scale_freq: int = 6,
    perlin_scale_time: int = 6,
    min_perlin_scale_freq: int = 0,
    min_perlin_scale_time: int = 0,
    beta: float = 0.5,
    active_mel_top: int | None = None,
    fallback_mel_frac_range: tuple[float, float] = (0.03, 0.45),
    fallback_time_frac_range: tuple[float, float] = (0.04, 0.55),
) -> np.ndarray:
    """Thresholded Perlin binary mask; empty threshold → random band fallback."""
    exp_x = int(torch.randint(min_perlin_scale_freq, perlin_scale_freq, (1,)).item())
    exp_y = int(torch.randint(min_perlin_scale_time, perlin_scale_time, (1,)).item())

    perlin_noise = rand_perlin_2d_np(
        (n_mels, T),
        (int(2**exp_x), int(2**exp_y)),
    )

    threshold = torch.rand(1).item() * beta + beta  # uniform in [beta, 2*beta]
    perlin_thr = (np.abs(perlin_noise) > threshold).astype(np.float32)

    if perlin_thr.sum() == 0.0:
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

    Each sample draws **Band** (prob ``1 - perlin_prob``) or **Perlin**
    (prob ``perlin_prob``).

    Band branch
    -----------
    Samples ``n`` non-overlapping frequency bands where both bandwidth and
    position are drawn uniformly — no low-frequency bias.  Each band gets its
    own independently sampled segment count and per-segment masked run.

    * ``n_bands_range``     – (min, max) number of bands per mask.
    * ``bandwidth_range``   – (min, max) mel-bin height of each band.
    * ``band_n_segs_range`` – (min, max) time segments **per band**.
    * ``band_aug_frac_range`` – fill fraction of the run inside each segment.
    * ``band_active_mel_top`` – restrict bands to mel rows [0, top); None = all.

    Perlin branch (regularizer)
    ---------------------------
    Thresholded anisotropic noise to break axis-aligned bias.
    * ``perlin_prob`` – probability of choosing this branch (~0.2–0.3 recommended).
    * ``perlin_active_mel_top`` – zero out Perlin mask above this mel row.

    Projection diagnostics
    ----------------------
    Call ``coverage_stats(mask_np)`` to get raw and max-pool projected coverage
    at the two U-Net latent resolutions (32×80, 16×40).  Optimal targets:
    raw 25–45 %, projected 40–65 %.

    Args:
        n_mels: mel bins in the spectrogram.
        T: time frames in the spectrogram.
        q_shape: output spatial shape; masks are resampled (nearest) if different.
        perlin_prob: probability of the Perlin branch.
        n_bands_range: inclusive (min, max) for number of frequency bands.
        bandwidth_range: inclusive (min, max) mel-bin bandwidth per band.
        band_n_segs_range: inclusive (min, max) time segments per band.
        band_aug_frac_range: fill fraction for the run inside each segment.
        band_active_mel_top: restrict bands to rows [0, top). None = n_mels.
        perlin_active_mel_top: zero Perlin mask above this row. None = n_mels.
    """

    # Coverage targets (informational — used by coverage_stats)
    TARGET_RAW   = (0.25, 0.45)
    TARGET_PROJ  = (0.40, 0.65)
    LATENT_SHAPES: Sequence[tuple[int, int]] = ((32, 80), (16, 40))

    def __init__(
        self,
        n_mels: int = 128,
        T: int = 320,
        q_shape: tuple[int, int] | None = None,
        # Perlin
        perlin_prob: float = 0.2,
        perlin_active_mel_top: int | None = None,
        # Band
        n_bands_range: tuple[int, int] = (1, 4),
        bandwidth_range: tuple[int, int] = (4, 64),
        band_n_segs_range: tuple[int, int] = (1, 6),
        band_aug_frac_range: tuple[float, float] = (0.15, 0.65),
        band_active_mel_top: int | None = None,
        **_: object,
    ) -> None:
        self.n_mels = n_mels
        self.T = T
        self.q_shape = q_shape or (n_mels, T)
        self.perlin_prob = float(np.clip(perlin_prob, 0.0, 1.0))
        self.perlin_active_mel_top = perlin_active_mel_top

        self.n_bands_range     = n_bands_range
        self.bandwidth_range   = bandwidth_range
        self.band_n_segs_range = band_n_segs_range
        self.band_aug_frac_range = band_aug_frac_range
        self.band_active_mel_top = band_active_mel_top

        self._rng = random.Random()  # isolated RNG — does not perturb global state

    # ------------------------------------------------------------------
    # Band branch
    # ------------------------------------------------------------------

    def _band_mask(self) -> np.ndarray:
        """
        Multi-band mask: n unbiased non-overlapping bands × per-band segmented
        time runs.  All four stochastic parameters are sampled independently.
        """
        mask = np.zeros((self.n_mels, self.T), dtype=np.float32)

        active_top = int(np.clip(
            self.band_active_mel_top if self.band_active_mel_top is not None
            else self.n_mels,
            1, self.n_mels,
        ))

        n_bands = self._rng.randint(*self.n_bands_range)
        bw_min, bw_max = self.bandwidth_range

        # Clamp bw_min so n bands always fit; warn instead of crash
        min_total = n_bands * bw_min
        if min_total > active_top:
            bw_min = max(1, active_top // n_bands)

        try:
            bands = _sample_bands(n_bands, bw_min, bw_max, active_top, self._rng)
        except RuntimeError:
            # Graceful fallback: single band via original single-rect logic
            bands = self._fallback_single_band(active_top)

        lo_frac, hi_frac = self.band_aug_frac_range

        for mel_start, mel_end in bands:
            n_segs = self._rng.randint(*self.band_n_segs_range)
            segments = _make_time_segments(self.T, n_segs)

            for seg_start, seg_end in segments:
                seg_len = seg_end - seg_start
                if seg_len < 1:
                    continue
                fill = self._rng.uniform(lo_frac, hi_frac)
                run_len = max(1, min(seg_len, round(fill * seg_len)))
                run_start = self._rng.randint(0, seg_len - run_len)
                t0 = seg_start + run_start
                mask[mel_start:mel_end, t0 : t0 + run_len] = 1.0

        return mask

    def _fallback_single_band(self, active_top: int) -> list[tuple[int, int]]:
        """Single random band — used when multi-band placement fails."""
        bw = self._rng.randint(
            max(1, self.bandwidth_range[0]),
            min(self.bandwidth_range[1], active_top),
        )
        r0 = self._rng.randint(0, active_top - bw)
        return [(r0, r0 + bw)]

    # ------------------------------------------------------------------
    # Perlin branch
    # ------------------------------------------------------------------

    def _perlin_mask(self) -> np.ndarray:
        m = _perlin_mask(self.n_mels, self.T, active_mel_top=self.perlin_active_mel_top)
        top = self.perlin_active_mel_top
        if top is not None and int(top) < self.n_mels:
            m = m.copy()
            m[int(top) :, :] = 0.0
        return m

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @staticmethod
    def _maxpool2d(mask: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
        """
        Binary 2-D max pooling with non-overlapping windows.

        Any 1 inside a pooling window → 1 in output (binary OR semantics).
        Window size is inferred from mask.shape / (out_h, out_w); must divide
        evenly.
        """
        H, W = mask.shape
        if H % out_h != 0 or W % out_w != 0:
            raise ValueError(
                f"Mask shape {mask.shape} not evenly divisible by "
                f"({out_h}, {out_w})."
            )
        kh, kw = H // out_h, W // out_w
        return (
            mask.reshape(out_h, kh, out_w, kw)
            .max(axis=(1, 3))
            .astype(np.uint8)
        )

    def coverage_stats(self, mask: np.ndarray) -> dict[str, float]:
        """
        Return raw and max-pool projected coverage for a single (n_mels, T)
        mask.  Useful for offline auditing and hyperparameter tuning.

        Example::

            stats = strategy.coverage_stats(mask_np)
            # {'raw': 0.31, 'proj_32x80': 0.52, 'proj_16x40': 0.58}
        """
        stats: dict[str, float] = {"raw": float(mask.mean())}
        for lh, lw in self.LATENT_SHAPES:
            try:
                proj = self._maxpool2d(mask.astype(np.uint8), lh, lw)
                stats[f"proj_{lh}x{lw}"] = float(proj.mean())
            except ValueError:
                pass  # skip shapes that don't divide evenly
        return stats

    def coverage_ok(self, mask: np.ndarray) -> bool:
        """
        True when raw coverage falls within TARGET_RAW.

        Useful for rejection sampling inside a training DataLoader worker::

            for _ in range(20):
                m = strategy._sample_mask_numpy()
                if strategy.coverage_ok(m):
                    break
        """
        raw = float(mask.mean())
        return self.TARGET_RAW[0] <= raw <= self.TARGET_RAW[1]


# ---------------------------------------------------------------------------
# AnomalyMapGenerator — training entry point  (interface unchanged)
# ---------------------------------------------------------------------------

class AnomalyMapGenerator:
    """
    Wraps :class:`SpectromorphicMaskStrategy` for training-loop use.

    Args:
        spectrogram_shape: ``(n_mels, T)`` of the model input.
        q_shape: output shape (defaults to ``spectrogram_shape``).
        **strategy_kwargs: forwarded verbatim to
            :class:`SpectromorphicMaskStrategy`.
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
        normal / zero masks is handled at the dataset level.
        """
        return self._strategy(batch_size, device)

    def generate_for_training_sample(
        self,
        device: torch.device | str,
    ) -> torch.Tensor:
        """Convenience wrapper for a single sample."""
        return self.generate(1, device)

    # Expose diagnostics without requiring a strategy import at the call site
    def coverage_stats(self, mask: np.ndarray) -> dict[str, float]:
        """See :meth:`SpectromorphicMaskStrategy.coverage_stats`."""
        return self._strategy.coverage_stats(mask)