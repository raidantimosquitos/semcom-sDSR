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
    shear_range: tuple[float, float] | None = (-0.08, 0.08),
    q_range: tuple[float, float] = (0.80, 0.95),
    active_mel_top: int | None = None,
    max_coverage: float = 0.20,
) -> np.ndarray:
    """
    Thresholded 2-D Perlin mask (binary float32) adapted for DCASE2020 Task 2.

    Scale: ``perlin_scaley`` ∈ {2, 4, 8} (≥2 frequency cells; scaley=1 is
    degenerate) × ``perlin_scalex`` ∈ {8, 16, 32, 64} (wide time extent).

    Threshold: quantile-based (``q ∈ q_range``) on the signed noise field —
    keeps the top ``1 - q`` fraction (≈5–20%) as one connected horizontal
    stripe rather than scattered blobs.

    Shear: optional mild shear along the frequency axis (default ±0.08,
    ≈25 mel-bin drift) models gradual harmonic drift.

    ``active_mel_top``: restrict mask to ``[0, active_mel_top)`` mel bins.

    ``max_coverage``: explicit pixel-count cap (fraction of ``n_mels × T``);
    excess active pixels are randomly cleared after thresholding, keeping
    coverage consistent with the band mask's ``band_max_coverage``.
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

    # Explicit coverage cap — mirrors band_mask's band_max_coverage.
    total = n_mels * T
    active = int(perlin_bin.sum())
    max_active = int(max_coverage * total)
    if active > max_active:
        flat_idx = np.flatnonzero(perlin_bin)
        to_clear = np.random.choice(flat_idx, active - max_active, replace=False)
        perlin_bin.flat[to_clear] = 0.0

    if perlin_bin.sum() == 0:
        bw = random.randint(2, max(3, n_mels // 8))
        bw_start = random.randint(0, max(0, top - bw))
        bw_end = min(bw_start + bw, top)
        perlin_bin[bw_start:bw_end, :] = 1.0

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
    Spectromorphic anomaly masks for sDSR training on DCASE2020 Task 2.

    Each call draws one mask type per sample:

    * **Band** (prob ``1 - perlin_prob``): three-step rectangular band process
      → frequency-selective, temporally segmented strips with optional
      harmonic series.  Primary DCASE-motivated strategy.
    * **Perlin** (prob ``perlin_prob``): thresholded anisotropic Perlin noise
      → smooth, blob-shaped patches used as a low-probability regularizer
      to prevent the model from over-fitting to rectangular mask boundaries.

    Args:
        n_mels: mel bins in the spectrogram.
        T: time frames in the spectrogram.
        q_shape: output spatial shape; masks are up/down-sampled if different
            from ``(n_mels, T)``.

        perlin_prob: probability of choosing the Perlin branch (recommended:
            0.2–0.3 so band masks remain the primary strategy).
        perlin_q_range: quantile ``(q_min, q_max)`` for the signed threshold;
            keeps the top ``1 - q`` fraction of the signed noise field, giving
            ≈5–20% active pixels independent of scale.
        perlin_shear_range: mild frequency-axis shear range (default ±0.08,
            ≈25 mel-bin drift); ``None`` disables shearing.
        perlin_active_mel_top: zero out Perlin mask rows ≥ this bin index.
            ``None`` = all bins.  Recommended: ``int(n_mels * 0.875)`` to
            exclude the near-silent top bins.
        perlin_max_coverage: active-pixel cap for Perlin masks (fraction of
            ``n_mels × T``); mirrors ``band_max_coverage`` for consistency.

        band_max_bands: max independent band groups per mask (uniform in
            ``[1, band_max_bands]``).
        band_n_segs_range: ``(min, max)`` number of time segments (random
            cut-points, not evenly spaced).
        band_aug_frac_range: ``(min, max)`` fill fraction for consecutive runs
            within each segment; independent of bandwidth.
        band_max_coverage: active-pixel cap as a fraction of ``n_mels × T``;
            excess pixels are randomly cleared.
        band_harmonic_prob: probability of generating a harmonic series instead
            of a single band.  Harmonics share bandwidth and temporal pattern.
        band_max_harmonics: maximum number of harmonics in the series.
        band_active_mel_top: zero out band mask rows ≥ this bin index.
            ``None`` = all bins.
    """

    def __init__(
        self,
        n_mels: int = 128,
        T: int = 320,
        q_shape: tuple[int, int] | None = None,
        # --- branch probability ---
        perlin_prob: float = 0.25,
        # --- Perlin parameters ---
        perlin_q_range: tuple[float, float] = (0.80, 0.95),
        perlin_shear_range: tuple[float, float] | None = (-0.08, 0.08),
        perlin_active_mel_top: int | None = None,
        perlin_max_coverage: float = 0.20,
        # --- Band parameters ---
        band_max_bands: int = 3,
        band_n_segs_range: tuple[int, int] = (1, 5),
        band_aug_frac_range: tuple[float, float] = (0.3, 1.0),
        band_max_coverage: float = 0.35,
        band_harmonic_prob: float = 0.3,
        band_max_harmonics: int = 4,
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
        self.perlin_max_coverage = float(np.clip(perlin_max_coverage, 0.0, 1.0))
        self.band_max_bands = max(1, band_max_bands)
        self.band_n_segs_range = band_n_segs_range
        self.band_aug_frac_range = band_aug_frac_range
        self.band_max_coverage = float(np.clip(band_max_coverage, 0.0, 1.0))
        self.band_harmonic_prob = float(np.clip(band_harmonic_prob, 0.0, 1.0))
        self.band_max_harmonics = max(2, band_max_harmonics)
        self.band_active_mel_top = band_active_mel_top


    def _band_mask(self, n_bands: int | None = None) -> np.ndarray:
        """
        Three-step band mask adapted for DCASE2020 Task 2 anomaly signatures.

        Step 1 — frequency band selection (log-uniform bandwidth):
          Bandwidth is drawn log-uniformly from ``[2, active_top // 3]`` mel
          bins, giving equal probability to narrow tonal lines and moderate
          broadband bands.  With probability ``band_harmonic_prob``, a harmonic
          series is generated instead: ``n_harmonics`` bands of equal width
          placed at uniform mel-bin spacing, all sharing the same temporal
          pattern from Steps 2–3 (physically: the same fault modulates multiple
          harmonics simultaneously).

        Step 2 — time segmentation (random cut-points):
          ``n_segs`` unique random interior cut-points divide ``[0, T)`` into
          variable-length segments, avoiding the regular periodicity of
          evenly-spaced boundaries.

        Step 3 — consecutive run per segment (fill fraction decoupled from BW):
          For each segment a fill fraction ``f ∈ band_aug_frac_range`` is drawn
          independently; a single consecutive run of length
          ``round(f × seg_len)`` is placed at a random offset.  All bands in a
          harmonic series receive the same ``(t0, t1)`` run (shared activation).

        A coverage cap thins active pixels to at most ``band_max_coverage``
        of the spectrogram area.
        """
        mask = np.zeros((self.n_mels, self.T), dtype=np.float32)
        active_top = (
            self.band_active_mel_top
            if self.band_active_mel_top is not None
            else self.n_mels
        )
        nb = n_bands if n_bands is not None else random.randint(1, self.band_max_bands)

        for _ in range(nb):
            # ── Step 1: frequency rows ────────────────────────────────────────
            bw_max_bins = max(3, active_top // 3)
            log_bw = random.uniform(math.log2(2), math.log2(bw_max_bins))
            band_h = max(2, round(2 ** log_bw))

            if random.random() < self.band_harmonic_prob:
                # Harmonic series: N copies of the same band at equal mel-bin gap.
                n_harmonics = random.randint(2, self.band_max_harmonics)
                gap_min = band_h + 1          # at least 1 silent bin between bands
                max_span = active_top - band_h
                gap_max = max(gap_min, max_span // max(1, n_harmonics - 1))
                gap = random.randint(gap_min, gap_max)
                # Fundamental center: ensure all harmonics fit below active_top.
                max_c0 = active_top - band_h // 2 - (n_harmonics - 1) * gap
                if max_c0 < band_h // 2:
                    n_harmonics, gap, max_c0 = 1, 0, max(band_h // 2, active_top - band_h)
                c0 = random.randint(band_h // 2, max(band_h // 2, max_c0))
                row_ranges: list[tuple[int, int]] = []
                for k in range(n_harmonics):
                    center = c0 + k * gap
                    r0 = max(0, center - band_h // 2)
                    r1 = min(active_top, r0 + band_h)
                    if r0 < r1:
                        row_ranges.append((r0, r1))
            else:
                # Single contiguous band.
                r0 = random.randint(0, max(0, active_top - band_h))
                r1 = min(r0 + band_h, active_top)
                row_ranges = [(r0, r1)]

            if not row_ranges:
                continue

            # ── Step 2: random time segmentation ─────────────────────────────
            n_segs = random.randint(
                self.band_n_segs_range[0], self.band_n_segs_range[1]
            )
            if n_segs <= 1 or self.T < 2:
                segments = [(0, self.T)]
            else:
                n_cuts = min(n_segs - 1, self.T - 1)
                cut_pts = sorted(random.sample(range(1, self.T), n_cuts))
                bounds = [0] + cut_pts + [self.T]
                segments = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]

            # ── Step 3: consecutive run per segment ───────────────────────────
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
                for r0, r1 in row_ranges:
                    mask[r0:r1, t0:t1] = 1.0

        # Coverage cap: randomly thin active pixels to at most band_max_coverage.
        total = self.n_mels * self.T
        active = int(mask.sum())
        max_active = int(self.band_max_coverage * total)
        if active > max_active:
            flat_idx = np.flatnonzero(mask)
            to_clear = np.random.choice(flat_idx, active - max_active, replace=False)
            mask.flat[to_clear] = 0.0

        return mask

    def _perlin_mask(self) -> np.ndarray:
        return _perlin_mask(
            self.n_mels,
            self.T,
            q_range=self.perlin_q_range,
            shear_range=self.perlin_shear_range,
            active_mel_top=self.perlin_active_mel_top,
            max_coverage=self.perlin_max_coverage,
        )

    # -- public interface ----------------------------------------------------

    def __call__(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        """Return ``(B, 1, *q_shape)`` binary float32 mask tensor.

        Each sample independently draws Perlin (prob ``perlin_prob``) or
        Band (prob ``1 - perlin_prob``).
        """
        masks = [
            torch.from_numpy(
                self._perlin_mask() if random.random() < self.perlin_prob
                else self._band_mask()
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