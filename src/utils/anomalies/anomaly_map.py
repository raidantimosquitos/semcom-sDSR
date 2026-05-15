"""
Spectromorphic anomaly mask generation for sDSR training.

Band masks (narrow / harmonic), optional full-band short-time bursts, and
Perlin noise as a low-frequency regularizer.  Masks are solid rectangles
(no pixel thinning) so max-pool projection to latent space stays coherent.
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

    Each call draws one mask type per sample (mutually exclusive branches):

    * **Band** (remaining mass after Perlin and wide-burst): three-step
      rectangular band process → frequency-selective, temporally segmented
      strips with optional harmonic series.  Primary DCASE-motivated strategy.
    * **Wide-band bursts** (prob ``wide_burst_prob``): all mel rows in
      ``[0, wide_burst_active_mel_top)`` (or full ``n_mels`` when that is
      ``None``) active only on **several** short, non-overlapping time runs
      (impulsive / repeated transient broadband).  No multi-band harmonics.
    * **Perlin** (prob ``perlin_prob``): thresholded anisotropic Perlin noise
      → smooth, blob-shaped patches used as a low-probability regularizer
      to prevent the model from over-fitting to rectangular mask boundaries.

    Branch probabilities are ``perlin_prob``, ``wide_burst_prob``, and
    ``1 - perlin_prob - wide_burst_prob`` for Band.  ``wide_burst_prob`` is
    clipped so the three sum to 1.

    Args:
        n_mels: mel bins in the spectrogram.
        T: time frames in the spectrogram.
        q_shape: output spatial shape; masks are up/down-sampled if different
            from ``(n_mels, T)``.

        perlin_prob: probability of choosing the Perlin branch (recommended:
            0.2–0.3 so band masks remain the primary strategy).
        wide_burst_prob: probability of the full-row multi-burst template.
        wide_burst_n_range: inclusive range for how many time bursts to **attempt**
            to place (non-overlapping; fewer may fit if ``T`` is small).
        wide_burst_max_bursts: hard cap on burst count (default 5).
        wide_burst_time_frames_range: min/max length (frames) of each burst.
        wide_burst_active_mel_top: row ``r1`` for the wide band (rows
            ``[0, r1)``).  ``None`` uses all ``n_mels`` (full spectrum).
        perlin_q_range: quantile ``(q_min, q_max)`` for the signed threshold;
            keeps the top ``1 - q`` fraction of the signed noise field, giving
            ≈5–20% active pixels independent of scale.
        perlin_shear_range: mild frequency-axis shear range (default ±0.08,
            ≈25 mel-bin drift); ``None`` disables shearing.
        perlin_active_mel_top: zero out Perlin mask rows ≥ this bin index.
            ``None`` = all bins.  Recommended: ``int(n_mels * 0.875)`` to
            exclude the near-silent top bins.
        band_n_segs_range: ``(min, max)`` number of time segments (random
            cut-points, not evenly spaced).
        band_aug_frac_range: ``(min, max)`` fill fraction for consecutive runs
            within each segment; independent of bandwidth.
        band_harmonic_prob: probability of generating a harmonic series instead
            of a single band.  Harmonics share bandwidth and temporal pattern.
            ``n_harmonics`` is automatically capped so total frequency footprint
            stays ≤ ``active_top // 3``, bounding coverage at design time.
        band_max_harmonics: upper bound on harmonics in the series.
        band_active_mel_top: zero out band mask rows ≥ this bin index.
            ``None`` = all bins.
    """

    def __init__(
        self,
        n_mels: int = 128,
        T: int = 320,
        q_shape: tuple[int, int] | None = None,
        # --- branch probabilities ---
        perlin_prob: float = 0.2,
        wide_burst_prob: float = 0.4,
        # --- wide-band multi-burst ---
        wide_burst_n_range: tuple[int, int] = (1, 5),
        wide_burst_max_bursts: int = 5,
        wide_burst_time_frames_range: tuple[int, int] = (3, 28),
        wide_burst_active_mel_top: int | None = None,
        # --- Perlin parameters ---
        perlin_q_range: tuple[float, float] = (0.80, 0.95),
        perlin_shear_range: tuple[float, float] | None = (-0.08, 0.08),
        perlin_active_mel_top: int | None = None,
        # --- Band parameters ---
        band_n_segs_range: tuple[int, int] = (1, 5),
        band_aug_frac_range: tuple[float, float] = (0.3, 1.0),
        band_harmonic_prob: float = 0.3,
        band_max_harmonics: int = 4,
        band_active_mel_top: int | None = None,
        **_: object,
    ) -> None:
        self.n_mels = n_mels
        self.T = T
        self.q_shape = q_shape or (n_mels, T)
        self.perlin_prob = float(np.clip(perlin_prob, 0.0, 1.0))
        p_w = float(np.clip(wide_burst_prob, 0.0, 1.0))
        self.wide_burst_prob = min(p_w, max(0.0, 1.0 - self.perlin_prob))
        self.wide_burst_n_range = (
            int(wide_burst_n_range[0]),
            int(wide_burst_n_range[1]),
        )
        self.wide_burst_time_frames_range = (
            int(wide_burst_time_frames_range[0]),
            int(wide_burst_time_frames_range[1]),
        )
        self.wide_burst_active_mel_top = wide_burst_active_mel_top
        self.wide_burst_max_bursts = max(1, int(wide_burst_max_bursts))
        self.perlin_q_range = perlin_q_range
        self.perlin_shear_range = perlin_shear_range
        self.perlin_active_mel_top = perlin_active_mel_top
        self.band_n_segs_range = band_n_segs_range
        self.band_aug_frac_range = band_aug_frac_range
        self.band_harmonic_prob = float(np.clip(band_harmonic_prob, 0.0, 1.0))
        self.band_max_harmonics = max(2, band_max_harmonics)
        self.band_active_mel_top = band_active_mel_top

    def _wideband_burst_mask(self) -> np.ndarray:
        """
        Full (or configured) mel span with several short, non-overlapping
        temporal bursts.  One mask type for broadband impulsive / repeated
        transient signatures; no harmonic multi-bands.
        """
        mask = np.zeros((self.n_mels, self.T), dtype=np.float32)
        top = (
            self.wide_burst_active_mel_top
            if self.wide_burst_active_mel_top is not None
            else self.n_mels
        )
        r0, r1 = 0, int(np.clip(top, 1, self.n_mels))

        lo_n, hi_n = self.wide_burst_n_range
        lo_n, hi_n = max(1, lo_n), max(lo_n, hi_n)
        n_max = min(hi_n, self.T, self.wide_burst_max_bursts)
        n_min = min(max(1, lo_n), n_max)
        n_bursts = random.randint(n_min, n_max)

        lo_f, hi_f = self.wide_burst_time_frames_range
        lo_f, hi_f = max(1, lo_f), max(lo_f, hi_f)
        lo_f = min(lo_f, self.T)
        hi_f = min(hi_f, self.T)

        free = np.ones(self.T, dtype=bool)
        for _ in range(n_bursts):
            L = random.randint(lo_f, hi_f)
            if L > self.T or int(free.sum()) < L:
                break
            candidates: list[int] = []
            limit = self.T - L + 1
            for s in range(limit):
                if bool(free[s : s + L].all()):
                    candidates.append(s)
            if not candidates:
                break
            s = random.choice(candidates)
            mask[r0:r1, s : s + L] = 1.0
            free[s : s + L] = False

        if mask.sum() == 0:
            L = min(self.T, max(1, lo_f))
            s = random.randint(0, max(0, self.T - L))
            mask[r0:r1, s : s + L] = 1.0

        return mask

    def _band_mask(self) -> np.ndarray:
        """
        Three-step band mask for one anomaly event adapted for DCASE2020 Task 2.

        Each call generates exactly **one anomaly event** — one frequency
        selection (single band or harmonic series) with one shared temporal
        pattern.  This maps cleanly to one anomaly in the latent space because
        the mask projects via max-pool: any pixel active in a latent cell's
        receptive field marks the whole cell.  Generating multiple independent
        events per call would create unrelated substituted regions that do not
        correspond to a single physical fault.

        Step 1 — frequency row selection (log-uniform bandwidth):
          ``band_h`` drawn log-uniformly from ``[2, active_top // 3]``, giving
          equal probability of narrow tonal lines and moderate broadband bands.
          Single band or harmonic series (prob ``band_harmonic_prob``).  In
          harmonic mode, ``n_harmonics`` is further capped so total frequency
          footprint stays ≤ ``active_top // 3`` (same as a single max-width
          band): ``n_harmonics ≤ (active_top // 3) // band_h``.  This bounds
          coverage at design time without any pixel-level thinning.

        Step 2 — time segmentation (random cut-points):
          ``n_segs`` unique random interior cut-points partition ``[0, T)``
          into variable-length segments (no regular grid).

        Step 3 — consecutive run per segment:
          Fill fraction ``f ∈ band_aug_frac_range`` drawn per segment.  All
          bands in the harmonic series receive the same run ``(t0, t1)``
          (shared temporal activation matches the physical coupling of
          harmonics in a bearing/gear fault).

        No post-hoc pixel thinning is applied: coverage is bounded solely by
        the parametric design (log-uniform bandwidth, frequency footprint cap,
        fill fraction range), ensuring the latent-space mask remains a set of
        clean, unperforated rectangles.
        """
        mask = np.zeros((self.n_mels, self.T), dtype=np.float32)
        active_top = (
            self.band_active_mel_top
            if self.band_active_mel_top is not None
            else self.n_mels
        )

        # ── Step 1: frequency rows ─────────────────────────────────────────────
        bw_max_bins = max(3, active_top // 3)
        log_bw = random.uniform(math.log2(2), math.log2(bw_max_bins))
        band_h = max(2, round(2 ** log_bw))

        max_n_from_footprint = max(1, (active_top // 3) // band_h)
        if random.random() < self.band_harmonic_prob and max_n_from_footprint >= 2:
            # Harmonic series: N copies of the same band at equal mel-bin gap.
            # n_harmonics capped so total frequency footprint ≤ active_top // 3.
            n_harmonics = random.randint(2, min(self.band_max_harmonics, max_n_from_footprint))
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
            row_ranges = [(0, min(band_h, active_top))]

        # ── Step 2: random time segmentation ──────────────────────────────────
        n_segs = random.randint(self.band_n_segs_range[0], self.band_n_segs_range[1])
        if n_segs <= 1 or self.T < 2:
            segments = [(0, self.T)]
        else:
            n_cuts = min(n_segs - 1, self.T - 1)
            cut_pts = sorted(random.sample(range(1, self.T), n_cuts))
            bounds = [0] + cut_pts + [self.T]
            segments = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]

        # ── Step 3: consecutive run per segment ───────────────────────────────
        for seg_start, seg_end in segments:
            seg_len = seg_end - seg_start
            if seg_len < 1:
                continue
            fill = random.uniform(self.band_aug_frac_range[0], self.band_aug_frac_range[1])
            run_len = max(1, min(seg_len, round(fill * seg_len)))
            run_start = random.randint(0, seg_len - run_len)
            t0 = seg_start + run_start
            t1 = t0 + run_len
            for r0, r1 in row_ranges:
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

    # -- public interface ----------------------------------------------------

    def _sample_mask_numpy(self) -> np.ndarray:
        u = random.random()
        if u < self.perlin_prob:
            return self._perlin_mask()
        if u < self.perlin_prob + self.wide_burst_prob:
            return self._wideband_burst_mask()
        return self._band_mask()

    def __call__(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        """Return ``(B, 1, *q_shape)`` binary float32 mask tensor.

        Each sample independently draws Perlin, wide-band multi-burst, or
        Band with probabilities ``perlin_prob``, ``wide_burst_prob``, and
        ``1 - perlin_prob - wide_burst_prob``.
        """
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