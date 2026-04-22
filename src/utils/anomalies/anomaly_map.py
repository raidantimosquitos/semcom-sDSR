"""
Anomaly map generation for sDSR training.

Strategies (same interface: ``__call__(batch_size, device)`` → ``(B, 1, H, W)``):

1. NonStationarySpectromorphicMaskStrategy – linear-Hz band (mapped to mel bins) × renewal time,
   optional Perlin (e.g. valve, slider). Aliased as ``SpectromorphicMaskStrategy``.
2. StationarySpectromorphicMaskStrategy – multi-band linear-Hz (mapped to mel) × independent
   renewal time, optional Perlin (fan, pump, ToyConveyor, ToyCar).

:class:`AnomalyMapGenerator` holds both and dispatches from the DCASE ``machine_type``
argument on each generate call.
"""

from __future__ import annotations

import math
from typing import Any
import random
import numpy as np
import torch
import torch.nn.functional as F

from .perlin import rand_perlin_2d_np

def hz_to_mel(hz: float) -> float:
    """Mel scale (Hz) — same family as typical log-mel frontends."""
    return 2595.0 * math.log10(1.0 + max(hz, 0.0) / 700.0)

def hz_band_to_mel_bin_range(
    f0_hz: float,
    bw_hz: float,
    n_mels: int,
    f_min_hz: float,
    f_max_hz: float,
) -> tuple[int, int]:
    """
    Map a linear-frequency band ``[f0_hz, f0_hz + bw_hz)`` (clipped to ``[f_min_hz, f_max_hz]``)
    to half-open mel-bin indices ``[i0, i1)`` consistent with a uniform mel partition between
    ``f_min_hz`` and ``f_max_hz`` (same convention as spacing along the mel axis in log-mel).
    """
    if n_mels <= 0:
        return 0, 0
    f_min_hz = float(f_min_hz)
    f_max_hz = float(f_max_hz)
    if f_max_hz <= f_min_hz:
        return 0, min(1, n_mels)

    f_lo = max(f_min_hz, f0_hz)
    f_hi = min(f_max_hz, f0_hz + max(bw_hz, 0.0))
    if f_hi <= f_lo:
        f_hi = min(f_max_hz, f_lo + 1.0)

    mel_min = hz_to_mel(f_min_hz)
    mel_max = hz_to_mel(f_max_hz)
    span = mel_max - mel_min
    if span <= 0:
        return 0, min(1, n_mels)

    m_lo = hz_to_mel(f_lo)
    m_hi = hz_to_mel(f_hi)
    i0 = (m_lo - mel_min) / span * n_mels
    i1 = (m_hi - mel_min) / span * n_mels

    start_f = int(math.floor(i0))
    end_f = int(math.ceil(i1))
    start_f = max(0, min(n_mels - 1, start_f))
    end_f = max(start_f + 1, min(n_mels, end_f))
    return start_f, end_f


def sample_f0_bw_hz_stratified(
    f_min_hz: float,
    f_max_hz: float,
    bw_min_hz: float,
    bw_max_hz: float,
    *,
    split_hz: float = 2000.0,
    low_stratum_prob: float = 0.7,
) -> tuple[float, float] | None:
    """
    Sample band start ``f0_hz`` and width ``bw_hz`` (linear Hz) from two strata:

    - **Low**: band contained in ``[0, split_hz] ∩ [f_min_hz, f_max_hz]``.
    - **High**: band contained in ``[split_hz, f_max_hz] ∩ [f_min_hz, f_max_hz]``.

    With probability ``low_stratum_prob`` the low stratum is tried first, then high;
    otherwise the order is reversed. ``bw_hz`` is uniform in
    ``[bw_min_hz, min(bw_max_hz, stratum_width)]``.

    Returns:
        ``(f0_hz, bw_hz)`` or ``None`` if neither stratum can fit ``bw_min_hz``.
    """
    lo_g = float(f_min_hz)
    hi_g = float(f_max_hz)
    bw_lo = float(max(0.0, bw_min_hz))
    bw_hi = float(max(bw_lo, bw_max_hz))
    p_low = float(min(max(low_stratum_prob, 0.0), 1.0))
    split_hz = float(split_hz)

    def _one_stratum(s_lo: float, s_hi: float) -> tuple[float, float] | None:
        span = s_hi - s_lo
        if span <= 0:
            return None
        max_bw = min(bw_hi, span)
        if bw_lo > max_bw + 1e-9:
            return None
        bw_hz = random.uniform(bw_lo, max_bw)
        f0_hz = random.uniform(s_lo, s_hi - bw_hz)
        return f0_hz, bw_hz

    low_bounds = (max(lo_g, 0.0), min(split_hz, hi_g))
    high_bounds = (max(lo_g, split_hz), hi_g)

    if random.random() < p_low:
        strata = (low_bounds, high_bounds)
    else:
        strata = (high_bounds, low_bounds)

    for s_lo, s_hi in strata:
        r = _one_stratum(s_lo, s_hi)
        if r is not None:
            return r
    return None


def _perlin_mask_is_nonempty_binary(m: np.ndarray) -> bool:
    """True iff ``m`` is strictly binary {0,1} and has at least one 1."""
    if m.size == 0:
        return False
    flat = m.reshape(-1)
    if not np.all((flat == 0.0) | (flat == 1.0)):
        return False
    return bool(flat.sum() > 0.0)


def default_spectromorphic_perlin(
    n_mels: int,
    T: int,
) -> np.ndarray:
    """
    Thresholded 2-D Perlin noise (non-stationary / stationary Perlin branch).

    Perlin ``res`` is the low-frequency grid size along mel and time. It must stay
    on the order of the spectrogram dimensions; otherwise ``rand_perlin_2d_np``
    builds enormous intermediate grids (e.g. if ``res_x`` were ``2**(res_y+k)``
    with ``res_y`` already a power of two, ``res_x`` could reach ``2^19``).

    If the binary mask is empty after thresholding, returns the raw noise field
    (soft fallback) so the result is still usable.
    """
    if n_mels <= 0 or T <= 0:
        return np.zeros((max(0, n_mels), max(0, T)), dtype=np.float32)

    # Independent coarse scales in [2, 16] on each axis (cost ~ O(n_mels * T)).
    res_y = 2 ** random.randint(1, 4)
    res_x = 2 ** random.randint(1, 4)
    res_y = max(2, min(res_y, n_mels))
    res_x = max(2, min(res_x, T))
    noise = rand_perlin_2d_np((n_mels, T), (res_y, res_x))
    threshold = random.uniform(0.3, 0.6)
    mask = (noise > threshold).astype(np.float32)
    if mask.sum() > 0:
        return mask
    else:
        return noise.astype(np.float32)


class NonStationarySpectromorphicMaskStrategy:
    """
    Non-stationary synthetic anomaly masks for spectrograms.

    Each sample is either:

      1. **Band + renewal time** (probability ``1 - perlin_prob``): exactly one
         contiguous mel band. Band edges are sampled in **linear Hz** on
         ``[f_min_hz, f_max_hz]``: ``bw_hz ~ Uniform(bw_min_hz, bw_max_hz)``,
         ``f0_hz ~ Uniform(f_min_hz, f_max_hz - bw_hz)``, then mapped to mel-bin
         indices via :func:`hz_band_to_mel_bin_range` / :func:`hz_to_mel`.

      2. **Perlin** (probability ``perlin_prob``): thresholded 2-D Perlin noise
         for blob-shaped masks.

    Output is binary, shape ``(1, 1, n_mels, T)``. Callers may downsample with
    ``F.interpolate`` when ``q_shape`` differs from spectrogram resolution.

    Args:
        perlin_prob: probability of choosing the Perlin branch per mask.
        f_min_hz, f_max_hz: linear frequency range matching the mel spectrogram (Hz).
        bw_min_hz, bw_max_hz: uniform range for band width in Hz (machine-like bands).
        hz_stratified_sampling: if True, draw ``(f0_hz, bw_hz)`` via
            :func:`sample_f0_bw_hz_stratified`; if False, uniform over full range.
        hz_stratum_split_hz: boundary (Hz) between low and high strata (default 2000).
        hz_low_stratum_prob: probability to try the low stratum first (default 0.7).
        fallback_band_bw_hz: linear bandwidth (Hz) for the time-full strip used when
            band+renewal yields no mask or Perlin does not yield a non-empty binary mask.
    """

    def __init__(
        self,
        spectrogram_shape: tuple[int, int] | None = None,
        q_shape: tuple[int, int] | None = None,
        n_mels: int | None = None,
        T: int | None = None,
        perlin_prob: float = 0.1,
        f_min_hz: float = 0.0,
        f_max_hz: float = 8_000.0,
        bw_min_hz: float = 40.0,
        bw_max_hz: float = 1000.0,
        hz_stratified_sampling: bool = True,
        hz_stratum_split_hz: float = 2000.0,
        hz_low_stratum_prob: float = 0.5,
        fallback_band_bw_hz: float = 40.0,
        **_kwargs: object,
    ) -> None:
        if spectrogram_shape is not None:
            self.n_mels = n_mels or spectrogram_shape[0]
            self.T = T or spectrogram_shape[1]
        else:
            self.n_mels = n_mels or 128
            self.T = T or 320
        self.q_shape = q_shape or (self.n_mels, self.T)
        self.perlin_prob = float(min(max(perlin_prob, 0.0), 1.0))
        self.f_min_hz = float(f_min_hz)
        self.f_max_hz = float(f_max_hz)
        self.bw_min_hz = float(max(0.0, bw_min_hz))
        self.bw_max_hz = float(max(self.bw_min_hz, bw_max_hz))
        self.hz_stratified_sampling = bool(hz_stratified_sampling)
        self.hz_stratum_split_hz = float(hz_stratum_split_hz)
        self.hz_low_stratum_prob = float(min(max(hz_low_stratum_prob, 0.0), 1.0))
        self.fallback_band_bw_hz = float(max(1.0, fallback_band_bw_hz))

    def _fallback_time_continuous_band_mask(self) -> np.ndarray:
        """
        Full-time contiguous mel strip: ``bw`` Hz wide in linear frequency, all ``T`` frames 1.

        ``f0`` is uniform on valid starts in ``[f_min_hz, f_max_hz]``.
        """
        n_mels, T = self.n_mels, self.T
        out = np.zeros((max(0, n_mels), max(0, T)), dtype=np.float32)
        if n_mels <= 0 or T <= 0:
            return out

        lo, hi = self.f_min_hz, self.f_max_hz
        span = float(hi - lo)
        if span <= 0:
            return out

        bw = min(self.fallback_band_bw_hz, span)
        f0_hi = lo + max(0.0, span - bw)
        f0_hz = random.uniform(lo, f0_hi) if f0_hi >= lo else lo
        f0_bin, f1_bin = hz_band_to_mel_bin_range(f0_hz, bw, n_mels, lo, hi)
        if f1_bin > f0_bin:
            out[f0_bin:f1_bin, :] = 1.0
        return out

    def _pick_single_hz_mel_band(self) -> list[tuple[int, int]]:
        """One contiguous band ``[f0, f1)`` in mel bins from uniform Hz ``f0``, ``bw``."""
        n_mels = self.n_mels
        if n_mels <= 0:
            return []

        lo, hi = self.f_min_hz, self.f_max_hz
        span_hz = hi - lo
        if span_hz <= 0:
            return []

        bw_lo = min(self.bw_min_hz, span_hz)
        bw_hi = min(self.bw_max_hz, span_hz)
        if bw_hi < bw_lo:
            bw_lo, bw_hi = bw_hi, bw_lo

        for _ in range(32):
            sampled: tuple[float, float] | None = None
            if self.hz_stratified_sampling:
                sampled = sample_f0_bw_hz_stratified(
                    lo,
                    hi,
                    bw_lo,
                    bw_hi,
                    split_hz=self.hz_stratum_split_hz,
                    low_stratum_prob=self.hz_low_stratum_prob,
                )
            if sampled is not None:
                f0_hz, bw_hz = sampled
            else:
                bw_hz = random.uniform(bw_lo, bw_hi)
                f0_hz = random.uniform(lo, max(lo, hi - bw_hz))
            f0, f1 = hz_band_to_mel_bin_range(f0_hz, bw_hz, n_mels, lo, hi)
            if f1 > f0:
                return [(f0, f1)]
        return []

    def _hierarchical_renewal_params(self, T: int) -> tuple[float, float, float, float]:
        """
        Geometric run lengths: mean on-run ``mu_on``, mean off-run ``mu_off``,
        each log-uniform in [1, T]. Return ``(p_on, p_off, mu_on, mu_off)`` with
        ``p = 1/mu`` for ``np.random.geometric``.
        """
        Tm = max(T, 1)
        lo = math.log(1.0 + 10.0)
        hi = math.log(float(Tm-1.0))
        mu_on = math.exp(random.uniform(lo, hi)) # before was hi
        # mu_on = math.exp(random.uniform(math.log(50), math.log(T)))
        mu_off = math.exp(random.uniform(lo, hi)) # before was lo
        # mu_off = math.exp(random.uniform(math.log(1), math.log(10)))
        p_on = 1.0 / max(mu_on, 1.0)
        p_off = 1.0 / max(mu_off, 1.0)
        eps = 1e-6
        p_on = min(max(p_on, eps), 1.0 - eps)
        p_off = min(max(p_off, eps), 1.0 - eps)
        return p_on, p_off, mu_on, mu_off

    @staticmethod
    def _alternating_renewal_1d(
        T: int,
        p_on: float,
        p_off: float,
        start_on: bool,
    ) -> np.ndarray:
        """Binary (T,) via alternating on/off geometric runs, truncated to T."""
        a = np.zeros(T, dtype=np.float32)
        if T <= 0:
            return a
        t = 0
        on = start_on
        while t < T:
            p = p_on if on else p_off
            L = int(np.random.geometric(p))
            L = max(1, min(L, T - t))
            if on:
                a[t : t + L] = 1.0
            t += L
            on = not on
        return a

    def _band_time_segments(self) -> np.ndarray:
        """
        Binary mask: one Hz-derived mel band × time from alternating renewal
        (geometric runs, hierarchical means).
        """
        n_mels, T = self.n_mels, self.T
        mask = np.zeros((n_mels, T), dtype=np.float32)
        bands = self._pick_single_hz_mel_band()
        if not bands:
            return self._fallback_time_continuous_band_mask()

        f0, f1 = bands[0]
        p_on, p_off, mu_on, mu_off = self._hierarchical_renewal_params(T)
        denom = mu_on + mu_off
        p_start_on = (mu_on / denom) if denom > 0 else 0.5
        track = self._alternating_renewal_1d(
            T, p_on, p_off, random.random() < p_start_on
        )
        mask[f0:f1, :] = track.reshape(1, -1)
        if float(mask.sum()) <= 0.0:
            return self._fallback_time_continuous_band_mask()
        return mask

    def _perlin(self) -> np.ndarray:
        m = default_spectromorphic_perlin(self.n_mels, self.T)
        if not _perlin_mask_is_nonempty_binary(m):
            return self._fallback_time_continuous_band_mask()
        return m

    def __call__(
        self,
        batch_size: int,
        device: torch.device | str,
    ) -> torch.Tensor:
        masks = []
        for _ in range(batch_size):
            if random.random() < self.perlin_prob:
                m = self._perlin()
            else:
                m = self._band_time_segments()
            masks.append(torch.from_numpy(m).unsqueeze(0).unsqueeze(0))
        M = torch.cat(masks, dim=0).to(device)
        if self.q_shape != (self.n_mels, self.T):
            M = F.interpolate(M, size=self.q_shape, mode="nearest")
        return M

# ---------------------------------------------------------------------------
# 2. AnomalyMapGenerator — unified entry point
# ---------------------------------------------------------------------------

class AnomalyMapGenerator:
    """
    Build spectromorphic anomaly masks for training.

    Holds both stationary and non-stationary strategies. Each call selects
    :class:`StationarySpectromorphicMaskStrategy` vs :class:`NonStationarySpectromorphicMaskStrategy`
    from the DCASE ``machine_type`` (``None`` → non-stationary).

    When ``force_anomaly=False``, each index independently receives an all-zero mask
    with probability ``zero_mask_prob``.

    Extra keyword arguments are forwarded to **both** strategy classes; each ignores
    keys it does not use (e.g. ``mel_n_bands_max`` only affects
    :class:`StationarySpectromorphicMaskStrategy`). Hz band args (``f_min_hz``,
    ``f_max_hz``, ``bw_min_hz``, ``bw_max_hz``) apply to both strategies.
    Optional stratified Hz sampling: ``hz_stratified_sampling`` (default True),
    ``hz_stratum_split_hz`` (default 2000), ``hz_low_stratum_prob`` (default 0.7),
    ``fallback_band_bw_hz`` (default 40): Hz width for fallback time-full strip.
    """

    def __init__(
        self,
        spectrogram_shape: tuple[int, int],
        q_shape: tuple[int, int] | None = None,
        n_mels: int | None = None,
        T: int | None = None,
        zero_mask_prob: float = 0.5,
        **strategy_kwargs: Any,
    ) -> None:
        self.spectrogram_shape = spectrogram_shape
        self.q_shape = q_shape or spectrogram_shape
        self.zero_mask_prob = zero_mask_prob

        _n_mels = n_mels or spectrogram_shape[0]
        _T = T or spectrogram_shape[1]

        self._nonstationary = NonStationarySpectromorphicMaskStrategy(
            spectrogram_shape,
            self.q_shape,
            _n_mels,
            _T,
            **strategy_kwargs,
        )

    def _strategy(
        self,
        machine_type: str | None,
    ) -> NonStationarySpectromorphicMaskStrategy:
        return (
            self._nonstationary
        )

    def _generate_one(
        self,
        device: torch.device | str,
        machine_type: str | None = None,
    ) -> torch.Tensor:
        """Generate a single non-zero mask ``(1, 1, H, W)``."""
        return self._strategy(machine_type)(1, device)

    def generate_for_training_sample(
        self,
        device: torch.device | str,
        force_anomaly: bool = True,
        machine_type: str | None = None,
    ) -> torch.Tensor:
        """Generate one mask for a single training sample (convenience wrapper)."""
        if force_anomaly:
            return self._generate_one(device, machine_type)
        mt_list = [machine_type] if machine_type is not None else None
        return self.generate(1, device, force_anomaly=False, machine_types=mt_list)

    def generate(
        self,
        batch_size: int,
        device: torch.device | str,
        force_anomaly: bool = False,
        machine_types: list[str] | None = None,
    ) -> torch.Tensor:
        """
        Generate a batch of anomaly masks.

        Args:
            batch_size: number of masks
            device: torch device
            force_anomaly: if True, skip ``zero_mask_prob`` (always real masks)
            machine_types: optional length-``batch_size`` list of DCASE machine types.
                If ``None``, every mask uses the non-stationary strategy.

        Returns:
            ``M``: ``(B, 1, n_mels, T)`` binary mask
        """
        if machine_types is not None and len(machine_types) != batch_size:
            raise ValueError(
                f"machine_types length ({len(machine_types)}) must equal batch_size ({batch_size})"
            )

        if force_anomaly:
            return torch.cat(
                [
                    self._generate_one(
                        device,
                        None if machine_types is None else machine_types[i],
                    )
                    for i in range(batch_size)
                ],
                dim=0,
            )

        masks: list[torch.Tensor] = []
        for i in range(batch_size):
            if random.random() < self.zero_mask_prob:
                masks.append(
                    torch.zeros(
                        1,
                        1,
                        *self.spectrogram_shape,
                        device=device,
                        dtype=torch.float32,
                    )
                )
            else:
                mt = None if machine_types is None else machine_types[i]
                masks.append(self._generate_one(device, mt))
        return torch.cat(masks, dim=0)


# Backward-compatible names (non-stationary class)
SpectromorphicMaskStrategy = NonStationarySpectromorphicMaskStrategy
LatentAlignedBandStrategy = NonStationarySpectromorphicMaskStrategy
AudioSpecificStrategy = NonStationarySpectromorphicMaskStrategy