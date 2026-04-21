"""
Anomaly map generation for sDSR training.

Strategies (same interface: ``__call__(batch_size, device)`` → ``(B, 1, H, W)``):

1. NonStationarySpectromorphicMaskStrategy – stratified mel band × renewal time,
   optional Perlin (e.g. valve, slider). Aliased as ``SpectromorphicMaskStrategy``.
2. StationarySpectromorphicMaskStrategy – harmonics, AM, Perlin
   (fan, pump, ToyConveyor, ToyCar).

:class:`AnomalyMapGenerator` holds both and dispatches from the DCASE ``machine_type``
argument on each generate call.
"""

from __future__ import annotations

import math
import random
import numpy as np
import torch
import torch.nn.functional as F

from .perlin import rand_perlin_2d_np

# DCASE machine types: stationary (dense tones) vs non-stationary mask family.
STATIONARY_SPECTROMORPHIC_MACHINE_TYPES: frozenset[str] = frozenset(
    {"fan", "ToyConveyor", "ToyCar"}
)


def default_spectromorphic_perlin(n_mels: int, T: int) -> np.ndarray:
    """Thresholded 2-D Perlin noise (same recipe as the non-stationary strategy's Perlin branch)."""
    res_y = 2 ** random.randint(1, 4)
    res_x = 2 ** random.randint(2, 5)
    noise = rand_perlin_2d_np((n_mels, T), (res_y, res_x))
    threshold = random.uniform(0.3, 0.6)
    return (noise > threshold).astype(np.float32)


def uses_stationary_spectromorphic_mask(machine_type: str | None) -> bool:
    """True iff ``machine_type`` should use stationary spectromorphic masks."""
    if machine_type is None:
        return False
    return machine_type in STATIONARY_SPECTROMORPHIC_MACHINE_TYPES


class NonStationarySpectromorphicMaskStrategy:
    """
    Non-stationary synthetic anomaly masks for spectrograms.

    Each sample is either:

      1. **Band + renewal time** (probability ``1 - perlin_prob``): exactly one
         contiguous mel band. The band is **stratified by mel**: each mask picks
         one stratum (low / mid / high mel when ``mel_n_strata=3``), then a
         start position and width so the band intersects that stratum (with
         fallback if impossible). Time activation is alternating renewal with
         geometric run lengths; per-mask means are log-uniform in ``[1, T]``.

      2. **Perlin** (probability ``perlin_prob``): thresholded 2-D Perlin noise
         for blob-shaped masks.

    Output is binary, shape ``(1, 1, n_mels, T)``. Callers may downsample with
    ``F.interpolate`` when ``q_shape`` differs from spectrogram resolution.

    Args:
        perlin_prob: probability of choosing the Perlin branch per mask.
        mel_n_strata: number of contiguous mel partitions for stratified band
            placement (e.g. ``3`` → low / mid / high). ``1`` disables stratification
            (uniform mel start as before).
    """

    def __init__(
        self,
        spectrogram_shape: tuple[int, int] | None = None,
        q_shape: tuple[int, int] | None = None,
        n_mels: int | None = None,
        T: int | None = None,
        perlin_prob: float = 0.05,
        mel_n_strata: int = 3,
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
        self.mel_n_strata = max(1, int(mel_n_strata))

    @staticmethod
    def _mel_stratum_bounds(n_mels: int, n_strata: int, s: int) -> tuple[int, int]:
        """Half-open mel index interval [L, R) for stratum ``s``."""
        L = s * n_mels // n_strata
        R = (s + 1) * n_mels // n_strata
        return L, R

    def _pick_single_stratified_mel_band(self) -> list[tuple[int, int]]:
        """
        One contiguous band ``[start_f, start_f + bw)`` intersecting a randomly
        chosen mel stratum. Bandwidth policy unchanged; if no valid ``start_f``,
        retry a few times then fall back to uniform start on ``[0, n_mels-bw]``.
        """
        n_mels = self.n_mels
        n_strata = self.mel_n_strata
        if n_mels <= 0:
            return []

        bw_lo = 1
        bw_hi = max(bw_lo, max(2, n_mels // 32))
        if random.random() < 0.5:
            bw_hi = max(bw_hi, n_mels // 8)

        bw_hi = min(bw_hi, n_mels)

        if n_strata <= 1:
            bw = random.randint(bw_lo, bw_hi)
            start_f = random.randint(0, n_mels - bw)
            return [(start_f, start_f + bw)]

        for _ in range(24):
            bw = random.randint(bw_lo, bw_hi)
            s = random.randint(0, n_strata - 1)
            L_s, R_s = self._mel_stratum_bounds(n_mels, n_strata, s)
            if R_s <= L_s:
                continue
            # [start_f, start_f + bw) intersects [L_s, R_s) iff start_f < R_s and start_f + bw > L_s
            lo = max(0, L_s - bw + 1)
            hi = min(n_mels - bw, R_s - 1)
            if lo <= hi:
                start_f = random.randint(lo, hi)
                return [(start_f, start_f + bw)]

        bw = random.randint(bw_lo, bw_hi)
        start_f = random.randint(0, n_mels - bw)
        return [(start_f, start_f + bw)]

    def _hierarchical_renewal_params(self, T: int) -> tuple[float, float, float, float]:
        """
        Geometric run lengths: mean on-run ``mu_on``, mean off-run ``mu_off``,
        each log-uniform in [1, T]. Return ``(p_on, p_off, mu_on, mu_off)`` with
        ``p = 1/mu`` for ``np.random.geometric``.
        """
        Tm = max(T, 1)
        lo = math.log(1.0)
        hi = math.log(float(Tm))
        mu_on = math.exp(random.uniform(lo, hi//2)) # before was hi
        # mu_on = math.exp(random.uniform(math.log(50), math.log(T)))
        mu_off = math.exp(random.uniform(lo, hi)) # before was hi
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
        Binary mask: one stratified mel band × time from alternating renewal
        (geometric runs, hierarchical means).
        """
        n_mels, T = self.n_mels, self.T
        mask = np.zeros((n_mels, T), dtype=np.float32)
        bands = self._pick_single_stratified_mel_band()
        if not bands:
            return mask

        f0, f1 = bands[0]
        p_on, p_off, mu_on, mu_off = self._hierarchical_renewal_params(T)
        denom = mu_on + mu_off
        p_start_on = (mu_on / denom) if denom > 0 else 0.5
        track = self._alternating_renewal_1d(
            T, p_on, p_off, random.random() < p_start_on
        )
        mask[f0:f1, :] = track.reshape(1, -1)
        return mask

    def _perlin(self) -> np.ndarray:
        return default_spectromorphic_perlin(self.n_mels, self.T)

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


class StationarySpectromorphicMaskStrategy:
    """
    Masks for machines with dense continuous tones (fans, pumps, conveyors).

    Each sample is one of: multi-band narrow harmonics with high temporal duty
    cycle, a single band with sinusoidal AM in time, or thresholded Perlin noise.
    Mixture: 70% / 15% / 15% by default (single ``random()`` split).

    Output is binary ``(B, 1, n_mels, T)`` with optional ``q_shape`` interpolation,
    Same tensor layout and ``q_shape`` handling as :class:`NonStationarySpectromorphicMaskStrategy`.
    """

    def __init__(
        self,
        spectrogram_shape: tuple[int, int] | None = None,
        q_shape: tuple[int, int] | None = None,
        n_mels: int | None = None,
        T: int | None = None,
        p_multi_band: float = 0.7,
        p_amplitude_mod: float = 0.85,
        **_kwargs: object,
    ) -> None:
        if spectrogram_shape is not None:
            self.n_mels = n_mels or spectrogram_shape[0]
            self.T = T or spectrogram_shape[1]
        else:
            self.n_mels = n_mels or 128
            self.T = T or 320
        self.q_shape = q_shape or (self.n_mels, self.T)
        self.p_multi_band = float(min(max(p_multi_band, 0.0), 1.0))
        self.p_amplitude_mod = float(min(max(p_amplitude_mod, self.p_multi_band), 1.0))

    def _multi_band_continuous(self) -> np.ndarray:
        """Narrow harmonic bands, high temporal duty cycle, optional mel jitter."""
        n_mels, T = self.n_mels, self.T
        mask = np.zeros((n_mels, T), dtype=np.float32)
        if n_mels <= 0 or T <= 0:
            return mask

        n_bands = random.randint(2, 5)
        denom = max(2 * n_bands, 1)
        step = max(1, n_mels // denom)
        f0_start = random.randint(0, max(0, n_mels // 4))

        for i in range(n_bands):
            f_center = f0_start + i * step + random.randint(-2, 2)
            f_center = int(max(0, min(f_center, n_mels - 1)))
            if f_center >= n_mels - 2:
                break
            bw = random.randint(1, 2)
            f_start = max(0, f_center - bw // 2)
            f_end = min(n_mels, f_start + bw)
            if f_end <= f_start:
                continue
            duty_cycle = random.uniform(0.8, 1.0)
            temporal_mask = (np.random.rand(T) < duty_cycle).astype(np.float32)
            mask[f_start:f_end, :] = np.maximum(
                mask[f_start:f_end, :],
                temporal_mask.reshape(1, -1),
            )
        return mask

    def _amplitude_modulation_mask(self) -> np.ndarray:
        """Single narrow band, sinusoidal envelope in time, thresholded."""
        n_mels, T = self.n_mels, self.T
        mask = np.zeros((n_mels, T), dtype=np.float32)
        if n_mels <= 0 or T <= 0:
            return mask

        bw = random.randint(1, 3)
        f_start = random.randint(0, max(0, n_mels - bw))
        mod_freq = random.uniform(0.05, 0.2)
        t = np.linspace(0.0, 2.0 * math.pi * mod_freq, T, dtype=np.float64)
        modulation = 0.5 + 0.5 * np.sin(t)
        threshold = random.uniform(0.3, 0.7)
        temporal_mask = (modulation > threshold).astype(np.float32)
        mask[f_start : f_start + bw, :] = temporal_mask.reshape(1, -1)
        return mask

    def _perlin(self) -> np.ndarray:
        return default_spectromorphic_perlin(self.n_mels, self.T)

    def __call__(
        self,
        batch_size: int,
        device: torch.device | str,
    ) -> torch.Tensor:
        masks = []
        p0, p1 = self.p_multi_band, self.p_amplitude_mod
        for _ in range(batch_size):
            u = random.random()
            if u < p0:
                m = self._multi_band_continuous()
            elif u < p1:
                m = self._amplitude_modulation_mask()
            else:
                m = self._perlin()
            masks.append(torch.from_numpy(m).unsqueeze(0).unsqueeze(0))
        M = torch.cat(masks, dim=0).to(device)
        if self.q_shape != (self.n_mels, self.T):
            M = F.interpolate(M, size=self.q_shape, mode="nearest")
        return M


# ---------------------------------------------------------------------------
# 4. AnomalyMapGenerator — unified entry point
# ---------------------------------------------------------------------------

class AnomalyMapGenerator:
    """
    Build spectromorphic anomaly masks for training.

    Holds both stationary and non-stationary strategies. Each call selects
    :class:`StationarySpectromorphicMaskStrategy` vs :class:`NonStationarySpectromorphicMaskStrategy`
    from the DCASE ``machine_type`` (``None`` → non-stationary).

    When ``force_anomaly=False``, each index independently receives an all-zero mask
    with probability ``zero_mask_prob``.
    """

    def __init__(
        self,
        spectrogram_shape: tuple[int, int],
        q_shape: tuple[int, int] | None = None,
        n_mels: int | None = None,
        T: int | None = None,
        zero_mask_prob: float = 0.5,
    ) -> None:
        self.spectrogram_shape = spectrogram_shape
        self.q_shape = q_shape or spectrogram_shape
        self.zero_mask_prob = zero_mask_prob

        _n_mels = n_mels or spectrogram_shape[0]
        _T = T or spectrogram_shape[1]

        self._nonstationary = NonStationarySpectromorphicMaskStrategy(
            spectrogram_shape, self.q_shape, _n_mels, _T,
        )
        self._stationary = StationarySpectromorphicMaskStrategy(
            spectrogram_shape, self.q_shape, _n_mels, _T,
        )
        # Backward compatibility: direct use of the non-stationary strategy only.
        self.audio_specific = self._stationary

    def _strategy(
        self,
        machine_type: str | None,
    ) -> NonStationarySpectromorphicMaskStrategy | StationarySpectromorphicMaskStrategy:
        return (
            self._stationary
            if uses_stationary_spectromorphic_mask(machine_type)
            else self._nonstationary
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