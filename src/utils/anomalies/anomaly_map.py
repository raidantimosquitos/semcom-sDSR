"""
Anomaly map generation for sDSR training.

Three strategies (same interface: __call__(batch_size, device) -> (B, 1, H, W)):

1. PerlinNoiseStrategy        – threshold/binarize Perlin noise (generic fallback)
2. AudioSpecificStrategy      – generic audio-domain mask (freq band + time segments)
3. MachineSpecificStrategy    – per-machine-type mask derived from DCASE2020 Task 2
                                spectrogram analysis (fan / pump / slider / toycar /
                                toyconveyor / valve)

AnomalyMapGenerator selects between strategies at call time.
"""

from __future__ import annotations

import random
from typing import Literal, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import rotate as ndimage_rotate

from .perlin import rand_perlin_2d_np

# ---------------------------------------------------------------------------
# Helpers for disjoint time-segment sampling (shared by audio_specific and
# machine-specific strategies that need them)
# ---------------------------------------------------------------------------

def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 <= b0 or b1 <= a0)


def _sample_disjoint_time_segments(
    n_seg: int,
    T: int,
    seg_len_lo: int,
    seg_len_hi: int,
    max_tries: int = 400,
) -> list[tuple[int, int]]:
    if T <= 4:
        return [(0, T)]
    lo = max(1, min(seg_len_lo, T))
    hi = min(seg_len_hi, T)
    if hi < lo:
        lo, hi = hi, lo
    intervals: list[tuple[int, int]] = []
    for _ in range(n_seg):
        placed = False
        for _ in range(max_tries):
            seg_len = random.randint(lo, hi)
            if seg_len > T:
                continue
            t_start = random.randint(0, T - seg_len)
            t_end = t_start + seg_len
            if any(_intervals_overlap(t_start, t_end, s, e) for s, e in intervals):
                continue
            intervals.append((t_start, t_end))
            placed = True
            break
        if not placed:
            break
    if not intervals:
        return [(0, min(hi, T))]
    return intervals


# ---------------------------------------------------------------------------
# 1. Perlin noise strategy (generic)
# ---------------------------------------------------------------------------

class PerlinNoiseStrategy:
    """
    Generate anomaly map by thresholding Perlin noise.
    Produces blob-like regions. Optionally rotates noise for varied orientations.
    """

    def __init__(
        self,
        spectrogram_shape: tuple[int, int],
        q_shape: tuple[int, int] | None = None,
        threshold: float = 0.5,
        perlin_scale_range: tuple[int, int] = (0, 6),
        rotate: bool = True,
        rotation_range: tuple[float, float] = (-90.0, 90.0),
    ) -> None:
        self.spectrogram_shape = spectrogram_shape
        # q_shape retained for backward compatibility; output is always spectrogram_shape
        self.q_shape = q_shape or spectrogram_shape
        self.threshold = threshold
        self.perlin_scale_range = perlin_scale_range
        self.rotate = rotate
        self.rotation_range = rotation_range

    def __call__(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        masks = []
        for _ in range(batch_size):
            res_y = 2 ** random.randint(*self.perlin_scale_range)
            res_x = 2 ** random.randint(*self.perlin_scale_range)
            noise = rand_perlin_2d_np(self.spectrogram_shape, (res_y, res_x))
            if self.rotate:
                angle = random.uniform(*self.rotation_range)
                noise = ndimage_rotate(noise, angle, reshape=False, order=1, mode="constant", cval=0)
            binary = (noise > self.threshold).astype(np.float32)
            masks.append(torch.from_numpy(binary).unsqueeze(0).unsqueeze(0))
        M = torch.cat(masks, dim=0).to(device)
        if self.q_shape != self.spectrogram_shape:
            M = F.interpolate(M, size=self.q_shape, mode="nearest")
        return M


# ---------------------------------------------------------------------------
# 2. Generic audio-specific strategy
# ---------------------------------------------------------------------------

class AudioSpecificStrategy:
    """
    Generate anomaly map for spectrograms — frequency band + time segments.
    Generic fallback when no machine-specific strategy is available.
    """

    def __init__(
        self,
        spectrogram_shape: tuple[int, int],
        q_shape: tuple[int, int] | None = None,
        n_mels: int | None = None,
        T: int | None = None,
        min_band_fraction: float = 0.02,
        max_band_fraction: float = 0.50,
        min_segments: int = 2,
        max_segments: int = 8,
    ) -> None:
        self.spectrogram_shape = spectrogram_shape
        self.q_shape = q_shape or spectrogram_shape
        self.n_mels = n_mels or spectrogram_shape[0]
        self.T = T or spectrogram_shape[1]
        self.min_band_fraction = min_band_fraction
        self.max_band_fraction = max_band_fraction
        self.min_segments = max(1, min_segments)
        self.max_segments = max(self.min_segments, max_segments)

    def _single_mask_numpy(self) -> np.ndarray:
        n_mels, T = self.n_mels, self.T
        M = np.zeros((n_mels, T), dtype=np.float32)
        bw = max(1, int(n_mels * random.uniform(self.min_band_fraction, self.max_band_fraction)))
        f_low = random.randint(0, max(0, n_mels - bw))
        f_high = min(n_mels, f_low + bw)
        n_seg = random.randint(self.min_segments, self.max_segments)
        seg_lo = max(4, T // 40)
        seg_hi = max(seg_lo, T // 6)
        for ts, te in _sample_disjoint_time_segments(n_seg, T, seg_lo, seg_hi):
            M[f_low:f_high, ts:te] = 1.0
        return M

    def __call__(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        masks = [
            torch.from_numpy(self._single_mask_numpy()).unsqueeze(0).unsqueeze(0)
            for _ in range(batch_size)
        ]
        M = torch.cat(masks, dim=0).to(device)
        if self.q_shape != self.spectrogram_shape:
            M = F.interpolate(M, size=self.q_shape, mode="nearest")
        return M


# ---------------------------------------------------------------------------
# 3. Machine-specific strategies
# ---------------------------------------------------------------------------

_MACHINE_STRATEGY_CLASSES: dict[str, str] = {
    "fan":          "src.utils.anomalies.machine_specific.fan.FanAnomalyStrategy",
    "pump":         "src.utils.anomalies.machine_specific.pump.PumpAnomalyStrategy",
    "slider":       "src.utils.anomalies.machine_specific.slider.SliderAnomalyStrategy",
    "toycar":       "src.utils.anomalies.machine_specific.toycar.ToyCarAnomalyStrategy",
    "toyconveyor":  "src.utils.anomalies.machine_specific.toyconveyor.ToyConveyorAnomalyStrategy",
    "valve":        "src.utils.anomalies.machine_specific.valve.ValveAnomalyStrategy",
}


def _load_machine_strategy(
    machine_type: str,
    spectrogram_shape: tuple[int, int],
    n_mels: int,
    T: int,
) -> object | None:
    """Lazily instantiate the machine-specific strategy class, or return None if unavailable."""
    key = machine_type.lower()
    if key not in _MACHINE_STRATEGY_CLASSES:
        return None
    module_path, cls_name = _MACHINE_STRATEGY_CLASSES[key].rsplit(".", 1)
    try:
        import importlib
        mod = importlib.import_module(module_path)
        cls = getattr(mod, cls_name)
        return cls(spectrogram_shape=spectrogram_shape, n_mels=n_mels, T=T)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 4. AnomalyMapGenerator — unified entry point
# ---------------------------------------------------------------------------

class AnomalyMapGenerator:
    """
    Generate anomaly map M for training.

    strategy options:
        "perlin"           – Perlin noise only
        "audio_specific"   – generic audio-specific only
        "machine_specific" – per-machine-type strategy (falls back to audio_specific)
        "both"             – 20% Perlin, 80% audio_specific (original default)
        "machine_both"     – 20% Perlin, 80% machine_specific (new recommended)

    When force_anomaly=False, each sample independently receives a zero mask
    with probability zero_mask_prob.
    """

    def __init__(
        self,
        strategy: Literal[
            "perlin", "audio_specific", "machine_specific", "both", "machine_both"
        ],
        spectrogram_shape: tuple[int, int],
        q_shape: tuple[int, int] | None = None,
        n_mels: int | None = None,
        T: int | None = None,
        zero_mask_prob: float = 0.5,
        machine_type: str | None = None,
    ) -> None:
        self.strategy_name = strategy
        self.spectrogram_shape = spectrogram_shape
        # q_shape kept for backward compat but masks are generated at spectrogram_shape
        self.q_shape = q_shape or spectrogram_shape
        self.zero_mask_prob = zero_mask_prob
        self.machine_type = machine_type

        _n_mels = n_mels or spectrogram_shape[0]
        _T = T or spectrogram_shape[1]

        self.perlin = (
            PerlinNoiseStrategy(spectrogram_shape, self.q_shape)
            if strategy in ("perlin", "both", "machine_both")
            else None
        )
        self.audio_specific = (
            AudioSpecificStrategy(spectrogram_shape, self.q_shape, _n_mels, _T)
            if strategy in ("audio_specific", "both")
            else None
        )
        # Machine-specific strategy (may be None if machine_type not recognised)
        self._machine_strategy = None
        if strategy in ("machine_specific", "machine_both") and machine_type is not None:
            self._machine_strategy = _load_machine_strategy(
                machine_type, spectrogram_shape, _n_mels, _T
            )
        # Always keep a generic audio_specific as fallback
        self._fallback = AudioSpecificStrategy(spectrogram_shape, self.q_shape, _n_mels, _T)

    def _get_machine_or_fallback(self) -> object:
        return self._machine_strategy if self._machine_strategy is not None else self._fallback

    def _generate_one(self, device: torch.device | str) -> torch.Tensor:
        """Generate a single non-zero mask (1, 1, H, W)."""
        name = self.strategy_name

        if name == "perlin":
            assert self.perlin is not None
            return self.perlin(1, device)

        if name == "audio_specific":
            assert self.audio_specific is not None
            return self.audio_specific(1, device)

        if name == "machine_specific":
            strat = self._get_machine_or_fallback()
            return strat(1, device)  # type: ignore[operator]

        if name == "both":
            # 20% Perlin, 80% audio_specific
            if random.random() < 0.20:
                assert self.perlin is not None
                return self.perlin(1, device)
            assert self.audio_specific is not None
            return self.audio_specific(1, device)

        if name == "machine_both":
            # 20% Perlin, 80% machine-specific
            if random.random() < 0.20:
                assert self.perlin is not None
                return self.perlin(1, device)
            strat = self._get_machine_or_fallback()
            return strat(1, device)  # type: ignore[operator]

        # Fallback
        return self._fallback(1, device)

    def generate_for_training_sample(
        self,
        device: torch.device | str,
        force_anomaly: bool = True,
        machine_type: str | None = None,
    ) -> torch.Tensor:
        """Generate one mask for a single training sample (convenience wrapper)."""
        if force_anomaly:
            return self._generate_one(device)
        return self.generate(1, device, force_anomaly=False)

    def generate(
        self,
        batch_size: int,
        device: torch.device | str,
        force_anomaly: bool = False,
        machine_types: Sequence[str] | None = None,
    ) -> torch.Tensor:
        """
        Generate batch of anomaly masks.

        Args:
            batch_size: number of masks
            device: torch device
            force_anomaly: if True, always generate real masks (skip zero_mask_prob)
            machine_types: per-sample machine types (ignored; use AnomalyMapGenerator
                           per machine type for per-type control)

        Returns:
            M: (B, 1, n_mels, T) binary mask
        """
        if force_anomaly:
            return torch.cat(
                [self._generate_one(device) for _ in range(batch_size)], dim=0
            )

        masks = []
        for _ in range(batch_size):
            if random.random() < self.zero_mask_prob:
                masks.append(
                    torch.zeros(1, 1, *self.spectrogram_shape, device=device, dtype=torch.float32)
                )
            else:
                masks.append(self._generate_one(device))
        return torch.cat(masks, dim=0)