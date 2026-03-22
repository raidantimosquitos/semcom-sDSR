"""
Anomaly map generation for sDSR training.

Strategies (same interface: __call__(batch_size, device) -> (B, 1, H, W)):
1. PerlinNoiseStrategy: threshold/binarize Perlin noise (DSR-style)
2. AudioSpecificStrategy: choose frequency band + time segments
3. MachineSpecificStrategy: machine-type-specific shapes (fan, pump, slider, valve, ToyCar, ToyConveyor)
"""

from __future__ import annotations

import random
from typing import Callable, Literal

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import rotate as ndimage_rotate

from .perlin import rand_perlin_2d_np

# Minimum mask extent so regions survive max-pool to coarse latent (8×8 cells)
MIN_FREQ_BINS = 4
MIN_TIME_FRAMES = 4


class PerlinNoiseStrategy:
    """
    Generate anomaly map by thresholding Perlin noise (sDSR / MVTec style).
    Produces blob-like anomaly regions. Optionally rotates the noise before
    thresholding (matching original DSR) for more varied blob orientations.
    """

    def __init__(
        self,
        spectrogram_shape: tuple[int, int],
        threshold: float = 0.5,
        perlin_scale_range: tuple[int, int] = (1, 5),
        rotate: bool = True,
        rotation_range: tuple[float, float] = (-120.0, 120.0),
    ) -> None:
        """
        Args:
            spectrogram_shape: (n_mels, T) spectrogram spatial dimensions
            threshold: binarization threshold
            perlin_scale_range: (min_exp, max_exp) for res = 2^randint(min_exp, max_exp)
            rotate: if True, apply random 2D rotation to noise before thresholding
            rotation_range: (min_deg, max_deg) for rotation angle in degrees
        """
        self.spectrogram_shape = spectrogram_shape
        self.threshold = threshold
        n_mels, T = spectrogram_shape
        # Cap max exponent so Perlin wavelength >= MIN_* (blobs survive max-pool)
        max_exp_freq = int(np.log2(max(1, n_mels // MIN_FREQ_BINS)))
        max_exp_time = int(np.log2(max(1, T // MIN_TIME_FRAMES)))
        effective_max = min(max_exp_freq, max_exp_time, perlin_scale_range[1])
        effective_min = min(perlin_scale_range[0], effective_max)
        self.perlin_scale_range = (effective_min, effective_max)
        self.rotate = rotate
        self.rotation_range = rotation_range

    def __call__(
        self,
        batch_size: int,
        device: torch.device | str,
    ) -> torch.Tensor:
        """
        Generate anomaly map M. Each mask uses Perlin noise (optionally rotated)
        then binarized at spectrogram shape.

        Returns:
            M: (B, 1, n_mels, T) binary mask
        """
        masks = []
        for _ in range(batch_size):
            res_y = 2 ** random.randint(*self.perlin_scale_range)
            res_x = 2 ** random.randint(*self.perlin_scale_range)
            res = (res_y, res_x)
            noise = rand_perlin_2d_np(self.spectrogram_shape, res)
            if self.rotate:
                angle = random.uniform(*self.rotation_range)
                noise = ndimage_rotate(
                    noise, angle, reshape=False, order=1, mode="constant", cval=0
                )
            binary = (noise > self.threshold).astype(np.float32)
            mask = torch.from_numpy(binary).unsqueeze(0).unsqueeze(0)
            masks.append(mask)
        M = torch.cat(masks, dim=0).to(device)
        return M


def _draw_fan(n_mels: int, T: int) -> np.ndarray:
    """
    Horizontal stripes at harmonic-like intervals; freq-specific, not broadband."""
    M = np.zeros((n_mels, T), dtype=np.float32)
    n_stripes = random.randint(1, 2)
    stripe_height_narrow = random.randint(4, 9)
    stripe_height_narrow = max(MIN_FREQ_BINS, min(stripe_height_narrow, n_mels))
    for _ in range(n_stripes):
        f_low = random.randint(0, max(0, n_mels - stripe_height_narrow))
        f_high = min(f_low + stripe_height_narrow, n_mels)
        M[f_low:f_high, :] = 1.0
    return M


def _draw_pump(n_mels: int, T: int) -> np.ndarray:
    """Periodic vertical bands in lower-mid freq (~30–90); preserve period."""
    M = np.zeros((n_mels, T), dtype=np.float32)
    n_stripes = random.randint(3, 6)
    stripe_width = random.randint(MIN_TIME_FRAMES, max(MIN_TIME_FRAMES, 20))
    max_start = max(0, T - stripe_width)
    if max_start <= 0:
        M[:, : min(stripe_width, T)] = 1.0
        return M
    pool = list(range(max_start))
    k = min(n_stripes, len(pool))
    positions = sorted(random.sample(pool, k))
    for t_start in positions:
        t_end = min(t_start + stripe_width, T)
        M[:, t_start:t_end] = 1.0
    return M


def _draw_slider(n_mels: int, T: int) -> np.ndarray:
    """Narrow full-height vertical stripes at 3–5 positions; broadband."""
    M = np.zeros((n_mels, T), dtype=np.float32)
    n_stripes = random.randint(3, 6)
    stripe_width = random.randint(MIN_TIME_FRAMES, max(MIN_TIME_FRAMES, 20))
    max_start = max(0, T - stripe_width)
    if max_start <= 0:
        M[:, : min(stripe_width, T)] = 1.0
        return M
    pool = list(range(max_start))
    k = min(n_stripes, len(pool))
    positions = sorted(random.sample(pool, k))
    for t_start in positions:
        t_end = min(t_start + stripe_width, T)
        M[:, t_start:t_end] = 1.0
    return M


def _draw_valve(n_mels: int, T: int) -> np.ndarray:
    """Small rectangular patches at periodic time, variable freq; tonally specific."""
    M = np.zeros((n_mels, T), dtype=np.float32)
    patch_h = random.randint(12, 40)
    patch_w = random.randint(4, 9)
    patch_h = max(MIN_FREQ_BINS, min(patch_h, n_mels))
    patch_w = max(MIN_TIME_FRAMES, min(patch_w, T))
    n_patches = random.randint(2, 5)
    period = max(patch_w + 10, T // (n_patches + 1))
    for i in range(n_patches):
        t_start = min((i * period) + random.randint(0, 20), T - patch_w)
        t_start = max(0, t_start)
        t_end = min(t_start + patch_w, T)
        f_low = random.randint(0, max(0, n_mels - patch_h))
        f_high = min(f_low + patch_h, n_mels)
        if t_end - t_start >= MIN_TIME_FRAMES and f_high - f_low >= MIN_FREQ_BINS:
            M[f_low:f_high, t_start:t_end] = 1.0
    return M


def _draw_toycar(n_mels: int, T: int) -> np.ndarray:
    """Medium blobs in mid-freq (30–90); slightly irregular."""
    M = np.zeros((n_mels, T), dtype=np.float32)
    f_low = max(0, min(30, n_mels - MIN_FREQ_BINS))
    f_high = min(91, n_mels)
    n_blobs = random.randint(1, 4)
    for _ in range(n_blobs):
        bh = random.randint(4, 9)
        bw = random.randint(8, 13)
        bh = max(MIN_FREQ_BINS, min(bh, f_high - f_low))
        bw = max(MIN_TIME_FRAMES, min(bw, T))
        t_start = random.randint(0, max(0, T - bw))
        t_end = min(t_start + bw, T)
        M[f_low:f_high, t_start:t_end] = 1.0
    return M


def _draw_toyconveyor(n_mels: int, T: int) -> np.ndarray:
    """Low-freq horizontal band (bottom ~20–30 bins) + periodic patches at belt period."""
    M = np.zeros((n_mels, T), dtype=np.float32)
    band_height = random.randint(20, 31)
    band_height = min(band_height, n_mels)
    M[:band_height, :] = 1.0
    period = random.randint(45, 55)
    patch_w = random.randint(MIN_TIME_FRAMES, 25)
    patch_h = random.randint(MIN_FREQ_BINS, 25)
    for i in range(2):
        t_center = (i + 1) * period + random.randint(-8, 8)
        t_start = max(0, t_center - patch_w // 2)
        t_end = min(T, t_start + patch_w)
        f_low = random.randint(0, max(0, n_mels - patch_h))
        f_high = min(f_low + patch_h, n_mels)
        if t_end - t_start >= MIN_TIME_FRAMES:
            M[f_low:f_high, t_start:t_end] = 1.0
    return M


_MACHINE_DRAWERS: dict[str, Callable[[int, int], np.ndarray]] = {
    "fan": _draw_fan,
    "pump": _draw_pump,
    "slider": _draw_slider,
    "valve": _draw_valve,
    "ToyCar": _draw_toycar,
    "ToyConveyor": _draw_toyconveyor,
}


class MachineSpecificStrategy:
    """
    Generate anomaly masks with machine-type-specific shapes (spectrogram space).
    Coarse/fine latent masks are derived from this single mask via max-pool elsewhere.
    """

    def __init__(
        self,
        spectrogram_shape: tuple[int, int],
        n_mels: int,
        T: int,
        machine_type: str,
    ) -> None:
        self.spectrogram_shape = spectrogram_shape
        self.n_mels = n_mels
        self.T = T
        self._types = [t.strip() for t in machine_type.split("+") if t.strip()]

    def __call__(
        self,
        batch_size: int,
        device: torch.device | str,
    ) -> torch.Tensor:
        n_mels, T = self.n_mels, self.T
        masks = []
        for _ in range(batch_size):
            if self._types:
                chosen = random.choice(self._types)
                drawer = _MACHINE_DRAWERS.get(chosen)
                if drawer is not None:
                    M = drawer(n_mels, T)
                    mask = torch.from_numpy(M.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                else:
                    mask = torch.zeros(1, 1, n_mels, T, device=device, dtype=torch.float32)
            else:
                mask = torch.zeros(1, 1, n_mels, T, device=device, dtype=torch.float32)
            masks.append(mask)
        return torch.cat(masks, dim=0).to(device)


class AnomalyMapGenerator:
    """
    Generate anomaly map M for training using one of several strategies.
    When force_anomaly=False, each sample gets an independent draw: with
    probability zero_mask_prob a zero mask (no anomaly), else a generated mask.
    """

    def __init__(
        self,
        strategy: Literal["perlin", "audio_specific", "both", "machine_specific"],
        spectrogram_shape: tuple[int, int],
        n_mels: int | None = None,
        T: int | None = None,
        zero_mask_prob: float = 0.5,
        machine_type: str | None = None,
    ) -> None:
        """
        Args:
            strategy: 'perlin', 'audio_specific', 'both', or 'machine_specific'
            spectrogram_shape: (n_mels, T)
            n_mels, T: required for audio_specific and machine_specific
            zero_mask_prob: per-sample probability of returning a zero mask (no anomaly)
            machine_type: required when strategy == 'machine_specific' (e.g. 'fan' or 'ToyCar+ToyConveyor+fan+...')
        """
        self.strategy_name = strategy
        self.spectrogram_shape = spectrogram_shape
        self.zero_mask_prob = zero_mask_prob
        self.perlin = (
            PerlinNoiseStrategy(spectrogram_shape)
            if strategy in ("perlin", "both")
            else None
        )
        self.machine_specific = (
            MachineSpecificStrategy(spectrogram_shape, n_mels, T, machine_type=machine_type or "")
            if strategy in ("both", "machine_specific") and n_mels is not None and T is not None
            else None
        )

    def _generate_one(self, device: torch.device | str) -> torch.Tensor:
        """Generate a single non-zero mask (1, 1, H, W)."""
        if self.strategy_name == "perlin":
            assert self.perlin is not None
            return self.perlin(1, device)
        if self.strategy_name == "machine_specific":
            assert self.machine_specific is not None
            return self.machine_specific(1, device)
        if random.random() < 0.2:
            assert self.perlin is not None
            return self.perlin(1, device)
        assert self.machine_specific is not None
        return self.machine_specific(1, device)

    def generate(
        self,
        batch_size: int,
        device: torch.device | str,
        force_anomaly: bool = False,
    ) -> torch.Tensor:
        """
        Generate anomaly map M.

        When force_anomaly=False, each sample is decided independently: with
        probability zero_mask_prob the mask is all zeros, else one mask from
        the strategy (per-sample 50% no-anomaly, matching original DSR).

        Args:
            batch_size: number of masks to generate
            device: torch device
            force_anomaly: if True, skip zero_mask_prob and always generate real masks

        Returns:
            M: (B, 1, n_mels, T) binary mask
        """
        if force_anomaly:
            if self.strategy_name == "perlin":
                assert self.perlin is not None
                return self.perlin(batch_size, device)
            if self.strategy_name == "machine_specific":
                assert self.machine_specific is not None
                return self.machine_specific(batch_size, device)
            if random.random() < 0.2:
                assert self.perlin is not None
                return self.perlin(batch_size, device)
            assert self.machine_specific is not None
            return self.machine_specific(batch_size, device)

        masks = []
        for _ in range(batch_size):
            if random.random() < self.zero_mask_prob:
                masks.append(
                    torch.zeros(
                        1, 1, *self.spectrogram_shape,
                        device=device, dtype=torch.float32
                    )
                )
            else:
                masks.append(self._generate_one(device))
        return torch.cat(masks, dim=0)
