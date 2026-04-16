"""
Anomaly map generation for sDSR training.

Strategies (same interface: __call__(batch_size, device) -> (B, 1, H, W)):

1. PerlinNoiseStrategy            – threshold/binarize Perlin noise (generic fallback)
2. MixNoiseStrategy               – smoothed random field + Poisson bursts, quantile cut
3. SpectromorphicMaskStrategy     – four sub-strategies sampled per-item: full_band,
                                    multi_band, diffuse_rect, perlin (replaces
                                    LatentAlignedBandStrategy; includes Perlin internally)
4. MachineSpecificStrategy        – per-machine-type mask derived from DCASE2020 Task 2
                                    spectrogram analysis (fan / pump / slider / toycar /
                                    toyconveyor / valve)

AnomalyMapGenerator selects between strategies at call time.
``AudioSpecificStrategy`` and ``LatentAlignedBandStrategy`` are kept as aliases
for backward compatibility.
"""

from __future__ import annotations

import random
from typing import Literal, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter, rotate as ndimage_rotate

from .perlin import rand_perlin_2d_np

# ---------------------------------------------------------------------------
# Helpers for disjoint time-segment sampling (used by machine-specific
# strategies that need them)
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
# 1b. Mix strategy (smooth field + structured bursts)
# ---------------------------------------------------------------------------

class MixNoiseStrategy:
    """
    Smooth random base (Gaussian-blurred white noise or optional Perlin) plus Poisson-count
    structured additions, then z-score and quantile threshold.

    Array layout is (n_mels, T): row index = mel / frequency, column = time. Modes: ``v``
    adds a Gaussian ridge along time (broadband temporal event); ``h`` adds a Gaussian ridge
    along frequency (narrowband over time); ``d`` a thickened diagonal; ``blob`` a 2D Gaussian.
    """

    def __init__(
        self,
        spectrogram_shape: tuple[int, int],
        q_shape: tuple[int, int] | None = None,
        lambda_k: float = 3.0,
        quantile: float = 0.98,
        base: Literal["smoothed_white", "perlin"] = "smoothed_white",
        perlin_scale_range: tuple[int, int] = (0, 6),
        diag_half_width: int = 1,
        norm_eps: float = 1e-6,
    ) -> None:
        self.spectrogram_shape = spectrogram_shape
        self.q_shape = q_shape or spectrogram_shape
        self.lambda_k = max(0.0, lambda_k)
        self.quantile = quantile
        self.base = base
        self.perlin_scale_range = perlin_scale_range
        self.diag_half_width = max(0, diag_half_width)
        self.norm_eps = norm_eps

    def _scaled_smooth_sigmas(self, H: int, W: int) -> tuple[float, float]:
        sigma_f = max(1e-3, 8.0 * H / 128.0)
        sigma_t = max(1e-3, 20.0 * W / 320.0)
        return sigma_f, sigma_t

    def _single_mask_numpy(self) -> np.ndarray:
        H, W = self.spectrogram_shape
        sigma_f0, sigma_t0 = self._scaled_smooth_sigmas(H, W)

        if self.base == "perlin":
            res_y = 2 ** random.randint(*self.perlin_scale_range)
            res_x = 2 ** random.randint(*self.perlin_scale_range)
            Z = rand_perlin_2d_np((H, W), (res_y, res_x)).astype(np.float64)
            blur_f = max(0.5, sigma_f0 / 4.0)
            blur_t = max(0.5, sigma_t0 / 4.0)
            Z = gaussian_filter(Z, sigma=(blur_f, blur_t))
        else:
            Z = np.random.randn(H, W).astype(np.float64)
            Z = gaussian_filter(Z, sigma=(sigma_f0, sigma_t0))

        scale_h = H / 128.0
        scale_w = W / 320.0

        K = int(np.random.poisson(self.lambda_k)) if self.lambda_k > 0 else 0
        for _ in range(K):
            typ = random.choice(["v", "h", "d", "blob"])

            if typ == "v":
                t0 = random.randint(0, W - 1) if W > 0 else 0
                sigma_t = float(np.random.uniform(2.0, 10.0) * scale_w)
                sigma_t = max(sigma_t, 1e-3)
                t_ax = np.arange(W, dtype=np.float64)
                row = np.exp(-((t_ax - t0) ** 2) / (2.0 * sigma_t**2))
                Z += row[np.newaxis, :]

            elif typ == "h":
                f0 = random.randint(0, H - 1) if H > 0 else 0
                sigma_f = float(np.random.uniform(1.0, 5.0) * scale_h)
                sigma_f = max(sigma_f, 1e-3)
                f_ax = np.arange(H, dtype=np.float64)
                col = np.exp(-((f_ax - f0) ** 2) / (2.0 * sigma_f**2))
                Z += col[:, np.newaxis]

            elif typ == "d":
                a = float(np.random.uniform(-0.2, 0.2))
                b = float(np.random.uniform(0.0, float(H)))
                hw = self.diag_half_width
                for t in range(W):
                    f_line = int(round(a * t + b))
                    for df in range(-hw, hw + 1):
                        ff = f_line + df
                        if 0 <= ff < H:
                            Z[ff, t] += 1.0

            else:
                f0 = random.randint(0, H - 1) if H > 0 else 0
                t0 = random.randint(0, W - 1) if W > 0 else 0
                sigma_f = float(np.random.uniform(3.0, 10.0) * scale_h)
                sigma_t = float(np.random.uniform(5.0, 20.0) * scale_w)
                sigma_f = max(sigma_f, 1e-3)
                sigma_t = max(sigma_t, 1e-3)
                F, Tgrid = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
                Z += np.exp(
                    -((F - f0) ** 2) / (2.0 * sigma_f**2)
                    -((Tgrid - t0) ** 2) / (2.0 * sigma_t**2)
                )

        z_mean = float(Z.mean())
        z_std = max(float(Z.std()), self.norm_eps)
        Z = (Z - z_mean) / z_std
        tau = float(np.quantile(Z, self.quantile))
        return (Z > tau).astype(np.float32)

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
# Per-machine-type mask geometry presets (derived from L2 distance maps)
# ---------------------------------------------------------------------------

MASK_PRESETS: dict[str, dict] = {
    "pump":         {"full_time_prob": 0.2, "max_band_frac": 0.15, "max_segments": 8, "perlin_prob": 0.2},
    "slider":       {"full_time_prob": 0.2, "max_band_frac": 0.15, "max_segments": 8, "perlin_prob": 0.2},
    "valve":        {"full_time_prob": 0.2, "max_band_frac": 0.15, "max_segments": 8, "perlin_prob": 0.2},
    "ToyCar":       {"full_time_prob": 1.0, "max_band_frac": 0.15, "max_segments": 8, "perlin_prob": 0.2},
    "ToyConveyor":  {"full_time_prob": 1.0, "max_band_frac": 0.15, "max_segments": 8, "perlin_prob": 0.2},
    "fan":          {"full_time_prob": 0.85, "max_band_frac": 0.10, "max_segments": 4, "perlin_prob": 0.12},
}


# ---------------------------------------------------------------------------
# 2. SpectromorphicMaskStrategy (replaces LatentAlignedBandStrategy)
# ---------------------------------------------------------------------------

class SpectromorphicMaskStrategy:
    """
    Anomaly mask generation following the AudDSR paper approach.

    Two processes mixed per-item:

      1. **Band + time segments** (``1 - perlin_prob``):
         One frequency band of random width is chosen, then 1-N disjoint time
         segments within that band are marked as anomalous.  Most of the time
         a single segment covers the full duration (dominant real pattern);
         occasionally 2-5 shorter segments are used for variety.

      2. **Perlin noise** (``perlin_prob``):
         Thresholded and rotated Perlin noise for blob-shaped regularisation
         that prevents overfitting to rectangular masks.

    All masks are binary, shape (1, 1, n_mels, T) at spectrogram resolution.
    Callers project to fine/coarse grids with avg_pool + (> 0) threshold.

    Args:
        n_mels: spectrogram height (128)
        T: spectrogram width (320)
        perlin_prob: probability of choosing Perlin mode per item
        full_time_prob: within band mode, probability that a single segment
            covers 85-100 % of the time axis (vs. multiple shorter segments)
        max_band_frac: maximum band width as a fraction of n_mels
        max_segments: maximum number of disjoint time segments when not
            using full-time mode
    """

    def __init__(
        self,
        spectrogram_shape: tuple[int, int] | None = None,
        q_shape: tuple[int, int] | None = None,
        n_mels: int | None = None,
        T: int | None = None,
        perlin_prob: float = 0.4,
        full_time_prob: float = 0.3,
        max_band_frac: float = 0.1,
        max_segments: int = 8,
        **_kwargs: object,
    ) -> None:
        if spectrogram_shape is not None:
            self.n_mels = n_mels or spectrogram_shape[0]
            self.T = T or spectrogram_shape[1]
        else:
            self.n_mels = n_mels or 128
            self.T = T or 320
        self.q_shape = q_shape or (self.n_mels, self.T)
        self.perlin_prob = perlin_prob
        self.full_time_prob = full_time_prob
        self.max_band_width = max(6, int(self.n_mels * max_band_frac))
        self.max_segments = max(1, max_segments)

    # -- band + time segments --------------------------------------------------

    def _band_time_segments(self) -> np.ndarray:
        n_mels, T = self.n_mels, self.T
        mask = np.zeros((n_mels, T), dtype=np.float32)

        band_w = random.randint(4, self.max_band_width)
        f0 = random.randint(0, n_mels - band_w)

        if random.random() < self.full_time_prob:
            coverage = random.uniform(0.75, 1.0)
            t_len = max(1, int(T * coverage))
            t0 = random.randint(0, max(0, T - t_len))
            mask[f0 : f0 + band_w, t0 : t0 + t_len] = 1.0
        else:
            n_seg = random.randint(2, self.max_segments)
            seg_lo = max(1, T // 20)
            seg_hi = max(seg_lo, T // 4)
            segments = _sample_disjoint_time_segments(n_seg, T, seg_lo, seg_hi)
            for t0, t1 in segments:
                mask[f0 : f0 + band_w, t0:t1] = 1.0

        return mask

    # -- perlin ----------------------------------------------------------------

    def _perlin(self) -> np.ndarray:
        res_y = 2 ** random.randint(3, 4)
        res_x = 2 ** random.randint(1, 2)
        noise = rand_perlin_2d_np((self.n_mels, self.T), (res_y, res_x))
        # angle = random.uniform(-90.0, 90.0)
        # noise = ndimage_rotate(noise, angle, reshape=False, order=1,
        #                       mode="constant", cval=0.0)
        threshold = random.uniform(0.3, 0.6)
        return (noise > threshold).astype(np.float32)

    # -- public interface ------------------------------------------------------

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


# Backward-compatible aliases
LatentAlignedBandStrategy = SpectromorphicMaskStrategy
AudioSpecificStrategy = SpectromorphicMaskStrategy


# ---------------------------------------------------------------------------
# 4. AnomalyMapGenerator — unified entry point
# ---------------------------------------------------------------------------

class AnomalyMapGenerator:
    """
    Generate anomaly map M for training.

    strategy options:
        "perlin"           – Perlin noise only
        "mix"              – smoothed field + burst mix, quantile threshold
        "audio_specific"   – SpectromorphicMaskStrategy (recommended; includes
                             Perlin internally so a separate "both" mode is
                             no longer needed)

    When force_anomaly=False, each sample independently receives a zero mask
    with probability zero_mask_prob.
    """

    def __init__(
        self,
        strategy: Literal[
            "perlin",
            "mix",
            "audio_specific"
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
        self.q_shape = q_shape or spectrogram_shape
        self.zero_mask_prob = zero_mask_prob

        _n_mels = n_mels or spectrogram_shape[0]
        _T = T or spectrogram_shape[1]

        self.perlin = (
            PerlinNoiseStrategy(spectrogram_shape, self.q_shape)
            if strategy == "perlin"
            else None
        )
        self.mix = (
            MixNoiseStrategy(spectrogram_shape, self.q_shape)
            if strategy == "mix"
            else None
        )
        self.audio_specific = (
            AudioSpecificStrategy(
                spectrogram_shape, self.q_shape, _n_mels, _T,
                machine_type=machine_type,
            )
            if strategy == "audio_specific"
            else None
        )
        self._fallback = AudioSpecificStrategy(
            spectrogram_shape, self.q_shape, _n_mels, _T,
            machine_type=machine_type,
        )

    def _generate_one(self, device: torch.device | str) -> torch.Tensor:
        """Generate a single non-zero mask (1, 1, H, W)."""
        name = self.strategy_name

        if name == "perlin":
            assert self.perlin is not None
            return self.perlin(1, device)

        if name == "mix":
            assert self.mix is not None
            return self.mix(1, device)

        if name == "audio_specific":
            assert self.audio_specific is not None
            return self.audio_specific(1, device)

        return self._fallback(1, device)

    def generate_for_training_sample(
        self,
        device: torch.device | str,
        force_anomaly: bool = True,
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
    ) -> torch.Tensor:
        """
        Generate batch of anomaly masks.

        Args:
            batch_size: number of masks
            device: torch device
            force_anomaly: if True, always generate real masks (skip zero_mask_prob)

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