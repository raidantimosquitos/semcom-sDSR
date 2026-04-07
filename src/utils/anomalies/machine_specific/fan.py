"""
Fan anomaly mask generator.

Observed pattern (DCASE2020 Task 2):
    - Full-duration (or near-full) horizontal frequency bands.
    - Concentrated in mid-to-low mel range (~bins 15–65).
    - Often 1–3 overlapping bands of varying width (10–30 bins each).
    - id_00: mid-frequency broadband noise across ~mel 30–70
    - id_02: strong band around mel 15–40, attenuated highs
    - id_04: wide mid-range ~mel 20–70
    - id_06: sharp drop-outs near mel 20–40, narrow dark stripes

Strategy:
    1. Pick 1–3 frequency bands in mel[lo_range, hi_range].
    2. Each band spans T_frac * T time frames (0.7–1.0 of full duration),
       starting at a small random offset so some variability in start/end.
    3. Add a secondary faint band with probability 0.4 to mimic the
       multi-stripe appearance of id_06.
    4. Bands may partially overlap (union mask).
"""

from __future__ import annotations

import random
import numpy as np
import torch


class FanAnomalyStrategy:
    """
    Synthetic anomaly masks for the Fan machine type.

    Call signature: strategy(batch_size, device) -> (B, 1, n_mels, T) binary mask
    """

    # mel bin ranges where fan anomalies concentrate (inclusive low, exclusive high)
    _BAND_RANGE_LO = (10, 50)   # (lo_min, lo_max) for band start
    _BAND_RANGE_HI = (10, 30)   # (width_min, width_max) for band width in bins

    def __init__(
        self,
        spectrogram_shape: tuple[int, int],
        n_mels: int,
        T: int,
        min_bands: int = 1,
        max_bands: int = 3,
        min_time_frac: float = 0.6,
        max_time_frac: float = 1.0,
        secondary_band_prob: float = 0.4,
    ) -> None:
        self.spectrogram_shape = spectrogram_shape
        self.n_mels = n_mels
        self.T = T
        self.min_bands = max(1, min_bands)
        self.max_bands = max(self.min_bands, max_bands)
        self.min_time_frac = min_time_frac
        self.max_time_frac = max_time_frac
        self.secondary_band_prob = secondary_band_prob

    def _single_mask_numpy(self) -> np.ndarray:
        n_mels, T = self.n_mels, self.T
        M = np.zeros((n_mels, T), dtype=np.float32)

        n_bands = random.randint(self.min_bands, self.max_bands)
        for _ in range(n_bands):
            # Frequency band: lo in [10, 50], width in [10, 30] (clamped to n_mels)
            lo_min = max(0, int(n_mels * 0.08))
            lo_max = max(lo_min + 1, int(n_mels * 0.40))
            f_low = random.randint(lo_min, lo_max)

            w_min = max(1, int(n_mels * 0.08))
            w_max = max(w_min + 1, int(n_mels * 0.25))
            width = random.randint(w_min, w_max)
            f_high = min(n_mels, f_low + width)

            # Time extent: near-full with random start offset
            t_frac = random.uniform(self.min_time_frac, self.max_time_frac)
            t_len = max(1, int(T * t_frac))
            t_start = random.randint(0, max(0, T - t_len))
            t_end = min(T, t_start + t_len)

            M[f_low:f_high, t_start:t_end] = 1.0

        # Optional secondary faint band (mimics multi-stripe pattern seen in id_06)
        if random.random() < self.secondary_band_prob:
            f_low2 = random.randint(0, max(0, int(n_mels * 0.5)))
            w2 = random.randint(max(1, int(n_mels * 0.03)), max(2, int(n_mels * 0.10)))
            f_high2 = min(n_mels, f_low2 + w2)
            t_frac2 = random.uniform(0.8, 1.0)
            t_len2 = max(1, int(T * t_frac2))
            t_start2 = random.randint(0, max(0, T - t_len2))
            M[f_low2:f_high2, t_start2:t_start2 + t_len2] = 1.0

        return M

    def __call__(
        self,
        batch_size: int,
        device: torch.device | str,
    ) -> torch.Tensor:
        masks = [
            torch.from_numpy(self._single_mask_numpy()).unsqueeze(0).unsqueeze(0)
            for _ in range(batch_size)
        ]
        return torch.cat(masks, dim=0).to(device)