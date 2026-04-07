"""
Slider anomaly mask generator.

Observed pattern (DCASE2020 Task 2):
    - Broad mid-frequency bands, full duration.
    - id_00: broad positive lobe mel 60–128, negative lobe mel 0–30
    - id_02: mid band ~mel 30–60 with narrow negative stripe
    - id_04: wide upper-band anomaly ~mel 80–128 (strong positive)
    - id_06: broad full-duration changes spread across mel 0–128

    Slider anomalies span wider frequency ranges than pump/fan and often affect
    both low AND high frequency regions simultaneously (dual-band).

Strategy:
    1. Single wide band (prob 0.5): width 20–60 bins, random position, full duration.
    2. Dual band (prob 0.5): two separate regions (upper + lower), mimicking the
       positive-high / negative-low pattern.
    3. With prob 0.3, erode one edge of a band slightly to leave a sharp boundary.
"""

from __future__ import annotations

import random
import numpy as np
import torch


class SliderAnomalyStrategy:
    """
    Synthetic anomaly masks for the Slider machine type.

    Call signature: strategy(batch_size, device) -> (B, 1, n_mels, T) binary mask
    """

    def __init__(
        self,
        spectrogram_shape: tuple[int, int],
        n_mels: int,
        T: int,
        dual_band_prob: float = 0.50,
        sharp_edge_prob: float = 0.30,
        min_time_frac: float = 0.75,
        max_time_frac: float = 1.0,
    ) -> None:
        self.spectrogram_shape = spectrogram_shape
        self.n_mels = n_mels
        self.T = T
        self.dual_band_prob = dual_band_prob
        self.sharp_edge_prob = sharp_edge_prob
        self.min_time_frac = min_time_frac
        self.max_time_frac = max_time_frac

    def _time_interval(self) -> tuple[int, int]:
        T = self.T
        t_frac = random.uniform(self.min_time_frac, self.max_time_frac)
        t_len = max(1, int(T * t_frac))
        t_start = random.randint(0, max(0, T - t_len))
        return t_start, t_start + t_len

    def _single_mask_numpy(self) -> np.ndarray:
        n_mels, T = self.n_mels, self.T
        M = np.zeros((n_mels, T), dtype=np.float32)

        if random.random() < self.dual_band_prob:
            # Dual band: upper and lower frequency regions
            # Upper band in mel[55%, 100%]
            hi_lo = max(0, int(n_mels * 0.50))
            hi_hi = max(hi_lo + 1, int(n_mels * 0.80))
            f_hi_low = random.randint(hi_lo, hi_hi)
            w_hi = random.randint(max(1, int(n_mels * 0.15)), max(2, int(n_mels * 0.45)))
            f_hi_high = min(n_mels, f_hi_low + w_hi)
            ts, te = self._time_interval()
            M[f_hi_low:f_hi_high, ts:te] = 1.0

            # Lower band in mel[0, 35%]
            lo_hi = max(1, int(n_mels * 0.30))
            f_lo_low = random.randint(0, max(0, lo_hi - 1))
            w_lo = random.randint(max(1, int(n_mels * 0.05)), max(2, int(n_mels * 0.20)))
            f_lo_high = min(n_mels, f_lo_low + w_lo)
            ts2, te2 = self._time_interval()
            M[f_lo_low:f_lo_high, ts2:te2] = 1.0
        else:
            # Single wide band
            lo_min = 0
            lo_max = max(0, int(n_mels * 0.55))
            f_low = random.randint(lo_min, lo_max)
            w_min = max(2, int(n_mels * 0.16))
            w_max = max(w_min + 1, int(n_mels * 0.50))
            width = random.randint(w_min, w_max)
            f_high = min(n_mels, f_low + width)
            ts, te = self._time_interval()
            M[f_low:f_high, ts:te] = 1.0

        # Sharp edge: thin strip at one boundary (mimics the narrow dark stripe in id_02)
        if random.random() < self.sharp_edge_prob:
            # Add a narrow contiguous band just outside an existing band boundary
            nz_rows = np.where(M.any(axis=1))[0]
            if len(nz_rows) > 0:
                edge = int(nz_rows[0]) - random.randint(1, max(1, int(n_mels * 0.04)))
                if edge >= 0:
                    w_edge = random.randint(1, max(1, int(n_mels * 0.04)))
                    ts_e, te_e = self._time_interval()
                    M[max(0, edge): min(n_mels, edge + w_edge), ts_e:te_e] = 1.0

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