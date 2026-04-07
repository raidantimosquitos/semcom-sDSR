"""
Pump anomaly mask generator.

Observed pattern (DCASE2020 Task 2):
    - Narrow full-duration horizontal bands (strong, localized).
    - id_00: broad lower frequency elevation, mel 0–40
    - id_02: uniform spread across mel 0–60 with mild banding
    - id_04: extremely strong single narrow band ~mel 60–80 (dark = lower energy in anom)
    - id_06: diffuse lower frequency shift mel 0–30

Strategy:
    1. Dominant mode (prob 0.6): single narrow band (5–18 bins wide) placed
       randomly in mel[5, 90], spanning 80–100% of duration.
    2. Diffuse mode (prob 0.4): wider band (20–50 bins) at low-mid frequencies
       (mel 0–60), full duration, mimicking id_00/id_06.
    3. With prob 0.25, add a second thin band (3–8 bins) nearby the first to
       mimic the doubled-stripe pattern occasionally visible.
"""

from __future__ import annotations

import random
import numpy as np
import torch


class PumpAnomalyStrategy:
    """
    Synthetic anomaly masks for the Pump machine type.

    Call signature: strategy(batch_size, device) -> (B, 1, n_mels, T) binary mask
    """

    def __init__(
        self,
        spectrogram_shape: tuple[int, int],
        n_mels: int,
        T: int,
        narrow_band_prob: float = 0.60,
        double_band_prob: float = 0.25,
    ) -> None:
        self.spectrogram_shape = spectrogram_shape
        self.n_mels = n_mels
        self.T = T
        self.narrow_band_prob = narrow_band_prob
        self.double_band_prob = double_band_prob

    def _single_mask_numpy(self) -> np.ndarray:
        n_mels, T = self.n_mels, self.T
        M = np.zeros((n_mels, T), dtype=np.float32)

        if random.random() < self.narrow_band_prob:
            # Narrow isolated band anywhere in mel[5%, 92%]
            lo_min = max(0, int(n_mels * 0.04))
            lo_max = max(lo_min + 1, int(n_mels * 0.88))
            f_low = random.randint(lo_min, lo_max)
            w_min = max(1, int(n_mels * 0.04))
            w_max = max(w_min + 1, int(n_mels * 0.15))
            width = random.randint(w_min, w_max)
            f_high = min(n_mels, f_low + width)

            t_frac = random.uniform(0.80, 1.0)
            t_len = max(1, int(T * t_frac))
            t_start = random.randint(0, max(0, T - t_len))
            M[f_low:f_high, t_start:t_start + t_len] = 1.0

            # Optional second thin band nearby
            if random.random() < self.double_band_prob:
                gap = random.randint(2, max(3, int(n_mels * 0.06)))
                f_low2 = max(0, f_low + width + gap)
                w2 = random.randint(max(1, int(n_mels * 0.02)), max(2, int(n_mels * 0.07)))
                f_high2 = min(n_mels, f_low2 + w2)
                if f_high2 > f_low2:
                    M[f_low2:f_high2, t_start:t_start + t_len] = 1.0
        else:
            # Diffuse wider band at low-mid frequencies
            lo_max = max(1, int(n_mels * 0.45))
            f_low = random.randint(0, lo_max)
            w_min = max(2, int(n_mels * 0.15))
            w_max = max(w_min + 1, int(n_mels * 0.45))
            width = random.randint(w_min, w_max)
            f_high = min(n_mels, f_low + width)
            # Full duration for diffuse mode
            M[f_low:f_high, :] = 1.0

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