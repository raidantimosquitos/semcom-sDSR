"""
ToyCar anomaly mask generator.

Observed pattern (DCASE2020 Task 2):
    - Mid-frequency range (~mel 40–80), modest magnitude differences.
    - Temporally structured: anomaly differences are NOT uniform over time;
      they show stronger deviations in later time frames (id_04 shows a
      large dark block ~t=100–320 in mel 50–80).
    - id_01/id_02: relatively uniform across time but concentrated mid-freq.
    - id_04: strong localized drop in mel 50–80 for t > 100.

    Unlike fan/pump/slider, ToyCar anomalies do NOT span the full duration
    uniformly — there is a structured temporal component.

Strategy:
    1. Pick a mid-frequency band (mel 30–90).
    2. With prob 0.5: full-duration coverage (uniform case, id_01/id_02-like).
    3. With prob 0.5: partial temporal coverage — one or two contiguous
       time segments covering 40–85% of duration, possibly right-aligned
       (id_04-like, where deviation grows in the second half).
    4. With prob 0.3: add a second, narrower sub-band below the main band
       to mimic the multi-stripe residual.
"""

from __future__ import annotations

import random
import numpy as np
import torch


class ToyCarAnomalyStrategy:
    """
    Synthetic anomaly masks for the ToyCar machine type.

    Call signature: strategy(batch_size, device) -> (B, 1, n_mels, T) binary mask
    """

    def __init__(
        self,
        spectrogram_shape: tuple[int, int],
        n_mels: int,
        T: int,
        temporal_structure_prob: float = 0.50,
        secondary_band_prob: float = 0.30,
    ) -> None:
        self.spectrogram_shape = spectrogram_shape
        self.n_mels = n_mels
        self.T = T
        self.temporal_structure_prob = temporal_structure_prob
        self.secondary_band_prob = secondary_band_prob

    def _single_mask_numpy(self) -> np.ndarray:
        n_mels, T = self.n_mels, self.T
        M = np.zeros((n_mels, T), dtype=np.float32)

        # Mid-frequency band: mel [25%, 70%] with width [10%, 30%]
        lo_min = max(0, int(n_mels * 0.22))
        lo_max = max(lo_min + 1, int(n_mels * 0.60))
        f_low = random.randint(lo_min, lo_max)
        w_min = max(1, int(n_mels * 0.10))
        w_max = max(w_min + 1, int(n_mels * 0.32))
        width = random.randint(w_min, w_max)
        f_high = min(n_mels, f_low + width)

        if random.random() < self.temporal_structure_prob:
            # Partial temporal: one large contiguous segment, possibly right-aligned
            t_frac = random.uniform(0.40, 0.85)
            t_len = max(1, int(T * t_frac))
            # Bias toward right half (later frames) with probability 0.6
            if random.random() < 0.60:
                t_start = max(0, T - t_len - random.randint(0, max(0, int(T * 0.15))))
            else:
                t_start = random.randint(0, max(0, T - t_len))
            t_end = min(T, t_start + t_len)
            M[f_low:f_high, t_start:t_end] = 1.0
        else:
            # Full duration, uniform
            M[f_low:f_high, :] = 1.0

        # Optional secondary narrower band below main band
        if random.random() < self.secondary_band_prob:
            gap = random.randint(2, max(3, int(n_mels * 0.05)))
            f2_low = max(0, f_low - gap - random.randint(2, max(3, int(n_mels * 0.08))))
            w2 = random.randint(max(1, int(n_mels * 0.04)), max(2, int(n_mels * 0.12)))
            f2_high = min(n_mels, f2_low + w2)
            if f2_high > f2_low:
                # Same temporal extent as main band
                t_nz = np.where(M.any(axis=0))[0]
                if len(t_nz) > 0:
                    ts2, te2 = int(t_nz[0]), int(t_nz[-1]) + 1
                else:
                    ts2, te2 = 0, T
                M[f2_low:f2_high, ts2:te2] = 1.0

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