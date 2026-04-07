"""
ToyConveyor anomaly mask generator.

Observed pattern (DCASE2020 Task 2):
    - The MOST DISTINCTIVE machine type: very narrow horizontal bands (1–8 mel bins)
      spanning only PARTIAL time ranges (NOT full duration).
    - id_01: 2–3 well-separated thin stripes, each spanning ~60–80% of duration,
      staggered slightly in time.
    - id_02: narrow pair of stripes ~mel 35–45, partial duration with a gap.
    - id_03: two very narrow stripes ~mel 35 and ~mel 75, partial time coverage.

    Key distinctions vs other machines:
      * Very narrow bands (2–8 bins, NOT 20–60)
      * Partial temporal coverage with MULTIPLE distinct stripes (not one blob)
      * Stripes are well-separated in frequency
      * Time coverage is often the second or middle portion, NOT necessarily full

Strategy:
    1. Sample 2–5 thin bands (1–8 mel bins each), spread across mel[20, 110].
       Bands must be separated by at least 5 bins from each other.
    2. Each band gets its own independent time interval (40–85% of T),
       creating the staggered appearance.
    3. With prob 0.35 a band is "doubled" — a nearly-adjacent twin (gap 2–5 bins).
"""

from __future__ import annotations

import random
import numpy as np
import torch


class ToyConveyorAnomalyStrategy:
    """
    Synthetic anomaly masks for the ToyConveyor machine type.

    Call signature: strategy(batch_size, device) -> (B, 1, n_mels, T) binary mask
    """

    def __init__(
        self,
        spectrogram_shape: tuple[int, int],
        n_mels: int,
        T: int,
        min_stripes: int = 2,
        max_stripes: int = 4,
        double_stripe_prob: float = 0.35,
        min_stripe_sep: int = 5,
        min_time_frac: float = 0.40,
        max_time_frac: float = 0.85,
    ) -> None:
        self.spectrogram_shape = spectrogram_shape
        self.n_mels = n_mels
        self.T = T
        self.min_stripes = max(1, min_stripes)
        self.max_stripes = max(self.min_stripes, max_stripes)
        self.double_stripe_prob = double_stripe_prob
        self.min_stripe_sep = max(1, min_stripe_sep)
        self.min_time_frac = min_time_frac
        self.max_time_frac = max_time_frac

    def _place_stripes(self, n_stripes: int) -> list[tuple[int, int]]:
        """Sample n_stripes non-adjacent (separated by min_stripe_sep) band centers."""
        n_mels = self.n_mels
        lo_bound = max(0, int(n_mels * 0.15))
        hi_bound = min(n_mels - 1, int(n_mels * 0.90))
        w_min = max(1, int(n_mels * 0.01))
        w_max = max(w_min + 1, int(n_mels * 0.07))  # 1–8 bins for 128 mels

        stripes: list[tuple[int, int]] = []  # (f_low, f_high)
        occupied: list[tuple[int, int]] = []

        attempts = 0
        while len(stripes) < n_stripes and attempts < 500:
            attempts += 1
            f_low = random.randint(lo_bound, max(lo_bound, hi_bound - w_max))
            width = random.randint(w_min, w_max)
            f_high = min(n_mels, f_low + width)

            # Check separation from existing stripes
            too_close = any(
                not (f_high + self.min_stripe_sep <= s or f_low >= e + self.min_stripe_sep)
                for s, e in occupied
            )
            if too_close:
                continue
            stripes.append((f_low, f_high))
            occupied.append((f_low, f_high))

        return stripes

    def _time_interval(self) -> tuple[int, int]:
        T = self.T
        t_frac = random.uniform(self.min_time_frac, self.max_time_frac)
        t_len = max(1, int(T * t_frac))
        t_start = random.randint(0, max(0, T - t_len))
        return t_start, t_start + t_len

    def _single_mask_numpy(self) -> np.ndarray:
        n_mels, T = self.n_mels, self.T
        M = np.zeros((n_mels, T), dtype=np.float32)

        n_stripes = random.randint(self.min_stripes, self.max_stripes)
        stripes = self._place_stripes(n_stripes)

        for f_low, f_high in stripes:
            ts, te = self._time_interval()
            M[f_low:f_high, ts:te] = 1.0

            # Optional twin stripe immediately adjacent (doubled appearance)
            if random.random() < self.double_stripe_prob:
                gap = random.randint(2, max(3, int(n_mels * 0.04)))
                f2_low = f_high + gap
                w2 = random.randint(max(1, int(n_mels * 0.01)), max(2, int(n_mels * 0.05)))
                f2_high = min(n_mels, f2_low + w2)
                if f2_high > f2_low and f2_high <= n_mels:
                    # Slightly shifted time window for the twin
                    shift = random.randint(-int(T * 0.05), int(T * 0.05))
                    ts2 = max(0, ts + shift)
                    te2 = min(T, te + shift)
                    if te2 > ts2:
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