"""
Valve anomaly mask generator.

Observed pattern (DCASE2020 Task 2):
    - Broad, diffuse anomaly patterns spanning mid-to-high frequencies (~mel 30–120).
    - id_00: scattered across mel 40–120, mostly time-uniform with light noise texture.
    - id_02: similarly broad, relatively uniform in time, mel 0–100.
    - id_04: mid-high with emphasis around mel 40–80.
    - id_06: especially broad, nearly full-frequency (mel 0–128), with mixed pos/neg.

    Valve anomalies are the most spatially diffuse — they look like broadband
    amplitude modulation rather than isolated narrow bands. The key feature is
    their WIDE frequency coverage and relatively UNIFORM temporal profile.

Strategy:
    1. Wide band (prob 0.65): single band spanning 35–70% of mel range,
       predominantly in mid-upper frequencies, full or near-full duration.
    2. Full-spectrum diffuse (prob 0.35): 3–6 overlapping bands tiling mel[0, 128],
       each independently spanning >85% of duration, simulating the scattered
       texture of id_06.
    3. With prob 0.4, add low-amplitude Perlin-style banding by subdividing the
       mask into sub-regions with random on/off, creating a mottled appearance
       within the active region. This is implemented as column-wise dropout:
       every ~5 time frames a short gap is introduced.
"""

from __future__ import annotations

import random
import numpy as np
import torch


class ValveAnomalyStrategy:
    """
    Synthetic anomaly masks for the Valve machine type.

    Call signature: strategy(batch_size, device) -> (B, 1, n_mels, T) binary mask
    """

    def __init__(
        self,
        spectrogram_shape: tuple[int, int],
        n_mels: int,
        T: int,
        full_spectrum_prob: float = 0.35,
        mottled_prob: float = 0.40,
        dropout_density: float = 0.08,  # fraction of time columns to drop
    ) -> None:
        self.spectrogram_shape = spectrogram_shape
        self.n_mels = n_mels
        self.T = T
        self.full_spectrum_prob = full_spectrum_prob
        self.mottled_prob = mottled_prob
        self.dropout_density = dropout_density

    def _apply_column_dropout(self, M: np.ndarray) -> np.ndarray:
        """
        Randomly zero out short temporal gaps within the mask to create a
        slightly mottled texture (mimics valve's mixed positive/negative pattern).
        """
        T = self.T
        n_drops = max(1, int(T * self.dropout_density))
        for _ in range(n_drops):
            t0 = random.randint(0, max(0, T - 1))
            gap_len = random.randint(1, max(1, int(T * 0.02)))
            t1 = min(T, t0 + gap_len)
            M[:, t0:t1] = 0.0
        return M

    def _single_mask_numpy(self) -> np.ndarray:
        n_mels, T = self.n_mels, self.T
        M = np.zeros((n_mels, T), dtype=np.float32)

        if random.random() < self.full_spectrum_prob:
            # Full-spectrum diffuse: tile with 3–6 overlapping bands
            n_tiles = random.randint(3, 6)
            for _ in range(n_tiles):
                f_low = random.randint(0, max(0, int(n_mels * 0.60)))
                w = random.randint(max(1, int(n_mels * 0.12)), max(2, int(n_mels * 0.50)))
                f_high = min(n_mels, f_low + w)
                t_frac = random.uniform(0.85, 1.0)
                t_len = max(1, int(T * t_frac))
                t_start = random.randint(0, max(0, T - t_len))
                M[f_low:f_high, t_start:t_start + t_len] = 1.0
        else:
            # Wide single band in mid-to-high mel range
            lo_min = max(0, int(n_mels * 0.20))
            lo_max = max(lo_min + 1, int(n_mels * 0.50))
            f_low = random.randint(lo_min, lo_max)
            w_min = max(2, int(n_mels * 0.28))
            w_max = max(w_min + 1, int(n_mels * 0.65))
            width = random.randint(w_min, w_max)
            f_high = min(n_mels, f_low + width)

            t_frac = random.uniform(0.80, 1.0)
            t_len = max(1, int(T * t_frac))
            t_start = random.randint(0, max(0, T - t_len))
            M[f_low:f_high, t_start:t_start + t_len] = 1.0

        # Mottled texture: random small temporal gaps within active region
        if random.random() < self.mottled_prob:
            M = self._apply_column_dropout(M)

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