"""
BER-based bit-flip injector for byte payloads with optional header protection.

This is used to approximate a digital link by applying post-FEC residual BER(SNR)
directly to recovered payload bits, avoiding expensive FEC decoding for every clip.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class BERCycle:
    snr_db: np.ndarray  # (M,)
    ber: np.ndarray     # (M,)

    def ber_at(self, snr_db: float) -> float:
        x = float(snr_db)
        # clamp outside range
        if x <= float(self.snr_db[0]):
            return float(self.ber[0])
        if x >= float(self.snr_db[-1]):
            return float(self.ber[-1])
        # linear interpolation in log10(BER) space is usually smoother
        eps = 1e-15
        y = np.log10(np.clip(self.ber, eps, 1.0))
        yi = float(np.interp(x, self.snr_db, y))
        return float(10.0**yi)


def load_ber_curve_csv(path: str | Path) -> BERCycle:
    p = Path(path)
    xs: list[float] = []
    ys: list[float] = []
    with open(p, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            xs.append(float(row["snr_db"]))
            ys.append(float(row["ber_postfec"]))
    if not xs:
        raise ValueError(f"No rows in BER curve CSV: {p}")
    order = np.argsort(np.asarray(xs))
    snr = np.asarray(xs, dtype=np.float64)[order]
    ber = np.asarray(ys, dtype=np.float64)[order]
    return BERCycle(snr_db=snr, ber=ber)


def bitflip_bytes(
    blob: bytes,
    *,
    ber: float,
    protect_bytes: int = 0,
    seed: int | None = None,
) -> bytes:
    """
    Flip bits in blob i.i.d. with probability ber, excluding the first protect_bytes.
    Bit order: LSB-first within each byte.
    """
    p = float(ber)
    if p <= 0.0:
        return blob
    if p >= 1.0:
        p = 1.0
    n_protect = max(0, int(protect_bytes))
    if n_protect >= len(blob):
        return blob

    rng = np.random.default_rng(seed)
    head = blob[:n_protect]
    tail = np.frombuffer(blob[n_protect:], dtype=np.uint8).copy()

    bits = np.unpackbits(tail, bitorder="little")
    flips = rng.random(bits.shape[0]) < p
    bits ^= flips.astype(np.uint8)
    out_tail = np.packbits(bits, bitorder="little").astype(np.uint8).tobytes()
    return head + out_tail

