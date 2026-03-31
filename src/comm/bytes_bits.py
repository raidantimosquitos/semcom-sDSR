"""
Byte/bit conversion helpers (LSB-first per byte) matching src/utils/bitstream.py.
"""

from __future__ import annotations

import numpy as np


def bytes_to_bits_lsb_first(payload: bytes) -> np.ndarray:
    out = np.zeros(len(payload) * 8, dtype=np.uint8)
    k = 0
    for b in payload:
        for i in range(8):
            out[k] = (b >> i) & 1
            k += 1
    return out


def bits_to_bytes_lsb_first(bits: np.ndarray) -> bytes:
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    n_bits = int(bits.size)
    n_bytes = (n_bits + 7) // 8
    out = bytearray(n_bytes)
    for i in range(n_bits):
        if int(bits[i]) == 1:
            out[i // 8] |= 1 << (i % 8)
    return bytes(out)

