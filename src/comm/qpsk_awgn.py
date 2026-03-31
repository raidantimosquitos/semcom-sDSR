"""
QPSK (Gray) modulation + AWGN utilities.

Conventions:
- Bits are 0/1.
- QPSK Gray mapping (independent signs on I and Q):
    b0 -> I sign, b1 -> Q sign
    0 -> +1, 1 -> -1
  and normalized by 1/sqrt(2) so Es=1.
- Eb/N0 to Es/N0 uses: Es/N0 = Eb/N0 * bits_per_symbol * code_rate
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

SQRT2_INV = 1.0 / math.sqrt(2.0)


def qpsk_mod_gray(bits: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Args:
        bits: (N,) uint8 {0,1}
    Returns:
        symbols: (N/2,) complex64
        pad_bits: number of pad bits appended (0 or 1)
    """
    if bits.ndim != 1:
        bits = bits.reshape(-1)
    if bits.size % 2 != 0:
        bits = np.concatenate([bits, np.zeros(1, dtype=np.uint8)])
        pad = 1
    else:
        pad = 0
    b0 = bits[0::2]
    b1 = bits[1::2]
    i = np.where(b0 == 0, 1.0, -1.0)
    q = np.where(b1 == 0, 1.0, -1.0)
    s = (i + 1j * q) * SQRT2_INV
    return s.astype(np.complex64), pad


def awgn_qpsk(
    symbols: np.ndarray,
    ebn0_db: float,
    rng: np.random.Generator,
    code_rate: float = 1.0,
    bits_per_symbol: int = 2,
) -> tuple[np.ndarray, float]:
    """
    Returns:
        rx: noisy symbols
        noise_var: per-dimension noise variance (sigma^2)
    """
    ebn0 = 10.0 ** (ebn0_db / 10.0)
    esn0 = ebn0 * bits_per_symbol * float(code_rate)
    noise_var = 1.0 / (2.0 * esn0)
    noise_std = math.sqrt(noise_var)
    noise = rng.normal(0.0, noise_std, size=symbols.shape) + 1j * rng.normal(
        0.0, noise_std, size=symbols.shape
    )
    return symbols + noise.astype(np.complex64), float(noise_var)


def qpsk_soft_llr(rx: np.ndarray, noise_var: float, pad_bits: int = 0) -> np.ndarray:
    """
    For Gray QPSK with independent I/Q:
      LLR(bit) = 2*component / noise_var
    Returns (N_bits,) float64
    """
    llr_i = 2.0 * rx.real / float(noise_var)
    llr_q = 2.0 * rx.imag / float(noise_var)
    out = np.empty(rx.size * 2, dtype=np.float64)
    out[0::2] = llr_i
    out[1::2] = llr_q
    if pad_bits:
        out = out[:-pad_bits]
    return out


def qpsk_hard_demod(rx: np.ndarray, pad_bits: int = 0) -> np.ndarray:
    """Hard decision demod. Returns uint8 bits (N_bits,)."""
    out = np.empty(rx.size * 2, dtype=np.uint8)
    out[0::2] = (rx.real < 0.0).astype(np.uint8)
    out[1::2] = (rx.imag < 0.0).astype(np.uint8)
    if pad_bits:
        out = out[:-pad_bits]
    return out

