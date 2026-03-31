"""
LDPC (pyldpc) block coding helpers.

This wraps the same approach used in scripts/simulate_ldpc_qpsk.py and the
notebooks/comm_stack/ldpc_tests.ipynb smoke tests.

Dependency: numpy, pyldpc.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:
    import pyldpc
    from pyldpc.utils import binaryproduct
except Exception as e:  # pragma: no cover
    pyldpc = None  # type: ignore[assignment]
    binaryproduct = None  # type: ignore[assignment]
    _pyldpc_import_error = e

from .qpsk_awgn import awgn_qpsk, qpsk_mod_gray, qpsk_soft_llr


@dataclass(frozen=True)
class LDPCConfig:
    n: int = 512
    d_v: int = 2
    d_c: int = 4
    maxiter: int = 100
    seed: int = 0


@dataclass(frozen=True)
class LDPCCode:
    H: np.ndarray
    G: np.ndarray
    n: int
    k: int

    @property
    def rate(self) -> float:
        return float(self.k) / float(self.n)


def make_ldpc_code(cfg: LDPCConfig) -> LDPCCode:
    if pyldpc is None:
        raise ImportError(
            "pyldpc is required for LDPC experiments. Install with: pip install numpy pyldpc"
        ) from _pyldpc_import_error
    if cfg.n % cfg.d_c != 0:
        raise ValueError(
            f"Invalid LDPCConfig for pyldpc regular LDPC: n={cfg.n} must be divisible by d_c={cfg.d_c}. "
            f"Fix by choosing --ldpc_n as a multiple of {cfg.d_c}, or adjust the UEP mapping."
        )
    H, G = pyldpc.make_ldpc(
        cfg.n, cfg.d_v, cfg.d_c, systematic=True, sparse=True, seed=cfg.seed
    )
    n = int(G.shape[0])
    k = int(G.shape[1])
    return LDPCCode(H=np.asarray(H), G=np.asarray(G), n=n, k=k)


def _ldpc_encode_block(msg_bits: np.ndarray, G: np.ndarray) -> np.ndarray:
    # c = G * m (mod 2)
    c = binaryproduct(G, msg_bits.astype(np.int64)) % 2
    return np.asarray(c, dtype=np.uint8).reshape(-1)


def _llr_to_pyldpc_obs(llr: np.ndarray) -> np.ndarray:
    # Map LLR -> soft BPSK observation in [-1,1]
    y = np.tanh(llr / 2.0).astype(np.float64)
    return np.clip(y, -0.999999, 0.999999)


def ldpc_qpsk_awgn_roundtrip(
    src_bits: np.ndarray,
    *,
    code: LDPCCode,
    ldpc_maxiter: int,
    ebn0_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Block-wise LDPC encode -> QPSK -> AWGN -> soft demod -> pyldpc BP decode.

    Args:
        src_bits: (N,) uint8 {0,1}
    Returns:
        rx_bits: (N,) uint8 {0,1}
    """
    if pyldpc is None:
        raise ImportError(
            "pyldpc is required for LDPC experiments. Install with: pip install numpy pyldpc"
        ) from _pyldpc_import_error

    if src_bits.ndim != 1:
        src_bits = src_bits.reshape(-1)
    src_n = int(src_bits.size)
    k = int(code.k)
    n = int(code.n)
    n_blocks = (src_n + k - 1) // k
    pad = n_blocks * k - src_n
    if pad:
        src_bits = np.concatenate([src_bits, np.zeros(pad, dtype=np.uint8)])

    decoded_blocks: list[np.ndarray] = []
    code_rate = code.rate
    snr_decode_db = ebn0_db + 10.0 * math.log10(2.0 * code_rate)

    for b in range(n_blocks):
        m = src_bits[b * k : (b + 1) * k]
        c = _ldpc_encode_block(m, code.G)  # (n,)

        tx_sym, qpsk_pad = qpsk_mod_gray(c)
        rx_sym, noise_var = awgn_qpsk(
            tx_sym, ebn0_db=ebn0_db, rng=rng, code_rate=code_rate
        )
        llr = qpsk_soft_llr(rx_sym, noise_var=noise_var, pad_bits=qpsk_pad)
        y = _llr_to_pyldpc_obs(llr)

        dec_codeword = pyldpc.decode(code.H, y, snr_decode_db, maxiter=ldpc_maxiter)
        dec_msg = (
            pyldpc.get_message(code.G, dec_codeword).astype(np.uint8).reshape(-1)
        )
        decoded_blocks.append(dec_msg)

    rx_bits = np.concatenate(decoded_blocks, axis=0)[:src_n]
    return rx_bits

