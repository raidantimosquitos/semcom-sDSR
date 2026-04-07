#!/usr/bin/env python3
"""
Calibrate post-FEC residual BER vs SNR for a fixed LDPC(approx rate=1/2) + BPSK + AWGN link.

This is intended to provide a fast abstraction for codec baselines (OPUS/JPEG):
- measure BER_postfec(SNR) once
- later, corrupt codec bytes by i.i.d. bit flips with probability BER_postfec(SNR)

Output CSV columns:
  snr_db, ber_postfec, n_bits, n_err
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

try:
    import pyldpc
    from pyldpc.utils import binaryproduct
except Exception as e:  # pragma: no cover
    raise ImportError("Requires pyldpc + numpy. Install with: pip install numpy pyldpc") from e


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--snr_db", type=float, nargs="+", default=[0, 2, 4, 6, 8, 10, 12, 15, 20])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=512, help="LDPC code length n (must be divisible by d_c).")
    p.add_argument("--d_v", type=int, default=2)
    p.add_argument("--d_c", type=int, default=4, help="For rate~1/2 use d_v=2,d_c=4.")
    p.add_argument("--maxiter", type=int, default=100)
    p.add_argument("--target_err", type=int, default=500, help="Stop per SNR after reaching this many bit errors (stability).")
    p.add_argument("--max_bits", type=int, default=2_000_000, help="Cap total decoded info bits per SNR.")
    p.add_argument("--out", type=str, required=True)
    return p.parse_args()


def bpsk_awgn_llr(x_bpsk: np.ndarray, snr_db: float) -> np.ndarray:
    """
    BPSK mapping: bit 0 -> +1, bit 1 -> -1
    AWGN: y = x + n, n~N(0, sigma^2)
    LLR for BPSK over AWGN: llr = 2*y/sigma^2

    Here snr_db is interpreted as Es/N0 (since BPSK has 1 bit/symbol and Es=Eb).
    sigma^2 = N0/2 = 1/(2*SNR) if Es is normalized to 1.
    """
    snr = 10.0 ** (snr_db / 10.0)
    noise_var = 1.0 / (2.0 * snr)
    y = x_bpsk + np.random.normal(0.0, math.sqrt(noise_var), size=x_bpsk.shape)
    llr = 2.0 * y / noise_var
    return llr.astype(np.float64)


def llr_to_pyldpc_obs(llr: np.ndarray) -> np.ndarray:
    # Map LLR -> soft BPSK observation in [-1,1] (same approach as ldpc_pyldpc.py)
    y = np.tanh(llr / 2.0).astype(np.float64)
    return np.clip(y, -0.999999, 0.999999)


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)

    if args.n % args.d_c != 0:
        raise ValueError(f"pyldpc regular LDPC requires n % d_c == 0 (got n={args.n}, d_c={args.d_c})")

    H, G = pyldpc.make_ldpc(args.n, args.d_v, args.d_c, systematic=True, sparse=True, seed=args.seed)
    n = int(G.shape[0])
    k = int(G.shape[1])
    rate = k / n
    print(f"LDPC: n={n} k={k} rate={rate:.4f} d_v={args.d_v} d_c={args.d_c} maxiter={args.maxiter}")

    rows: list[tuple[float, float, int, int]] = []
    for snr_db in args.snr_db:
        n_bits = 0
        n_err = 0

        # pyldpc.decode expects an SNR parameter used as var = 10^(-snr/10) in decoder.py.
        # We pass an equivalent SNR in dB consistent with our observation scaling.
        snr_for_decode = float(snr_db)

        while n_bits < args.max_bits and n_err < args.target_err:
            u = rng.integers(0, 2, size=k, dtype=np.int64)
            c = (binaryproduct(G, u) % 2).astype(np.int64).reshape(-1)

            # BPSK: 0->+1, 1->-1
            x = (1.0 - 2.0 * c).astype(np.float64)
            llr = bpsk_awgn_llr(x, snr_db=float(snr_db))
            y = llr_to_pyldpc_obs(llr)

            dec_codeword = pyldpc.decode(H, y, snr_for_decode, maxiter=int(args.maxiter))
            u_hat = pyldpc.get_message(G, dec_codeword).astype(np.int64).reshape(-1)

            n_bits += int(k)
            n_err += int(np.count_nonzero(u_hat != u))

        ber = (n_err / n_bits) if n_bits else 0.0
        print(f"SNR={snr_db:>6.2f} dB | BER_post={ber:.6e} ({n_err}/{n_bits})")
        rows.append((float(snr_db), float(ber), int(n_bits), int(n_err)))

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["snr_db", "ber_postfec", "n_bits", "n_err"])
        for r in rows:
            w.writerow([f"{r[0]:.6f}", f"{r[1]:.12e}", r[2], r[3]])
    print(f"Saved BER curve CSV to {out_path}")


if __name__ == "__main__":
    main()

