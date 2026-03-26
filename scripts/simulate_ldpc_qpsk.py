#!/usr/bin/env python3
"""
Simple LDPC + QPSK + AWGN simulator for index bitstreams (.bin) using pyldpc.

Scope:
- Focused on transmit_indices.py outputs (one .bin per clip).
- Block-wise LDPC coding (fixed (n, k) from pyldpc.make_ldpc).
- QPSK over AWGN channel, soft demod (per bit), LDPC decode.

Install:
  pip install numpy pyldpc
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
    raise ImportError(
        "This script requires pyldpc and numpy. Install with: pip install numpy pyldpc"
    ) from e


SQRT2_INV = 1.0 / math.sqrt(2.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simulate LDPC+QPSK+AWGN on .bin bitstreams.")
    p.add_argument("--input_dir", type=str, required=True, help="Directory containing input .bin files.")
    p.add_argument("--output_dir", type=str, required=True, help="Directory to write decoded .bin files.")
    p.add_argument("--ebn0_db", type=float, default=6.0, help="Eb/N0 in dB.")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--n", type=int, default=512, help="LDPC code length n.")
    p.add_argument("--d_v", type=int, default=2, help="LDPC variable-node degree.")
    p.add_argument("--d_c", type=int, default=4, help="LDPC check-node degree.")
    p.add_argument("--maxiter", type=int, default=100, help="LDPC BP decode max iterations.")
    p.add_argument("--csv", type=str, default=None, help="Optional CSV report path.")
    return p.parse_args()


def bytes_to_bits_lsb_first(payload: bytes) -> np.ndarray:
    out = np.zeros(len(payload) * 8, dtype=np.uint8)
    k = 0
    for b in payload:
        for i in range(8):
            out[k] = (b >> i) & 1
            k += 1
    return out


def bits_to_bytes_lsb_first(bits: np.ndarray) -> bytes:
    n_bits = int(bits.size)
    n_bytes = (n_bits + 7) // 8
    out = bytearray(n_bytes)
    for i in range(n_bits):
        if int(bits[i]) == 1:
            out[i // 8] |= 1 << (i % 8)
    return bytes(out)


def qpsk_mod_gray(bits: np.ndarray) -> tuple[np.ndarray, int]:
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
    code_rate: float,
    bits_per_symbol: int = 2,
) -> tuple[np.ndarray, float]:
    ebn0 = 10.0 ** (ebn0_db / 10.0)
    esn0 = ebn0 * bits_per_symbol * code_rate
    noise_var = 1.0 / (2.0 * esn0)
    noise_std = math.sqrt(noise_var)
    noise = rng.normal(0.0, noise_std, size=symbols.shape) + 1j * rng.normal(0.0, noise_std, size=symbols.shape)
    return symbols + noise.astype(np.complex64), noise_var


def qpsk_soft_llr(rx: np.ndarray, noise_var: float, pad_bits: int) -> np.ndarray:
    # For Gray QPSK with independent I/Q: llr(bit) = 2*component/noise_var
    llr_i = 2.0 * rx.real / noise_var
    llr_q = 2.0 * rx.imag / noise_var
    out = np.empty(rx.size * 2, dtype=np.float64)
    out[0::2] = llr_i
    out[1::2] = llr_q
    if pad_bits:
        out = out[:-pad_bits]
    return out


def ldpc_encode_block(msg_bits: np.ndarray, G: np.ndarray) -> np.ndarray:
    # c = G * m (mod 2)
    c = binaryproduct(G, msg_bits.astype(np.int64)) % 2
    return np.asarray(c, dtype=np.uint8).reshape(-1)


def llr_to_decode_observation(llr: np.ndarray, snr_db_for_decode: float) -> np.ndarray:
    """
    pyldpc.decode expects noisy BPSK observations y around {-1,+1}.
    We map LLR sign/magnitude to a bounded pseudo-observation in [-1,1].
    """
    # tanh(llr/2) is a standard mapping from LLR to soft bit estimate
    soft = np.tanh(llr / 2.0)
    # pyldpc uses x in {-1,+1}, where +1 maps to bit 0 and -1 to bit 1
    # llr > 0 => bit 0 => +1
    y = soft.astype(np.float64)
    # Slightly clip away exact boundaries for numerical stability
    y = np.clip(y, -0.999999, 0.999999)
    _ = snr_db_for_decode
    return y


def simulate_one_file(
    payload: bytes,
    H: np.ndarray,
    G: np.ndarray,
    k: int,
    n: int,
    ebn0_db: float,
    maxiter: int,
    rng: np.random.Generator,
) -> tuple[bytes, int, int]:
    src_bits = bytes_to_bits_lsb_first(payload)
    src_n = int(src_bits.size)
    n_blocks = (src_n + k - 1) // k
    src_pad = n_blocks * k - src_n
    if src_pad > 0:
        src_bits = np.concatenate([src_bits, np.zeros(src_pad, dtype=np.uint8)])

    decoded_blocks: list[np.ndarray] = []
    code_rate = float(k) / float(n)
    # pyldpc.decode expects an SNR value; we pass Es/N0-ish in dB.
    snr_decode_db = ebn0_db + 10.0 * math.log10(2.0 * code_rate)

    for b in range(n_blocks):
        m = src_bits[b * k : (b + 1) * k]
        c = ldpc_encode_block(m, G)  # coded bits, length n

        tx_sym, qpsk_pad = qpsk_mod_gray(c)
        rx_sym, noise_var = awgn_qpsk(tx_sym, ebn0_db=ebn0_db, rng=rng, code_rate=code_rate)
        llr = qpsk_soft_llr(rx_sym, noise_var=noise_var, pad_bits=qpsk_pad)
        y = llr_to_decode_observation(llr, snr_decode_db)

        dec_codeword = pyldpc.decode(H, y, snr_decode_db, maxiter=maxiter)
        dec_msg = pyldpc.get_message(G, dec_codeword).astype(np.uint8).reshape(-1)
        decoded_blocks.append(dec_msg)

    rx_bits = np.concatenate(decoded_blocks, axis=0)[:src_n]
    bit_errors = int(np.count_nonzero(rx_bits != bytes_to_bits_lsb_first(payload)))
    rx_payload = bits_to_bytes_lsb_first(rx_bits)
    return rx_payload, src_n, bit_errors


def main() -> None:
    args = parse_args()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in in_dir.iterdir() if p.is_file() and p.suffix == ".bin")
    if not files:
        raise FileNotFoundError(f"No .bin files found under {in_dir}")

    rng = np.random.default_rng(args.seed)

    H, G = pyldpc.make_ldpc(args.n, args.d_v, args.d_c, systematic=True, sparse=True, seed=args.seed)
    n = int(G.shape[0])
    k = int(G.shape[1])
    rate = k / n
    print(f"LDPC config: n={n}, k={k}, rate={rate:.4f}, d_v={args.d_v}, d_c={args.d_c}")

    rows: list[tuple[str, int, int, float, int]] = []
    total_bits = 0
    total_err = 0
    n_exact = 0

    for p in files:
        tx_payload = p.read_bytes()
        rx_payload, n_bits, n_err = simulate_one_file(
            tx_payload,
            H=H,
            G=G,
            k=k,
            n=n,
            ebn0_db=args.ebn0_db,
            maxiter=args.maxiter,
            rng=rng,
        )
        exact = int(tx_payload == rx_payload)
        if exact:
            n_exact += 1
        total_bits += n_bits
        total_err += n_err
        ber = (n_err / n_bits) if n_bits > 0 else 0.0
        rows.append((p.name, len(tx_payload), n_bits, ber, exact))
        (out_dir / p.name).write_bytes(rx_payload)

    total_ber = (total_err / total_bits) if total_bits > 0 else 0.0
    print(f"Processed {len(files)} files at Eb/N0={args.ebn0_db:.2f} dB")
    print(f"Total BER after LDPC decode: {total_ber:.6e} ({total_err}/{total_bits})")
    print(f"Exact byte match files: {n_exact}/{len(files)}")
    print(f"Decoded files written to: {out_dir}")

    csv_path = Path(args.csv) if args.csv else out_dir / "ldpc_qpsk_report.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "payload_bytes", "n_bits", "ber_after_ldpc", "exact_byte_match"])
        for r in rows:
            w.writerow([r[0], r[1], r[2], f"{r[3]:.8e}", r[4]])
    print(f"Report written to: {csv_path}")


if __name__ == "__main__":
    main()
