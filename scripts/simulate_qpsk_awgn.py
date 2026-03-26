#!/usr/bin/env python3
"""
Simple uncoded QPSK + AWGN bitstream simulator.

Goal:
- First-step PHY sanity check before adding LDPC.
- Read input .bin files, transmit all bits through QPSK+AWGN, hard-demodulate,
  write decoded .bin files, and report BER + exact byte match.

This uses the same LSB-first bit order as src/utils/bitstream.py.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import random


SQRT2_INV = 1.0 / math.sqrt(2.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simulate uncoded QPSK+AWGN for .bin bitstreams.")
    p.add_argument("--input_dir", type=str, required=True, help="Directory containing input .bin files.")
    p.add_argument("--output_dir", type=str, required=True, help="Directory to write decoded .bin files.")
    p.add_argument("--ebn0_db", type=float, default=30.0, help="Eb/N0 in dB (default: 30).")
    p.add_argument("--seed", type=int, default=0, help="Random seed for AWGN.")
    p.add_argument("--csv", type=str, default=None, help="Optional CSV report path.")
    return p.parse_args()


def bytes_to_bits_lsb_first(payload: bytes) -> list[int]:
    out: list[int] = []
    for b in payload:
        for i in range(8):
            out.append((b >> i) & 1)
    return out


def bits_to_bytes_lsb_first(bits: list[int]) -> bytes:
    n_bits = len(bits)
    n_bytes = (n_bits + 7) // 8
    out = bytearray(n_bytes)
    for i in range(n_bits):
        if bits[i] == 1:
            out[i // 8] |= 1 << (i % 8)
    return bytes(out)


def qpsk_mod_gray(bits: list[int]) -> tuple[list[complex], int]:
    """
    Gray mapping:
      00 -> (+1,+1)/sqrt(2)
      01 -> (-1,+1)/sqrt(2)
      11 -> (-1,-1)/sqrt(2)
      10 -> (+1,-1)/sqrt(2)
    Returns complex symbols and pad bit count.
    """
    if len(bits) % 2 != 0:
        bits = bits + [0]
        pad = 1
    else:
        pad = 0

    syms: list[complex] = []
    for i in range(0, len(bits), 2):
        b0 = bits[i]
        b1 = bits[i + 1]
        i_level = 1.0 if b0 == 0 else -1.0
        q_level = 1.0 if b1 == 0 else -1.0
        syms.append(complex(i_level * SQRT2_INV, q_level * SQRT2_INV))
    return syms, pad


def add_awgn(symbols: list[complex], ebn0_db: float, rng: random.Random, bits_per_symbol: int = 2, code_rate: float = 1.0) -> list[complex]:
    ebn0 = 10.0 ** (ebn0_db / 10.0)
    esn0 = ebn0 * bits_per_symbol * code_rate
    noise_var = 1.0 / (2.0 * esn0)
    noise_std = math.sqrt(noise_var)
    out: list[complex] = []
    for s in symbols:
        n = complex(rng.gauss(0.0, noise_std), rng.gauss(0.0, noise_std))
        out.append(s + n)
    return out


def qpsk_hard_demod_gray(rx: list[complex], pad_bits: int) -> list[int]:
    out: list[int] = []
    for r in rx:
        out.append(0 if r.real >= 0.0 else 1)
        out.append(0 if r.imag >= 0.0 else 1)
    if pad_bits:
        out = out[:-pad_bits]
    return out


def simulate_one(payload: bytes, ebn0_db: float, rng: random.Random) -> tuple[bytes, int, int]:
    tx_bits = bytes_to_bits_lsb_first(payload)
    tx_sym, pad = qpsk_mod_gray(tx_bits)
    rx_sym = add_awgn(tx_sym, ebn0_db=ebn0_db, rng=rng)
    rx_bits = qpsk_hard_demod_gray(rx_sym, pad_bits=pad)
    bit_errors = sum(1 for a, b in zip(tx_bits, rx_bits) if a != b)
    rx_payload = bits_to_bytes_lsb_first(rx_bits)
    return rx_payload, len(tx_bits), bit_errors


def main() -> None:
    args = parse_args()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in in_dir.iterdir() if p.is_file() and p.suffix == ".bin")
    if not files:
        raise FileNotFoundError(f"No .bin files found under {in_dir}")

    rng = random.Random(args.seed)
    rows: list[tuple[str, int, int, float, int]] = []
    total_bits = 0
    total_err = 0
    n_exact = 0

    for p in files:
        tx_payload = p.read_bytes()
        rx_payload, n_bits, n_err = simulate_one(tx_payload, ebn0_db=args.ebn0_db, rng=rng)
        exact_match = int(tx_payload == rx_payload)
        if exact_match:
            n_exact += 1
        total_bits += n_bits
        total_err += n_err
        ber = (n_err / n_bits) if n_bits > 0 else 0.0
        rows.append((p.name, len(tx_payload), n_bits, ber, exact_match))
        (out_dir / p.name).write_bytes(rx_payload)

    avg_ber = (total_err / total_bits) if total_bits > 0 else 0.0
    print(f"Processed {len(files)} files at Eb/N0={args.ebn0_db:.2f} dB")
    print(f"Total BER: {avg_ber:.6e} ({total_err}/{total_bits})")
    print(f"Exact byte match files: {n_exact}/{len(files)}")
    print(f"Decoded files written to: {out_dir}")

    csv_path = Path(args.csv) if args.csv else out_dir / "qpsk_awgn_report.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "payload_bytes", "n_bits", "ber", "exact_byte_match"])
        for r in rows:
            w.writerow([r[0], r[1], r[2], f"{r[3]:.8e}", r[4]])
    print(f"Report written to: {csv_path}")


if __name__ == "__main__":
    main()
