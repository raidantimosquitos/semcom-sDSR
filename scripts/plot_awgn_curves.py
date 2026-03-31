#!/usr/bin/env python3
"""
Plot AUC/pAUC vs SNR from AWGN evaluation CSVs.

Inputs:
- scripts/evaluate_awgn.py output (bitstream-first)
- scripts/evaluate_awgn_jscc.py output (JSCC)

This script expects wide rows with columns that include:
  machine_type, snr_db, seed, machine_id, auc, pauc
and optionally cu_total / avg_cu_total for labeling.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=str, nargs="+", required=True, help="One or more CSV files.")
    p.add_argument("--machine_id", type=str, default="average", help="Which machine_id row to plot (default: average).")
    p.add_argument("--metric", type=str, choices=["auc", "pauc"], default="auc")
    p.add_argument("--out", type=str, required=True, help="Output .png path.")
    return p.parse_args()


@dataclass(frozen=True)
class Row:
    method: str
    label: str
    snr_db: float
    seed: int
    value: float


def _read_rows(path: Path, metric: str, machine_id: str) -> list[Row]:
    rows: list[Row] = []
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for d in r:
            mid = d.get("machine_id", "")
            if mid != machine_id:
                continue
            snr = float(d["snr_db"])
            seed = int(float(d.get("seed", 0)))
            val = float(d[metric])
            method = d.get("method", "unknown")
            # build a human label
            if method == "bitstream":
                uep = d.get("uep", "")
                cu = d.get("avg_cu_total", "")
                label = f"bitstream-{uep} (cu≈{cu})" if cu else f"bitstream-{uep}"
            elif method == "jscc":
                cu = d.get("cu_total", "")
                label = f"jscc (cu={cu})" if cu else "jscc"
            else:
                label = method
            rows.append(Row(method=method, label=label, snr_db=snr, seed=seed, value=val))
    return rows


def _mean_over_seeds(rows: Iterable[Row]) -> dict[str, list[tuple[float, float]]]:
    # label -> snr -> mean(value)
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r.label][r.snr_db].append(r.value)
    out: dict[str, list[tuple[float, float]]] = {}
    for label, by_snr in by.items():
        pts = []
        for snr, vals in by_snr.items():
            pts.append((float(snr), sum(vals) / len(vals)))
        out[label] = sorted(pts, key=lambda x: x[0])
    return out


def main() -> None:
    args = parse_args()
    in_paths = [Path(p) for p in args.inputs]
    all_rows: list[Row] = []
    for p in in_paths:
        all_rows.extend(_read_rows(p, metric=args.metric, machine_id=args.machine_id))

    curves = _mean_over_seeds(all_rows)
    if not curves:
        raise SystemExit(f"No rows found for machine_id={args.machine_id!r} across inputs.")

    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise SystemExit("matplotlib is required: pip install matplotlib") from e

    plt.figure(figsize=(7, 4))
    for label, pts in curves.items():
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plt.plot(xs, ys, marker="o", label=label)
    plt.xlabel("SNR (dB)")
    plt.ylabel(args.metric.upper())
    plt.title(f"{args.metric.upper()} vs SNR (machine_id={args.machine_id})")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

