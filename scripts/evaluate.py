#!/usr/bin/env python3
"""
Evaluate sDSR on DCASE2020 Task 2 test set.

Usage:
  python scripts/evaluate.py --ckpt checkpoints/stage2/fan/stage2_fan_final.pt \\
    --data_path /path/to/dcase --machine_type fan [--output results_fan.csv]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from src.data.dataset import DCASE2020Task2LogMelDataset, DCASE2020Task2TestDataset
from src.engine.evaluator import AnomalyEvaluator
from src.models.vq_vae.autoencoders import VQ_VAE_2Layer
from src.models.sDSR.s_dsr import sDSR, sDSRConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, help="Stage 2 (sDSR) checkpoint path")
    p.add_argument("--data_path", type=str, required=True, help="Path to DCASE root")
    p.add_argument("--machine_type", type=str, default="fan")
    p.add_argument("--output", type=str, default=None, help="Optional CSV output path")
    p.add_argument("--pauc_max_fpr", type=float, default=0.1)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=32)
    return p.parse_args()


def build_s_dsr(n_mels: int, T: int, vq_vae: VQ_VAE_2Layer) -> sDSR:
    cfg = sDSRConfig(
        embedding_dim=128,
        num_hiddens=128,
        n_mels=n_mels,
        T=T,
    )
    return sDSR(vq_vae, cfg)


def main() -> None:
    args = parse_args()

    # Train dataset defines normalization (mean, std) and target_T; pass to test for consistency
    train_ds = DCASE2020Task2LogMelDataset(
        root=args.data_path,
        machine_type=args.machine_type,
        normalize=True,
    )
    _, _, n_mels, T = train_ds.data.shape

    test_ds = DCASE2020Task2TestDataset(
        root=args.data_path,
        machine_type=args.machine_type,
        mean=train_ds.mean,
        std=train_ds.std,
        target_T=train_ds.target_T,
    )

    vq_vae = VQ_VAE_2Layer(
        num_hiddens=128,
        num_residual_layers=2,
        num_residual_hiddens=64,
        num_embeddings=(4096, 4096),
        embedding_dim=128,
        commitment_cost=0.25,
        decay=0.99,
    )
    model = build_s_dsr(n_mels, T, vq_vae)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])

    evaluator = AnomalyEvaluator(
        model=model,
        test_dataset=test_ds,
        device=args.device,
        pauc_max_fpr=args.pauc_max_fpr,
        batch_size=args.batch_size,
    )
    results = evaluator.evaluate()

    for mt, ids in results.items():
        print(f"\n{mt}:")
        for k, v in ids.items():
            if isinstance(v, dict):
                print(f"  {k}: AUC={v['auc']:.4f}  pAUC={v['pauc']:.4f}")

    if args.output:
        out_path = Path(args.output)
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["machine_type", "machine_id", "AUC", "pAUC"])
            for mt, ids in results.items():
                for k, v in ids.items():
                    if isinstance(v, dict) and k != "average":
                        w.writerow([mt, k, f"{v['auc']:.4f}", f"{v['pauc']:.4f}"])
                if "average" in ids:
                    w.writerow([mt, "average", f"{ids['average']['auc']:.4f}", f"{ids['average']['pauc']:.4f}"])
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
