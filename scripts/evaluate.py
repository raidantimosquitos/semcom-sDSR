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
from typing import Callable

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
    p.add_argument("--output", type=str, default=None, help="CSV output path (default: <ckpt_parent>/results/results.csv)")
    p.add_argument("--plot", type=str, default=None, help="Path to save comparison plot (default: <ckpt_parent>/results/comparison.png)")
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

    # Results directory: same parent as checkpoint (e.g. checkpoints/stage2/fan/results/)
    results_dir = Path(args.ckpt).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    if args.output is None:
        args.output = str(results_dir / "results.csv")
    if args.plot is None:
        args.plot = str(results_dir / "comparison.png")

    log_path = results_dir / "evaluate.log"
    log_file = open(log_path, "w", encoding="utf-8")

    def tee(msg: str) -> None:
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    try:
        _run_evaluation(args, tee)
    finally:
        log_file.close()


def _run_evaluation(args: argparse.Namespace, tee: Callable[[str], None]) -> None:
    """Run evaluation; all user-facing output via tee (terminal + log file)."""
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
        num_embeddings=(1024, 4096),
        embedding_dim=128,
        commitment_cost=0.25,
        decay=0.99,
    )
    model = build_s_dsr(n_mels, T, vq_vae)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    print(ckpt.keys())
    exit()
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
        tee(f"\n{mt}:")
        for k, v in ids.items():
            if isinstance(v, dict):
                tee(
                    f"  {k}: mean AUC={v['auc']:.4f} pAUC={v['pauc']:.4f}  "
                    f"max AUC={v['auc_max']:.4f} pAUC={v['pauc_max']:.4f}  "
                    f"p95 AUC={v['auc_p95']:.4f} pAUC={v['pauc_p95']:.4f}"
                )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "machine_type", "machine_id",
                "AUC_mean", "pAUC_mean", "AUC_max", "pAUC_max", "AUC_p95", "pAUC_p95",
            ])
            for mt, ids in results.items():
                for k, v in ids.items():
                    if isinstance(v, dict) and k != "average":
                        w.writerow([
                            mt, k,
                            f"{v['auc']:.4f}", f"{v['pauc']:.4f}",
                            f"{v['auc_max']:.4f}", f"{v['pauc_max']:.4f}",
                            f"{v['auc_p95']:.4f}", f"{v['pauc_p95']:.4f}",
                        ])
                if "average" in ids:
                    v = ids["average"]
                    w.writerow([
                        mt, "average",
                        f"{v['auc']:.4f}", f"{v['pauc']:.4f}",
                        f"{v['auc_max']:.4f}", f"{v['pauc_max']:.4f}",
                        f"{v['auc_p95']:.4f}", f"{v['pauc_p95']:.4f}",
                    ])
        tee(f"\nResults saved to {out_path}")

    if args.plot and results:
        _plot_comparison(results, Path(args.plot), log=tee)


def _plot_comparison(
    results: dict, save_path: Path, log: Callable[[str], None] | None = None
) -> None:
    """Bar chart comparing AUC and pAUC for mean, max, and p95 aggregation methods."""
    if log is None:
        log = print
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        log("Skipping plot: matplotlib required (pip install matplotlib)")
        return

    # Use average metrics over the single machine_type
    for mt, ids in results.items():
        if "average" not in ids:
            continue
        v = ids["average"]
        methods = ["mean", "max", "p95"]
        aucs = [v["auc"], v["auc_max"], v["auc_p95"]]
        paucs = [v["pauc"], v["pauc_max"], v["pauc_p95"]]

        x = np.arange(len(methods))
        width = 0.35

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        ax_auc, ax_pauc = axes[0], axes[1]  # type: ignore[index]

        ax_auc.bar(x - width / 2, aucs, width, label="AUC")
        ax_auc.set_ylabel("AUC")
        ax_auc.set_title("AUC by score aggregation")
        ax_auc.set_xticks(x)
        ax_auc.set_xticklabels(methods)
        ax_auc.legend()

        ax_pauc.bar(x - width / 2, paucs, width, label="pAUC", color="C1")
        ax_pauc.set_ylabel("pAUC")
        ax_pauc.set_title("pAUC by score aggregation")
        ax_pauc.set_xticks(x)
        ax_pauc.set_xticklabels(methods)
        ax_pauc.legend()

        fig.suptitle(f"Anomaly detection: {mt} (average)")
        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
        log(f"Plot saved to {save_path}")
        break


if __name__ == "__main__":
    main()
