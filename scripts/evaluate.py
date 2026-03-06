#!/usr/bin/env python3
"""
Evaluate sDSR on DCASE2020 Task 2 test set.

Uses the two-stage scheme: Stage 1 provides the Discrete Encoder, codebook (VQ1/VQ2),
and General Object Decoder; Stage 2 provides the Object Specific Decoder and
Anomaly Detector.

Usage:
  python scripts/evaluate.py --stage1_ckpt checkpoints/stage1/fan/best.pt \\
    --stage2_ckpt checkpoints/stage2/fan/stage2_fan_final.pt \\
    --data_path /path/to/dcase --machine_type fan [--output results_fan.csv]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader

from src.data.dataset import (
    DCASE2020Task2LogMelDataset,
    DCASE2020Task2TestDataset,
    get_norm_stats_from_stage1_ckpt,
)
from src.engine.evaluator import AnomalyEvaluator
from src.models.vq_vae.autoencoders import VQ_VAE_2Layer
from src.models.sDSR.s_dsr import sDSR, sDSRConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--stage1_ckpt", type=str, required=True, help="Stage 1 checkpoint (encoder, codebook, general decoder)")
    p.add_argument("--stage2_ckpt", type=str, required=True, help="Stage 2 checkpoint (object-specific decoder, anomaly detector)")
    p.add_argument("--data_path", type=str, required=True, help="Path to DCASE root")
    p.add_argument("--machine_type", type=str, default="fan")
    p.add_argument("--output", type=str, default=None, help="CSV output path (default: <stage2_ckpt_parent>/results/results.csv)")
    p.add_argument("--plot", type=str, default=None, help="Path to save comparison plot (default: <stage2_ckpt_parent>/results/comparison.png)")
    p.add_argument("--pauc_max_fpr", type=float, default=0.1)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--no_score_norm", action="store_true", help="Disable machine-ID conditioned anomaly score normalization")
    return p.parse_args()


def build_s_dsr(n_mels: int, T: int, vq_vae: VQ_VAE_2Layer, embedding_dim: int) -> sDSR:
    cfg = sDSRConfig(
        embedding_dim=embedding_dim,
        num_hiddens=128,
        n_mels=n_mels,
        T=T,
    )
    return sDSR(vq_vae, cfg)


def _compute_train_score_stats(
    model: torch.nn.Module,
    train_ds: DCASE2020Task2LogMelDataset,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, tuple[float, float]], tuple[float, float]]:
    """
    Run model on all train samples, collect anomaly score (mean) per sample with machine_id.
    Return per-machine_id (mean, std) and global (mean, std) fallback.
    """
    from collections import defaultdict

    model.eval()
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    # (score_mean, machine_id) per sample
    by_id: dict[str, list[float]] = defaultdict(list)
    all_scores: list[float] = []

    with torch.no_grad():
        for batch in loader:
            x, _labels, machine_ids = batch
            x = x.to(device)
            m_out = model(x)
            logits = m_out[:, 1]
            flat = logits.view(m_out.shape[0], -1)
            sc_mean = flat.mean(dim=1).cpu()
            for i in range(x.shape[0]):
                mid = machine_ids[i] if isinstance(machine_ids[i], str) else str(machine_ids[i])
                s = sc_mean[i].item()
                by_id[mid].append(s)
                all_scores.append(s)

    train_score_stats: dict[str, tuple[float, float]] = {}
    for mid, scores in by_id.items():
        arr = scores
        mean_val = sum(arr) / len(arr)
        var = sum((x - mean_val) ** 2 for x in arr) / len(arr) if len(arr) > 1 else 0.0
        std_val = var ** 0.5
        train_score_stats[mid] = (mean_val, std_val)

    if all_scores:
        global_mean = sum(all_scores) / len(all_scores)
        global_var = sum((x - global_mean) ** 2 for x in all_scores) / len(all_scores)
        global_std = global_var ** 0.5
        fallback = (global_mean, global_std)
    else:
        fallback = (0.0, 1.0)

    return train_score_stats, fallback


def main() -> None:
    args = parse_args()

    # Results directory: same parent as stage2 checkpoint (e.g. checkpoints/stage2/fan/results/)
    results_dir = Path(args.stage2_ckpt).resolve().parent / "results"
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
    # Load Stage 1 checkpoint once for norm stats and model weights
    stage1_ckpt = torch.load(args.stage1_ckpt, map_location="cpu", weights_only=True)
    norm_mean, norm_std = get_norm_stats_from_stage1_ckpt(stage1_ckpt, args.machine_type)
    if norm_mean is not None and norm_std is not None and "target_T" in stage1_ckpt:
        train_ds = DCASE2020Task2LogMelDataset(
            root=args.data_path,
            machine_type=args.machine_type,
            normalize=True,
            norm_mean=norm_mean,
            norm_std=norm_std,
            target_T_override=stage1_ckpt["target_T"],
        )
    else:
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

    num_embeddings_top = stage1_ckpt["num_embeddings_top"]
    num_embeddings_bot = stage1_ckpt["num_embeddings_bot"]
    embedding_dim = stage1_ckpt["embedding_dim"]
    # Stage 1: encoder, codebook (VQ1/VQ2), and General Object Decoder
    vq_vae = VQ_VAE_2Layer(
        num_hiddens=128,
        num_residual_layers=2,
        num_residual_hiddens=64,
        num_embeddings=(num_embeddings_top, num_embeddings_bot),
        embedding_dim=embedding_dim,
        commitment_cost=0.25,
        decay=0.99,
    )
    vq_vae.load_state_dict(stage1_ckpt["model_state_dict"])

    # Full sDSR: Stage 1 modules (frozen in training) + Stage 2 modules
    model = build_s_dsr(n_mels, T, vq_vae, embedding_dim)

    stage2 = torch.load(args.stage2_ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(stage2["model_state_dict"])

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_score_stats: dict[str, tuple[float, float]] | None = None
    train_score_stats_fallback: tuple[float, float] | None = None
    if not args.no_score_norm:
        train_score_stats, train_score_stats_fallback = _compute_train_score_stats(
            model, train_ds, device, args.batch_size
        )
        tee("Calibrated per-machine_id anomaly score stats (mean score normalization enabled).")

    evaluator = AnomalyEvaluator(
        model=model,
        test_dataset=test_ds,
        device=args.device,
        pauc_max_fpr=args.pauc_max_fpr,
        batch_size=args.batch_size,
        train_score_stats=train_score_stats,
        train_score_stats_fallback=train_score_stats_fallback,
    )
    results = evaluator.evaluate()

    for mt, ids in results.items():
        tee(f"\n{mt}:")
        for k, v in ids.items():
            if isinstance(v, dict):
                tee(f"  {k}: AUC={v['auc']:.4f} pAUC={v['pauc']:.4f}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["machine_type", "machine_id", "AUC", "pAUC"])
            for mt, ids in results.items():
                for k, v in ids.items():
                    if isinstance(v, dict):
                        w.writerow([mt, k, f"{v['auc']:.4f}", f"{v['pauc']:.4f}"])
        tee(f"\nResults saved to {out_path}")

    if args.plot and results:
        _plot_comparison(results, Path(args.plot), log=tee)


def _plot_comparison(
    results: dict, save_path: Path, log: Callable[[str], None] | None = None
) -> None:
    """Bar chart for AUC and pAUC (mean anomaly score)."""
    if log is None:
        log = print
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        log("Skipping plot: matplotlib required (pip install matplotlib)")
        return

    for mt, ids in results.items():
        if "average" not in ids:
            continue
        v = ids["average"]
        fig, ax = plt.subplots(figsize=(5, 4))
        x = np.arange(2)
        vals = [v["auc"], v["pauc"]]
        labels = ["AUC", "pAUC"]
        ax.bar(x, vals, color=["C0", "C1"])
        ax.set_ylabel("Score")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(f"Anomaly detection: {mt} (average)")
        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
        log(f"Plot saved to {save_path}")
        break


if __name__ == "__main__":
    main()
