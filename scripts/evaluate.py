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

  Joint multi-type (same composite IDs as training):
  python scripts/evaluate.py ... --machine_types fan pump slider
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Tuple

import torch
from torch.utils.data import DataLoader

from src.data.dataset import (
    COMPOSITE_ID_SEP,
    DCASE2020Task2LogMelDataset,
    DCASE2020Task2TestDataset,
)
from src.engine.evaluator import AnomalyEvaluator
from src.models.vq_vae.autoencoders import VQ_VAE_2Layer
from src.models.sDSR.s_dsr import sDSR, sDSRConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--stage1_ckpt", type=str, required=True, help="Stage 1 checkpoint (encoder, codebook, general decoder)")
    p.add_argument("--stage2_ckpt", type=str, required=True, help="Stage 2 checkpoint (object-specific decoder, anomaly detector)")
    p.add_argument("--data_path", type=str, required=True, help="Path to DCASE root")
    p.add_argument("--machine_type", type=str, default="fan", help="Single machine type (ignored if --machine_types is set)")
    p.add_argument(
        "--machine_types",
        type=str,
        nargs="+",
        default=None,
        help="Multiple machine types: joint calibration (train normal) and test with composite machine IDs",
    )
    p.add_argument("--machine_id", type=str, default=None, help="If set, evaluate only on this machine_id (single-type only)")
    p.add_argument("--output", type=str, default=None, help="CSV output path (default: <stage2_ckpt_parent>/results/results.csv)")
    p.add_argument("--plot", type=str, default=None, help="Path to save comparison plot (default: <stage2_ckpt_parent>/results/comparison.png)")
    p.add_argument("--pauc_max_fpr", type=float, default=0.1)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--no_score_norm", action="store_true", help="Disable machine-ID conditioned anomaly score normalization")
    return p.parse_args()


def build_s_dsr(n_mels: int, T: int, vq_vae: VQ_VAE_2Layer, embedding_dim: Tuple[int, int], hidden_channels: Tuple[int, int]) -> sDSR:
    cfg = sDSRConfig(
        embedding_dim=embedding_dim,
        hidden_channels=hidden_channels,
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
            probs = torch.softmax(m_out, dim=1)
            anomaly_prob = probs[:, 1]
            sc_mean = anomaly_prob.view(m_out.shape[0], -1).mean(dim=1).cpu()
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


def _aggregate_per_machine_type(
    ids: dict[str, Any],
) -> dict[str, dict[str, float]] | None:
    """
    When machine_ids are composite ``{machine_type}__{id_XX}``, return mean AUC/pAUC
    per DCASE machine_type (mean over that type's ids). Returns None if no composite ids.
    """
    by_type: dict[str, list[tuple[float, float]]] = defaultdict(list)
    saw_composite = False
    for k, v in ids.items():
        if k == "average" or not isinstance(v, dict):
            continue
        if COMPOSITE_ID_SEP not in k:
            continue
        saw_composite = True
        base, _rest = k.split(COMPOSITE_ID_SEP, 1)
        by_type[base].append((float(v["auc"]), float(v["pauc"])))
    if not saw_composite or not by_type:
        return None
    out: dict[str, dict[str, float]] = {}
    for mt in sorted(by_type.keys()):
        pairs = by_type[mt]
        auc_m = sum(p[0] for p in pairs) / len(pairs)
        pauc_m = sum(p[1] for p in pairs) / len(pairs)
        out[mt] = {"auc": auc_m, "pauc": pauc_m}
    return out


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
    stage1_ckpt = torch.load(args.stage1_ckpt, map_location="cpu", weights_only=True)

    if args.machine_types is not None:
        if args.machine_id is not None:
            raise ValueError("--machine_id cannot be used with --machine_types")
        train_ds = DCASE2020Task2LogMelDataset(
            root=args.data_path,
            machine_types=list(args.machine_types),
            include_test=False,
        )
        test_ds = DCASE2020Task2TestDataset(
            root=args.data_path,
            machine_types=list(args.machine_types),
            target_T=train_ds.target_T,
        )
    else:
        train_ds = DCASE2020Task2LogMelDataset(
            root=args.data_path,
            machine_type=args.machine_type,
            machine_id=args.machine_id,
        )
        test_ds = DCASE2020Task2TestDataset(
            root=args.data_path,
            machine_type=args.machine_type,
            target_T=train_ds.target_T,
            machine_id=args.machine_id,
        )
    _, _, n_mels, T = train_ds.data.shape

    # Architecture from checkpoint (same as training); fallback for old checkpoints
    num_embeddings_coarse = stage1_ckpt["num_embeddings_coarse"]
    num_embeddings_fine = stage1_ckpt["num_embeddings_fine"]
    embedding_dim_coarse = stage1_ckpt["embedding_dim_coarse"]
    embedding_dim_fine = stage1_ckpt["embedding_dim_fine"]
    hidden_channels_coarse = stage1_ckpt["hidden_channels_coarse"]
    hidden_channels_fine = stage1_ckpt["hidden_channels_fine"]
    num_residual_layers = stage1_ckpt["num_residual_layers"]
    # Stage 1: encoder, codebook (VQ coarse/fine), and General Object Decoder
    vq_vae = VQ_VAE_2Layer(
        hidden_channels=(hidden_channels_coarse, hidden_channels_fine),
        num_residual_layers=num_residual_layers,
        num_embeddings=(num_embeddings_coarse, num_embeddings_fine),
        embedding_dim=(embedding_dim_coarse, embedding_dim_fine),
        commitment_cost=0.25,
        decay=0.99,
    )
    from src.utils.checkpoint_compat import migrate_vq_vae_state_dict
    state = dict(stage1_ckpt["model_state_dict"])
    migrate_vq_vae_state_dict(state)
    vq_vae.load_state_dict(state)

    # Full sDSR: Stage 1 modules (frozen in training) + Stage 2 modules
    model = build_s_dsr(
        n_mels, T, 
        vq_vae=vq_vae, 
        embedding_dim=(embedding_dim_coarse, embedding_dim_fine), 
        hidden_channels=(hidden_channels_coarse, hidden_channels_fine)
        )

    stage2 = torch.load(args.stage2_ckpt, map_location="cpu", weights_only=True)
    stage2_state = dict(stage2["model_state_dict"])
    migrate_vq_vae_state_dict(stage2_state)
    model.load_state_dict(stage2_state)

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

    for run_name, ids in results.items():
        tee(f"\n{run_name}:")
        for k, v in ids.items():
            if isinstance(v, dict):
                tee(f"  {k}: AUC={v['auc']:.4f} pAUC={v['pauc']:.4f}")

        per_type = _aggregate_per_machine_type(ids)
        if per_type is not None:
            tee("\n  Per machine_type (mean over machine_ids):")
            for mt in sorted(per_type.keys()):
                pt = per_type[mt]
                tee(f"    {mt}: AUC={pt['auc']:.4f} pAUC={pt['pauc']:.4f}")
            macro_auc = sum(per_type[mt]["auc"] for mt in per_type) / len(per_type)
            macro_pauc = sum(per_type[mt]["pauc"] for mt in per_type) / len(per_type)
            tee(
                f"\n  Macro over machine_types (mean of per-type averages): "
                f"AUC={macro_auc:.4f} pAUC={macro_pauc:.4f}"
            )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["machine_type", "machine_id", "AUC", "pAUC"])
            for run_name, ids in results.items():
                for k, v in ids.items():
                    if isinstance(v, dict):
                        w.writerow([run_name, k, f"{v['auc']:.4f}", f"{v['pauc']:.4f}"])
                per_type = _aggregate_per_machine_type(ids)
                if per_type is not None:
                    for mt in sorted(per_type.keys()):
                        pt = per_type[mt]
                        w.writerow(
                            [
                                run_name,
                                f"__type_avg__{mt}",
                                f"{pt['auc']:.4f}",
                                f"{pt['pauc']:.4f}",
                            ]
                        )
                    macro_auc = sum(per_type[mt]["auc"] for mt in per_type) / len(per_type)
                    macro_pauc = sum(per_type[mt]["pauc"] for mt in per_type) / len(per_type)
                    w.writerow(
                        [
                            run_name,
                            "__macro_type_avg__",
                            f"{macro_auc:.4f}",
                            f"{macro_pauc:.4f}",
                        ]
                    )
        tee(f"\nResults saved to {out_path}")

    if args.plot and results:
        _plot_comparison(results, Path(args.plot), log=tee)


def _plot_comparison(
    results: dict, save_path: Path, log: Callable[[str], None] | None = None
) -> None:
    """Bar chart for AUC and pAUC: overall average and (if composite IDs) per machine_type."""
    if log is None:
        log = print
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        log("Skipping plot: matplotlib required (pip install matplotlib)")
        return

    for run_name, ids in results.items():
        save_path.parent.mkdir(parents=True, exist_ok=True)

        per_type = _aggregate_per_machine_type(ids)
        if per_type is not None:
            types = sorted(per_type.keys())
            x = np.arange(len(types))
            width = 0.35
            aucs = [per_type[t]["auc"] for t in types]
            paucs = [per_type[t]["pauc"] for t in types]
            fig, ax = plt.subplots(figsize=(max(0.6 + len(types) * 2.0, 5), 4))
            ax.bar(x - width / 2, aucs, width, label="AUC", color="C0")
            ax.bar(x + width / 2, paucs, width, label="pAUC", color="C1")
            ax.set_xticks(x)
            ax.set_xticklabels(types, rotation=25, ha="right")
            ax.set_ylabel("Score")
            ax.set_title(f"Per machine_type average: {run_name}")
            ax.legend()
            plt.tight_layout()
            per_type_path = save_path.parent / f"{save_path.stem}_per_type{save_path.suffix}"
            plt.savefig(per_type_path, dpi=150)
            plt.close()
            log(f"Per-type plot saved to {per_type_path}")

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
        title = f"Anomaly detection: {run_name} (average over machine_ids)"
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        log(f"Plot saved to {save_path}")
        break


if __name__ == "__main__":
    main()
