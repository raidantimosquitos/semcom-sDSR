#!/usr/bin/env python3
"""
Main training CLI: train stage1 | stage2 | full

Usage:
  python scripts/train.py stage1 --data_path /path --machine_type fan --n_iter 20000
  python scripts/train.py stage1 --data_path /path --machine_type fan pump slider --n_iter 20000
  python scripts/train.py stage2 --data_path /path --machine_type fan --stage1_ckpt ./checkpoints/stage1/fan/best.pt --n_iter 10000
  python scripts/train.py full --data_path /path --machine_type fan
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal, Tuple

import torch
from torch.utils.data import Subset

from src.data.dataset import (
    DCASE2020Task2LogMelDataset,
    AudDSRAnomTrainDataset,
)
from src.engine.stage1 import Stage1Trainer
from src.engine.stage2 import Stage2Trainer
from src.models.vq_vae.autoencoders import VQ_VAE_2Layer
from src.models.sDSR.s_dsr import sDSR, sDSRConfig
from src.utils.stage1_norm import load_norm_from_stage1_ckpt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="sDSR training")
    sub = parser.add_subparsers(dest="stage", required=True)

    # Common args
    for p in (sub.add_parser("stage1"), sub.add_parser("stage2"), sub.add_parser("full")):
        p.add_argument("--data_path", type=str, required=True, help="Path to DCASE root (machine/train/normal/*.wav)")
        p.add_argument("--ckpt_dir", type=str, default="./checkpoints", help="Checkpoint directory")
        p.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
        p.add_argument("--log_every", type=int, default=100)
        p.add_argument("--ckpt_every", type=int, default=500)
        p.add_argument("--no_amp", action="store_true", help="Disable AMP")

    # Stage 1: machine_type can be a list for multi-type training
    s1 = sub.choices["stage1"]
    s1.add_argument("--machine_type", type=str, nargs="+", default=["fan", "pump", "slider", "valve", "ToyCar", "ToyConveyor"], help="One or more machine types (e.g. fan pump slider)")
    s1.add_argument("--batch_size", type=int, default=256, help="Batch size (default: 256)")
    s1.add_argument("--n_iter", type=int, default=20000)
    s1.add_argument("--lr", type=float, default=1e-4)
    s1.add_argument("--num_embeddings_coarse", type=int, default=512)
    s1.add_argument("--num_embeddings_fine", type=int, default=4096)
    s1.add_argument("--embedding_dim_fine", type=int, default=64)
    s1.add_argument("--embedding_dim_coarse", type=int, default=256)
    s1.add_argument("--hidden_channels_fine", type=int, default=64)
    s1.add_argument("--hidden_channels_coarse", type=int, default=256)
    s1.add_argument("--commitment_cost", type=float, default=0.25)
    s1.add_argument("--decay", type=float, default=0.99)
    s1.add_argument("--lambda_recon", type=float, default=1.0)
    s1.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    s1.add_argument("--include_test", action="store_true", help="Include test data in stage1 training")

    # Stage 2
    s2 = sub.choices["stage2"]
    s2.add_argument("--machine_type", type=str, default="fan", help="Machine type (e.g. fan, pump)")
    s2.add_argument("--machine_id", type=str, default=None, help="If set, train on this machine_id only; other IDs (same type) used as adversarial samples (mask all 1s)")
    s2.add_argument("--batch_size", type=int, default=16, help="Batch size (default: 16)")
    s2.add_argument("--n_iter", type=int, default=10000)
    s2.add_argument("--stage1_ckpt", type=str, required=True, help="Path to Stage 1 checkpoint")
    s2.add_argument("--lr", type=float, default=2e-4)
    s2.add_argument("--lambda_recon", type=float, default=10.0)
    s2.add_argument("--lambda_focal", type=float, default=1.0)
    s2.add_argument("--lambda_sub", type=float, default=1.0, help="Weight for subspace restriction loss L2(F̃, Q)")
    s2.add_argument("--anomaly_strategy", type=str, default="both", choices=["perlin", "machine_specific", "both"], help="Synthetic anomaly mask strategy, perlin noise, machine-specific masks or both")
    s2.add_argument("--fine_only_prob", type=float, default=0.65, help="Fraction of anomaly samples that inject at fine level only; rest inject at both levels (default 0.65 = 65%% fine-only, 35%% both)")
    s2.add_argument("--anomaly_sampling", type=str, default="distant", choices=["distant", "uniform"], help="Anomaly sampling strategy")
    s2.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")

    # Full
    full = sub.choices["full"]
    full.add_argument("--machine_type", type=str, default="fan", help="Machine type (e.g. fan, pump)")
    full.add_argument("--batch_size_s1", type=int, default=128, help="Stage 1 batch size")
    full.add_argument("--batch_size_s2", type=int, default=16, help="Stage 2 batch size")
    full.add_argument("--stage1_iter", type=int, default=20000)
    full.add_argument("--stage2_iter", type=int, default=10000)

    return parser.parse_args()


def build_vq_vae(
    n_mels: int,
    T: int,
    num_embeddings: Tuple[int, int],
    embedding_dim: Tuple[int, int],
    hidden_channels: Tuple[int, int],
    num_residual_layers: int = 2,
    commitment_cost: float = 0.25,
    decay: float = 0.99,
) -> VQ_VAE_2Layer:
    return VQ_VAE_2Layer(
        hidden_channels = hidden_channels,
        num_residual_layers=num_residual_layers,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        commitment_cost=commitment_cost,
        decay=decay,
    )


def build_s_dsr(
    vq_vae: VQ_VAE_2Layer,
    n_mels: int,
    T: int,
    hidden_channels: Tuple[int, int],
    embedding_dim: Tuple[int, int],
    anomaly_sampling: Literal["distant", "uniform"] = "distant",
    fine_only_prob: float = 0.65,
) -> sDSR:
    cfg = sDSRConfig(
        embedding_dim=embedding_dim,
        hidden_channels=hidden_channels,
        num_residual_layers=2,
        n_mels=n_mels,
        T=T,
        anomaly_sampling=anomaly_sampling,
        fine_only_prob=fine_only_prob,
    )
    return sDSR(vq_vae, cfg)


def run_stage1(args: argparse.Namespace) -> None:
    machine_types = args.machine_type if isinstance(args.machine_type, list) else [args.machine_type]
    if len(machine_types) == 1:
        dataset = DCASE2020Task2LogMelDataset(
            root=args.data_path,
            machine_type=machine_types[0],
        )
        run_name = machine_types[0]
    else:
        include_test = args.include_test
        dataset = DCASE2020Task2LogMelDataset(
            root=args.data_path,
            machine_types=machine_types,
            include_test=include_test,
        )
        run_name = "+".join(sorted(machine_types))
    _, _, n_mels, T = dataset.data.shape

    model = build_vq_vae(
        n_mels, T, 
        hidden_channels=(args.hidden_channels_coarse, args.hidden_channels_fine),
        embedding_dim=(args.embedding_dim_coarse, args.embedding_dim_fine),
        num_embeddings=(args.num_embeddings_coarse, args.num_embeddings_fine),
        num_residual_layers=2,
        commitment_cost=args.commitment_cost,
        decay=args.decay,
    )
    trainer = Stage1Trainer(
        model=model,
        dataset=dataset,
        machine_type=run_name,
        lambda_recon=args.lambda_recon,
        lr=args.lr,
        batch_size=args.batch_size,
        log_every=args.log_every,
        ckpt_every=args.ckpt_every,
        ckpt_dir=args.ckpt_dir,
        use_amp=not args.no_amp,
        device=args.device,
    )
    trainer.train(n_iterations=args.n_iter, resume_from=args.resume)


def run_stage2(args: argparse.Namespace) -> None:
    ckpt = torch.load(args.stage1_ckpt, map_location="cpu", weights_only=True)

    ds_common = dict(
        root=args.data_path,
        machine_type=args.machine_type,
    )
    q_vae_dataset = DCASE2020Task2LogMelDataset(
        **ds_common,
        machine_id=args.machine_id,
    )
    _, _, n_mels, T = q_vae_dataset.data.shape

    adversarial_dataset = None
    if args.machine_id is not None:
        # Full machine_type dataset (no machine_id filter) for adversarial samples
        q_vae_full = DCASE2020Task2LogMelDataset(**ds_common)
        adversarial_indices = [
            i for i in range(len(q_vae_full._machine_id_strs))
            if q_vae_full._machine_id_strs[i] != args.machine_id
        ]
        if adversarial_indices:
            adversarial_dataset = Subset(q_vae_full, adversarial_indices)

    train_dataset = AudDSRAnomTrainDataset(
        q_vae_dataset,
        strategy=args.anomaly_strategy,
        zero_mask_prob=0.5,
        adversarial_dataset=adversarial_dataset,
    )

    # Load VQ-VAE from Stage 1 (same architecture as training; support old checkpoint keys)
    from src.utils.checkpoint_compat import migrate_vq_vae_state_dict
    num_embeddings_coarse = ckpt["num_embeddings_coarse"]
    num_embeddings_fine = ckpt["num_embeddings_fine"]
    embedding_dim_fine = ckpt["embedding_dim_fine"]
    embedding_dim_coarse = ckpt["embedding_dim_coarse"]
    hidden_channels_fine = ckpt["hidden_channels_fine"]
    hidden_channels_coarse = ckpt["hidden_channels_coarse"]
    num_residual_layers = ckpt["num_residual_layers"]
    vq_vae = build_vq_vae(
        n_mels, T, 
        num_embeddings=(num_embeddings_coarse, num_embeddings_fine),
        embedding_dim=(embedding_dim_coarse, embedding_dim_fine),
        hidden_channels=(hidden_channels_coarse, hidden_channels_fine),
        num_residual_layers=num_residual_layers,
    )
    state = dict(ckpt["model_state_dict"])
    migrate_vq_vae_state_dict(state)
    vq_vae.load_state_dict(state)

    model = build_s_dsr(
        vq_vae, n_mels, T, 
        hidden_channels=(hidden_channels_coarse, hidden_channels_fine),
        embedding_dim=(embedding_dim_coarse, embedding_dim_fine),
        fine_only_prob=getattr(args, "fine_only_prob", 0.65),
        anomaly_sampling=args.anomaly_sampling,
    )
    trainer = Stage2Trainer(
        model=model,
        dataset=train_dataset,
        machine_type=args.machine_type,
        machine_id=args.machine_id,
        lambda_recon=args.lambda_recon,
        lambda_focal=args.lambda_focal,
        lambda_sub=args.lambda_sub,
        lr=args.lr,
        batch_size=args.batch_size,
        log_every=args.log_every,
        ckpt_every=args.ckpt_every,
        ckpt_dir=args.ckpt_dir,
        use_amp=not args.no_amp,
        device=args.device,
    )
    trainer.train(n_iterations=args.n_iter, resume_from=args.resume)


def run_full(args: argparse.Namespace) -> None:
    stage1_args = argparse.Namespace(
        data_path=args.data_path,
        machine_type=args.machine_type,
        ckpt_dir=args.ckpt_dir,
        device=args.device,
        batch_size=args.batch_size_s1,
        log_every=args.log_every,
        ckpt_every=args.ckpt_every,
        no_amp=args.no_amp,
        n_iter=args.stage1_iter,
        lr=2e-4,
        lambda_recon=1.0,
        resume=None,
    )
    run_stage1(stage1_args)

    ckpt_path = Path(args.ckpt_dir) / "stage1" / args.machine_type
    ckpt_files = sorted(ckpt_path.glob("stage1_*.pt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No stage1 checkpoint found in {ckpt_path}")
    stage1_ckpt = str(ckpt_files[-1])

    stage2_args = argparse.Namespace(
        data_path=args.data_path,
        machine_type=args.machine_type,
        ckpt_dir=args.ckpt_dir,
        device=args.device,
        batch_size=args.batch_size_s2,
        log_every=args.log_every,
        ckpt_every=args.ckpt_every,
        no_amp=args.no_amp,
        n_iter=args.stage2_iter,
        stage1_ckpt=stage1_ckpt,
        lr=2e-4,
        lambda_recon=10.0,
        lambda_focal=1.0,
        lambda_sub=1.0,
        anomaly_strategy="both",
        resume=None,
    )
    run_stage2(stage2_args)


def main() -> None:
    args = parse_args()
    if args.stage == "stage1":
        run_stage1(args)
    elif args.stage == "stage2":
        run_stage2(args)
    else:
        run_full(args)


if __name__ == "__main__":
    main()
