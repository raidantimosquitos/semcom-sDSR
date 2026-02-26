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

import torch

from src.data.dataset import DCASE2020Task2LogMelDataset, AudDSRAnomTrainDataset
from src.engine.stage1 import Stage1Trainer
from src.engine.stage2 import Stage2Trainer
from src.models.vq_vae.autoencoders import VQ_VAE_2Layer
from src.models.sDSR.s_dsr import sDSR, sDSRConfig


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
    s1.add_argument("--machine_type", type=str, nargs="+", default=["fan"], help="One or more machine types (e.g. fan pump slider)")
    s1.add_argument("--batch_size", type=int, default=128, help="Batch size (default: 128)")
    s1.add_argument("--n_iter", type=int, default=20000)
    s1.add_argument("--lr", type=float, default=1e-4)
    s1.add_argument("--lambda_recon", type=float, default=1.0)
    s1.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")

    # Stage 2
    s2 = sub.choices["stage2"]
    s2.add_argument("--machine_type", type=str, default="fan", help="Machine type (e.g. fan, pump)")
    s2.add_argument("--batch_size", type=int, default=16, help="Batch size (default: 16)")
    s2.add_argument("--n_iter", type=int, default=10000)
    s2.add_argument("--stage1_ckpt", type=str, required=True, help="Path to Stage 1 checkpoint")
    s2.add_argument("--lr", type=float, default=2e-4)
    s2.add_argument("--lambda_recon", type=float, default=10.0)
    s2.add_argument("--lambda_focal", type=float, default=1.0)
    s2.add_argument("--lambda_sub", type=float, default=1.0, help="Weight for subspace restriction loss L2(F̃, Q)")
    s2.add_argument("--anomaly_strategy", type=str, default="both", choices=["perlin", "audio_specific", "both"], help="Synthetic anomaly mask strategy at dataset level")
    s2.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")

    # Full
    full = sub.choices["full"]
    full.add_argument("--machine_type", type=str, default="fan", help="Machine type (e.g. fan, pump)")
    full.add_argument("--batch_size_s1", type=int, default=128, help="Stage 1 batch size")
    full.add_argument("--batch_size_s2", type=int, default=16, help="Stage 2 batch size")
    full.add_argument("--stage1_iter", type=int, default=20000)
    full.add_argument("--stage2_iter", type=int, default=10000)

    return parser.parse_args()


def build_vq_vae(n_mels: int, T: int) -> VQ_VAE_2Layer:
    return VQ_VAE_2Layer(
        num_hiddens=128,
        num_residual_layers=2,
        num_residual_hiddens=64,
        num_embeddings=(1024, 4096),
        embedding_dim=128,
        commitment_cost=0.25,
        decay=0.99,
    )


def build_s_dsr(vq_vae: VQ_VAE_2Layer, n_mels: int, T: int) -> sDSR:
    cfg = sDSRConfig(
        embedding_dim=128,
        num_hiddens=128,
        num_residual_layers=2,
        num_residual_hiddens=64,
        n_mels=n_mels,
        T=T,
    )
    return sDSR(vq_vae, cfg)


def run_stage1(args: argparse.Namespace) -> None:
    machine_types = args.machine_type if isinstance(args.machine_type, list) else [args.machine_type]
    if len(machine_types) == 1:
        dataset = DCASE2020Task2LogMelDataset(
            root=args.data_path,
            machine_type=machine_types[0],
            normalize=True,
        )
        run_name = machine_types[0]
    else:
        dataset = DCASE2020Task2LogMelDataset(
            root=args.data_path,
            machine_types=machine_types,
            normalize=True,
        )
        run_name = "+".join(sorted(machine_types))
    _, _, n_mels, T = dataset.data.shape

    model = build_vq_vae(n_mels, T)
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
    q_vae_dataset = DCASE2020Task2LogMelDataset(
        root=args.data_path,
        machine_type=args.machine_type,
        normalize=True,
    )
    _, _, n_mels, T = q_vae_dataset.data.shape

    train_dataset = AudDSRAnomTrainDataset(
        q_vae_dataset,
        strategy=args.anomaly_strategy,
        zero_mask_prob=0.5,
    )

    # Load VQ-VAE from Stage 1
    vq_vae = build_vq_vae(n_mels, T)
    ckpt = torch.load(args.stage1_ckpt, map_location="cpu", weights_only=True)
    vq_vae.load_state_dict(ckpt["model_state_dict"])

    model = build_s_dsr(vq_vae, n_mels, T)
    trainer = Stage2Trainer(
        model=model,
        dataset=train_dataset,
        machine_type=args.machine_type,
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
        lr=1e-4,
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
