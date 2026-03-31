#!/usr/bin/env python3
"""
Train a lightweight JSCC model to transmit VQ-VAE-2 index maps over AWGN.

Training target: reconstruct the *quantized feature tensors* (q_coarse/q_fine)
from received symbols, using an MSE loss. This keeps the pipeline simple and
works well as a baseline before task-loss fine-tuning.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import DCASE2020Task2LogMelDataset
from src.models.vq_vae.autoencoders import VQ_VAE_2Layer
from src.utils.checkpoint_compat import migrate_vq_vae_state_dict
from src.comm.jscc_cnn import JSCCDualMap, JSCCMapConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--stage1_ckpt", type=str, required=True)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--machine_type", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--snr_min", type=float, default=-5.0)
    p.add_argument("--snr_max", type=float, default=20.0)
    p.add_argument("--alpha_c", type=int, default=10)
    p.add_argument("--alpha_f", type=int, default=3)
    p.add_argument("--out", type=str, required=True, help="Output .pt path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.stage1_ckpt, map_location="cpu", weights_only=True)
    vq_vae = VQ_VAE_2Layer(
        hidden_channels=(ckpt["hidden_channels_coarse"], ckpt["hidden_channels_fine"]),
        num_residual_layers=ckpt["num_residual_layers"],
        num_embeddings=(ckpt["num_embeddings_coarse"], ckpt["num_embeddings_fine"]),
        embedding_dim=(ckpt["embedding_dim_coarse"], ckpt["embedding_dim_fine"]),
        commitment_cost=0.25,
        decay=0.99,
    )
    st = dict(ckpt["model_state_dict"])
    migrate_vq_vae_state_dict(st)
    vq_vae.load_state_dict(st)
    vq_vae = vq_vae.eval().to(device)

    train_ds = DCASE2020Task2LogMelDataset(
        root=args.data_path,
        machine_type=args.machine_type,
        include_test=False,
    )
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Infer map sizes by a single encode.
    x0, _, _ = next(iter(DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=0)))
    x0 = x0.to(device)
    with torch.inference_mode():
        idx_c0, idx_f0 = vq_vae.encode_to_indices(x0)
    Hc, Wc = int(idx_c0.shape[1]), int(idx_c0.shape[2])
    Hf, Wf = int(idx_f0.shape[1]), int(idx_f0.shape[2])

    coarse_cfg = JSCCMapConfig(
        num_embeddings=int(ckpt["num_embeddings_coarse"]),
        embedding_dim=int(ckpt["embedding_dim_coarse"]),
        H=Hc,
        W=Wc,
        alpha=int(args.alpha_c),
    )
    fine_cfg = JSCCMapConfig(
        num_embeddings=int(ckpt["num_embeddings_fine"]),
        embedding_dim=int(ckpt["embedding_dim_fine"]),
        H=Hf,
        W=Wf,
        alpha=int(args.alpha_f),
    )
    jscc = JSCCDualMap(coarse=coarse_cfg, fine=fine_cfg).to(device)
    opt = torch.optim.Adam(jscc.parameters(), lr=args.lr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(1, args.epochs + 1):
        jscc.train()
        total = 0.0
        n = 0
        for x, _lbl, _mid in loader:
            x = x.to(device)
            with torch.no_grad():
                idx_c, idx_f = vq_vae.encode_to_indices(x)
                q_fine_t, q_coarse_t = vq_vae.indices_to_quantized(idx_c, idx_f)

            snr_db = (torch.rand(x.shape[0], device=device) * (args.snr_max - args.snr_min) + args.snr_min).float()
            q_coarse_hat, q_fine_hat = jscc(idx_c, idx_f, snr_db=snr_db)

            loss = F.mse_loss(q_coarse_hat, q_coarse_t) + F.mse_loss(q_fine_hat, q_fine_t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item()) * x.shape[0]
            n += int(x.shape[0])

        avg = total / max(1, n)
        print(f"epoch {ep}/{args.epochs} | mse={avg:.6f} | cu/clip={jscc.channel_uses_per_clip}")

        torch.save(
            {
                "epoch": ep,
                "jscc_state_dict": jscc.state_dict(),
                "coarse_cfg": coarse_cfg.__dict__,
                "fine_cfg": fine_cfg.__dict__,
            },
            out_path,
        )

    print(f"Saved JSCC checkpoint: {out_path}")


if __name__ == "__main__":
    main()

