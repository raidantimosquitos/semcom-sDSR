#!/usr/bin/env python3
"""
Compute mean reconstruction MSE of a Stage-1 VQ-VAE checkpoint on the full DCASE test set.

Uses the same test pipeline as DCASE2020Task2TestDataset (crop, pad, global mel norm from checkpoint).
"""

from __future__ import annotations

import argparse
import json
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import DCASE2020Task2LogMelDataset, DCASE2020Task2TestDataset
from src.models.vq_vae.autoencoders import VQ_VAE_2Layer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage-1 test-set reconstruction MSE")
    p.add_argument("--stage1_ckpt", type=str, required=True)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument(
        "--machine_types",
        type=str,
        nargs="+",
        default=["fan", "pump", "slider", "valve", "ToyCar", "ToyConveyor"],
        help="Must match Stage-1 training (order-independent; same sorted run name as checkpoint dir)",
    )
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--json_out", type=str, default=None, help="If set, write one JSON object with avg_mse and metadata")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.stage1_ckpt, map_location="cpu", weights_only=True)
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
    _, _, n_mels, T = train_ds.data.shape
    num_embeddings_coarse = ckpt["num_embeddings_coarse"]
    num_embeddings_fine = ckpt["num_embeddings_fine"]
    embedding_dim_coarse = ckpt["embedding_dim_coarse"]
    embedding_dim_fine = ckpt["embedding_dim_fine"]
    hidden_channels_coarse = ckpt["hidden_channels_coarse"]
    hidden_channels_fine = ckpt["hidden_channels_fine"]
    num_residual_layers = ckpt["num_residual_layers"]

    model = VQ_VAE_2Layer(
        hidden_channels=(hidden_channels_coarse, hidden_channels_fine),
        num_residual_layers=num_residual_layers,
        num_embeddings=(num_embeddings_coarse, num_embeddings_fine),
        embedding_dim=(embedding_dim_coarse, embedding_dim_fine),
        commitment_cost=0.25,
        decay=0.99,
    )
    state = dict(ckpt["model_state_dict"])
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    total_se = 0.0
    total_el = 0
    n_samples = 0

    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device, non_blocking=True)
            out = model(x)
            recon = out[2]
            diff = (recon - x).float()
            total_se += (diff * diff).sum().item()
            total_el += x.numel()
            n_samples += x.shape[0]

    avg_mse = total_se / max(total_el, 1)

    line = (
        f"avg_mse={avg_mse:.8f}  n_test={n_samples}  ckpt={args.stage1_ckpt}  "
        f"K={num_embeddings_coarse},{num_embeddings_fine}  "
        f"emb_dim={embedding_dim_coarse},{embedding_dim_fine}"
    )
    print(line)

    if args.json_out:
        payload = {
            "avg_mse": avg_mse,
            "n_test_samples": n_samples,
            "stage1_ckpt": args.stage1_ckpt,
            "num_embeddings_coarse": int(num_embeddings_coarse),
            "num_embeddings_fine": int(num_embeddings_fine),
            "embedding_dim_coarse": int(embedding_dim_coarse),
            "embedding_dim_fine": int(embedding_dim_fine),
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    sys.stdout.flush()


if __name__ == "__main__":
    main()
