#!/usr/bin/env python3
"""
Encode DCASE2020 Task 2 test data to codebook indices and write one bitstream file
per test clip for GNURadio (FEC + modulation). Filenames match the test wav stems:
  {normal|anomaly}_id_{XX}_{number}.bin  (same as dataset layout under machine_type/test/).

Usage:
  python scripts/transmit_indices.py --stage1_ckpt checkpoints/stage1/fan/best.pt \\
    --data_path /path/to/dcase --machine_type fan --output_dir tx_bitstreams/fan \\
    [--output_meta tx_meta.csv]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import math
import torch

from src.data.dataset import (
    DCASE2020Task2LogMelDataset,
    DCASE2020Task2TestDataset,
)
from src.models.vq_vae.autoencoders import VQ_VAE_2Layer
from src.utils.stage1_norm import load_norm_from_stage1_ckpt
from src.utils.bitstream import (
    frame_size_bytes,
    pack_indices_to_frame,
    write_frame_file,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Encode test set to indices and write bitstream for GNURadio.")
    p.add_argument("--stage1_ckpt", type=str, required=True, help="Stage 1 checkpoint (encoder + codebooks)")
    p.add_argument("--data_path", type=str, required=True, help="Path to DCASE dataset root")
    p.add_argument("--machine_type", type=str, default="fan")
    p.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write one .bin per test clip (names match *.wav stems, see DCASE2020Task2TestDataset)",
    )
    p.add_argument("--output_meta", type=str, default=None, help="Optional CSV: clip_index, label, machine_id")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load Stage 1 checkpoint for norm stats and model weights
    ckpt = torch.load(args.stage1_ckpt, map_location="cpu", weights_only=True)
    norm_mean, norm_std = load_norm_from_stage1_ckpt(ckpt)
    train_ds = DCASE2020Task2LogMelDataset(
        root=args.data_path,
        machine_type=args.machine_type,
    )
    _, _, n_mels, T = train_ds.data.shape
    test_ds = DCASE2020Task2TestDataset(
        root=args.data_path,
        machine_type=args.machine_type,
        target_T=train_ds.target_T,
    )

    # Stage 1: VQ-VAE only (encoder + codebooks); load arch from checkpoint
    from src.utils.checkpoint_compat import migrate_vq_vae_state_dict
    num_embeddings_coarse = ckpt.get("num_embeddings_coarse", ckpt.get("num_embeddings_top"))
    num_embeddings_fine = ckpt.get("num_embeddings_fine", ckpt.get("num_embeddings_bot"))
    embedding_dim = ckpt["embedding_dim"]
    hidden_channels = ckpt["hidden_channels"]
    num_residual_layers = ckpt.get("num_residual_layers", 2)
    vq_vae = VQ_VAE_2Layer(
        hidden_channels=hidden_channels,
        num_residual_layers=num_residual_layers,
        num_embeddings=(num_embeddings_coarse, num_embeddings_fine),
        embedding_dim=embedding_dim,
        commitment_cost=0.25,
        decay=0.99,
    )
    state = dict(ckpt["model_state_dict"])
    migrate_vq_vae_state_dict(state)
    vq_vae.load_state_dict(state)
    vq_vae = vq_vae.to(device)
    vq_vae.eval()

    # Fine x2/x4 down (n_mels/2, T/4), coarse x8/x16 down (n_mels/8, T/16)
    bits_coarse = int(math.log2(num_embeddings_coarse))
    bits_fine = int(math.log2(num_embeddings_fine))
    H_fine = max(1, n_mels // 2)
    W_fine = max(1, T // 4)
    H_coarse = max(1, n_mels // 8)
    W_coarse = max(1, T // 16)
    frame_sz = frame_size_bytes(H_coarse, W_coarse, H_fine, W_fine, bits_coarse, bits_fine)

    meta_rows: list[tuple[int, int, str]] = []

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for idx in range(len(test_ds)):
            spec, label, machine_id = test_ds[idx]
            wav_path = test_ds.samples[idx][0]
            stem = Path(wav_path).stem
            bin_name = f"{stem}.bin"
            x = spec.unsqueeze(0).to(device)  # (1, 1, n_mels, T)
            indices_coarse, indices_fine = vq_vae.encode_to_indices(x)
            frame = pack_indices_to_frame(
                indices_coarse, indices_fine,
                bits_coarse=bits_coarse, bits_fine=bits_fine,
            )
            # Raw bytes only (one file == one clip).
            write_frame_file(str(out_dir / bin_name), frame)
            meta_rows.append((idx, label, machine_id))

    print(f"Wrote {len(meta_rows)} bitstream files ({frame_sz} bytes payload frame each) under {out_dir}")

    if args.output_meta:
        meta_path = Path(args.output_meta)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["clip_index", "label", "machine_id"])
            for row in meta_rows:
                w.writerow(row)
        print(f"Wrote metadata to {meta_path}")


if __name__ == "__main__":
    main()
