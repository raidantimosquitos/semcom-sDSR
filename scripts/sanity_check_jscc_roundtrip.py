#!/usr/bin/env python3
"""
Sanity check for JSCC dual-map roundtrip quality.

Compares:
  1) "Oracle" quantized tensors from VQ-VAE indices (no channel)
  2) JSCC reconstructed quantized tensors under AWGN at chosen SNR

This isolates whether poor downstream AUC is due to:
  - wiring / shape / model mismatch, vs
  - JSCC reconstruction being too distorted even at high SNR.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Ensure `import src.*` works when running as a script.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.dataset import DCASE2020Task2LogMelDataset
from src.models.vq_vae.autoencoders import VQ_VAE_2Layer
from src.comm.jscc_cnn import JSCCDualMap, JSCCMapConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--stage1_ckpt", type=str, required=True)
    p.add_argument("--jscc_ckpt", type=str, required=True)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--machine_type", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--snr_db", type=float, default=60.0, help="Use very high SNR to approximate noiseless channel.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    stage1 = torch.load(args.stage1_ckpt, map_location="cpu", weights_only=True)
    vq_vae = VQ_VAE_2Layer(
        hidden_channels=(stage1["hidden_channels_coarse"], stage1["hidden_channels_fine"]),
        num_residual_layers=stage1["num_residual_layers"],
        num_embeddings=(stage1["num_embeddings_coarse"], stage1["num_embeddings_fine"]),
        embedding_dim=(stage1["embedding_dim_coarse"], stage1["embedding_dim_fine"]),
        commitment_cost=0.25,
        decay=0.99,
    )
    st = dict(stage1["model_state_dict"])
    vq_vae.load_state_dict(st)
    vq_vae = vq_vae.eval().to(device)

    jscc_ckpt = torch.load(args.jscc_ckpt, map_location="cpu", weights_only=True)
    coarse_cfg = JSCCMapConfig(**jscc_ckpt["coarse_cfg"])
    fine_cfg = JSCCMapConfig(**jscc_ckpt["fine_cfg"])
    jscc = JSCCDualMap(coarse=coarse_cfg, fine=fine_cfg).eval().to(device)
    jscc.load_state_dict(jscc_ckpt["jscc_state_dict"])

    # Normalization is disabled project-wide: always use raw log-mel dB.
    use_norm = False
    norm_mean, norm_std = None, None

    ds = DCASE2020Task2LogMelDataset(
        root=args.data_path,
        machine_type=args.machine_type,
        include_test=False,
    )
    x, _lbl, _mid = next(iter(torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)))
    x = x.to(device)

    with torch.inference_mode():
        idx_c, idx_f = vq_vae.encode_to_indices(x)
        q_f_oracle, q_c_oracle = vq_vae.indices_to_quantized(idx_c, idx_f)
        snr = torch.full((x.shape[0],), float(args.snr_db), device=device, dtype=torch.float32)
        q_c_hat, q_f_hat = jscc(idx_c, idx_f, snr_db=snr)

        mse_c = F.mse_loss(q_c_hat, q_c_oracle).item()
        mse_f = F.mse_loss(q_f_hat, q_f_oracle).item()
        rel_c = (q_c_hat - q_c_oracle).pow(2).mean().sqrt() / (q_c_oracle.pow(2).mean().sqrt() + 1e-8)
        rel_f = (q_f_hat - q_f_oracle).pow(2).mean().sqrt() / (q_f_oracle.pow(2).mean().sqrt() + 1e-8)

    print(f"[jscc] channel_uses_per_clip={int(jscc.channel_uses_per_clip)}")
    print(f"[snr_db={args.snr_db}] MSE coarse={mse_c:.6g} fine={mse_f:.6g}")
    print(f"[snr_db={args.snr_db}] RelRMSE coarse={float(rel_c):.6g} fine={float(rel_f):.6g}")
    print(f"Shapes: idx_c={tuple(idx_c.shape)} idx_f={tuple(idx_f.shape)} q_c={tuple(q_c_oracle.shape)} q_f={tuple(q_f_oracle.shape)}")


if __name__ == "__main__":
    main()

