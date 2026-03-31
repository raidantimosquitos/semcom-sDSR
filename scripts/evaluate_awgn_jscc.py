#!/usr/bin/env python3
"""
Evaluate sDSR under AWGN using a trained JSCC model that transmits index maps.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn

from src.data.dataset import DCASE2020Task2LogMelDataset, DCASE2020Task2TestDataset
from src.engine.evaluator import AnomalyEvaluator
from src.models.vq_vae.autoencoders import VQ_VAE_2Layer
from src.models.sDSR.s_dsr import sDSR, sDSRConfig
from src.utils.checkpoint_compat import migrate_vq_vae_state_dict
from src.utils.stage1_norm import load_norm_from_stage1_ckpt

from src.comm.jscc_cnn import JSCCDualMap, JSCCMapConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--stage1_ckpt", type=str, required=True)
    p.add_argument("--stage2_ckpt", type=str, required=True)
    p.add_argument("--jscc_ckpt", type=str, required=True)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--machine_type", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--pauc_max_fpr", type=float, default=0.1)
    p.add_argument("--snr_db", type=float, nargs="+", default=[-5, 0, 5, 10, 15, 20])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--output", type=str, default=None)
    return p.parse_args()


def build_s_dsr(n_mels: int, T: int, vq_vae: VQ_VAE_2Layer, embedding_dim: tuple[int, int], hidden_channels: tuple[int, int], num_residual_layers: int) -> sDSR:
    cfg = sDSRConfig(
        embedding_dim=embedding_dim,
        hidden_channels=hidden_channels,
        num_residual_layers=num_residual_layers,
        n_mels=n_mels,
        T=T,
    )
    return sDSR(vq_vae, cfg)


class JSCCWrapper(nn.Module):
    def __init__(self, model: sDSR, vq_vae: VQ_VAE_2Layer, jscc: JSCCDualMap, *, device: torch.device, snr_db: float, seed: int) -> None:
        super().__init__()
        self.model = model
        self.vq_vae = vq_vae
        self.jscc = jscc
        self.device = device
        self.snr_db = float(snr_db)
        self.seed = int(seed)

    @property
    def channel_uses_per_clip(self) -> int:
        return int(self.jscc.channel_uses_per_clip)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        self.vq_vae.eval()
        self.jscc.eval()
        with torch.inference_mode():
            idx_c, idx_f = self.vq_vae.encode_to_indices(x.to(self.device))
            snr = torch.full((x.shape[0],), self.snr_db, device=self.device, dtype=torch.float32)
            q_coarse_hat, q_fine_hat = self.jscc(idx_c, idx_f, snr_db=snr)
            # JSCC returns q tensors directly (B,C,H,W)
            return self.model.forward_from_quantized(q_fine=q_fine_hat, q_coarse=q_coarse_hat)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    stage1_ckpt = torch.load(args.stage1_ckpt, map_location="cpu", weights_only=True)
    _norm_mean, _norm_std = load_norm_from_stage1_ckpt(stage1_ckpt)

    train_ds = DCASE2020Task2LogMelDataset(root=args.data_path, machine_type=args.machine_type, include_test=False)
    test_ds = DCASE2020Task2TestDataset(root=args.data_path, machine_type=args.machine_type, target_T=train_ds.target_T)
    _, _, n_mels, T = train_ds.data.shape

    vq_vae = VQ_VAE_2Layer(
        hidden_channels=(stage1_ckpt["hidden_channels_coarse"], stage1_ckpt["hidden_channels_fine"]),
        num_residual_layers=stage1_ckpt["num_residual_layers"],
        num_embeddings=(stage1_ckpt["num_embeddings_coarse"], stage1_ckpt["num_embeddings_fine"]),
        embedding_dim=(stage1_ckpt["embedding_dim_coarse"], stage1_ckpt["embedding_dim_fine"]),
        commitment_cost=0.25,
        decay=0.99,
    )
    st1 = dict(stage1_ckpt["model_state_dict"])
    migrate_vq_vae_state_dict(st1)
    vq_vae.load_state_dict(st1)

    model = build_s_dsr(
        n_mels,
        T,
        vq_vae=vq_vae,
        embedding_dim=(stage1_ckpt["embedding_dim_coarse"], stage1_ckpt["embedding_dim_fine"]),
        hidden_channels=(stage1_ckpt["hidden_channels_coarse"], stage1_ckpt["hidden_channels_fine"]),
        num_residual_layers=stage1_ckpt["num_residual_layers"],
    )
    stage2 = torch.load(args.stage2_ckpt, map_location="cpu", weights_only=True)
    st2 = dict(stage2["model_state_dict"])
    migrate_vq_vae_state_dict(st2)
    model.load_state_dict(st2)
    model = model.to(device)
    vq_vae = vq_vae.to(device)

    jscc_ckpt = torch.load(args.jscc_ckpt, map_location="cpu", weights_only=True)
    coarse_cfg = JSCCMapConfig(**jscc_ckpt["coarse_cfg"])
    fine_cfg = JSCCMapConfig(**jscc_ckpt["fine_cfg"])
    jscc = JSCCDualMap(coarse=coarse_cfg, fine=fine_cfg).to(device)
    jscc.load_state_dict(jscc_ckpt["jscc_state_dict"])

    out_path = Path(args.output) if args.output else (Path(args.stage2_ckpt).resolve().parent / "results" / "awgn_jscc_results.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["machine_type", "method", "jscc_ckpt", "snr_db", "seed", "cu_total", "machine_id", "auc", "pauc"])
        for snr_db in args.snr_db:
            for seed in args.seeds:
                wrapper = JSCCWrapper(model=model, vq_vae=vq_vae, jscc=jscc, device=device, snr_db=float(snr_db), seed=seed)
                evaluator = AnomalyEvaluator(
                    model=wrapper,
                    test_dataset=test_ds,
                    device=device,
                    pauc_max_fpr=args.pauc_max_fpr,
                    batch_size=args.batch_size,
                )
                res = evaluator.evaluate()
                ids = res.get(args.machine_type, {})
                for mid, v in ids.items():
                    if not isinstance(v, dict):
                        continue
                    w.writerow([args.machine_type, "jscc", args.jscc_ckpt, snr_db, seed, wrapper.channel_uses_per_clip, mid, v["auc"], v["pauc"]])
                f.flush()
                print(f"[{args.machine_type}] jscc cu_total={wrapper.channel_uses_per_clip}/clip snr={snr_db} seed={seed} avg={ids.get('average')}")

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

