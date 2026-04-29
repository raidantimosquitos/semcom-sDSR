#!/usr/bin/env python3
"""
JPEG spectrogram baseline:
- Take standardized log-mel spectrogram (as produced by dataset)
- Encode to JPEG bytes at a given quality
- (Optional) transmit bytes through LDPC+QPSK+AWGN and decode
- Decode JPEG back to spectrogram tensor
- Run sDSR Stage2 evaluation (AUC/pAUC@0.1)
"""

from __future__ import annotations

import argparse
import csv
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.data.dataset import DCASE2020Task2LogMelDataset, DCASE2020Task2TestDataset
from src.utils.audio import mel_norm_from_stage1_ckpt
from src.engine.evaluator import AnomalyEvaluator
from src.models.vq_vae.autoencoders import VQ_VAE_2Layer
from src.models.sDSR.s_dsr import sDSR, sDSRConfig

from src.comm.bitflip_ber import load_ber_curve_csv, bitflip_bytes
from src.comm.jpeg_payload import bitflip_jpeg_entropy_payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--stage1_ckpt", type=str, required=True)
    p.add_argument("--stage2_ckpt", type=str, required=True)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--machine_type", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--pauc_max_fpr", type=float, default=0.1)
    p.add_argument("--quality", type=int, default=50)
    p.add_argument("--snr_db", type=float, nargs="+", default=[0, 2, 4, 6, 8, 10, 12, 15, 20])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--use_channel", action="store_true", help="If set, corrupt JPEG bytes using BER(SNR) bit flips.")
    p.add_argument("--ber_curve", type=str, default=None, help="CSV with columns snr_db, ber_postfec (from calibrate_ldpc_bpsk_ber.py). Required when --use_channel.")
    p.add_argument("--channel_mode", type=str, choices=["jpeg_entropy", "prefix"], default="jpeg_entropy", help="Channel corruption mode: JPEG entropy-only payload masking (default) or legacy prefix protection.")
    p.add_argument("--protect_bytes", type=int, default=8, help="Protected prefix bytes. For channel_mode=jpeg_entropy this protects custom outer header bytes; for prefix it protects leading bytes only.")
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


def _jpeg_encode(spec: torch.Tensor, quality: int) -> bytes:
    # spec: (1, n_mels, T) float
    try:
        from PIL import Image
    except Exception as e:  # pragma: no cover
        raise ImportError("Pillow required: pip install pillow") from e

    x = spec.squeeze(0).detach().cpu().float().numpy()  # (H,W)
    # map to 0..255 using min/max per-clip (simple baseline)
    mn = float(x.min())
    mx = float(x.max())
    denom = (mx - mn) if (mx - mn) > 1e-8 else 1.0
    u8 = ((x - mn) / denom * 255.0).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(u8, mode="L")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=int(quality), optimize=True)
    payload = buf.getvalue()
    # store mn/mx as 2 float32 (8 bytes) header so we can invert scaling
    header = np.array([mn, mx], dtype=np.float32).tobytes()
    return header + payload


def _jpeg_decode(blob: bytes, n_mels: int, T: int) -> torch.Tensor:
    try:
        from PIL import Image
    except Exception as e:  # pragma: no cover
        raise ImportError("Pillow required: pip install pillow") from e

    if len(blob) < 8:
        raise ValueError("Invalid JPEG blob (too small).")
    mn, mx = np.frombuffer(blob[:8], dtype=np.float32).tolist()
    payload = blob[8:]
    img = Image.open(BytesIO(payload)).convert("L")
    arr = np.array(img, dtype=np.float32)
    # resize/crop if needed (should match)
    if arr.shape != (n_mels, T):
        arr = np.array(img.resize((T, n_mels)), dtype=np.float32)
    denom = (mx - mn) if (mx - mn) > 1e-8 else 1.0
    x = arr / 255.0 * denom + mn
    return torch.from_numpy(x).view(1, n_mels, T)


class JPEGBaselineWrapper(nn.Module):
    def __init__(
        self,
        model: sDSR,
        *,
        device: torch.device,
        quality: int,
        use_channel: bool,
        snr_db: float,
        seed: int,
        ber_curve_path: str | None,
        protect_bytes: int,
        channel_mode: str,
    ) -> None:
        super().__init__()
        self.model = model
        self.device = device
        self.quality = int(quality)
        self.use_channel = bool(use_channel)
        self.snr_db = float(snr_db)
        self.seed = int(seed)
        self.ber_curve = load_ber_curve_csv(ber_curve_path) if (use_channel and ber_curve_path) else None
        self.protect_bytes = int(protect_bytes)
        self.channel_mode = str(channel_mode)
        self._cu_total = 0
        self._n_total = 0
        self._decode_ok = 0
        self._decode_fail = 0

    def avg_channel_uses_per_clip(self) -> float:
        return float(self._cu_total / self._n_total) if self._n_total else 0.0

    def decode_success_stats(self) -> tuple[int, int, float]:
        ok = int(self._decode_ok)
        fail = int(self._decode_fail)
        tot = ok + fail
        return ok, fail, (ok / tot) if tot else 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,1,n_mels,T)
        rng = np.random.default_rng(self.seed)
        B = x.shape[0]
        n_mels = int(x.shape[2])
        T = int(x.shape[3])
        out_specs = []
        for i in range(B):
            blob = _jpeg_encode(x[i].detach().cpu(), quality=self.quality)
            if self.use_channel:
                if self.ber_curve is None:
                    raise RuntimeError("--use_channel requires --ber_curve")
                ber = self.ber_curve.ber_at(self.snr_db)
                seed = int(self.seed) ^ int(i) ^ (int(self.snr_db * 100) & 0xFFFF)
                if self.channel_mode == "jpeg_entropy":
                    blob = bitflip_jpeg_entropy_payload(
                        blob,
                        ber=ber,
                        prefix_protect_bytes=self.protect_bytes,
                        seed=seed,
                    )
                else:
                    blob = bitflip_bytes(
                        blob,
                        ber=ber,
                        protect_bytes=self.protect_bytes,
                        seed=seed,
                    )
                # channel uses approximation for LDPC(1/2)+BPSK
                self._cu_total += int(2 * len(blob) * 8)
                self._n_total += 1

            # Decode JPEG; if it fails (likely due to erasures), fall back to all-zeros spectrogram.
            try:
                spec = _jpeg_decode(blob, n_mels=n_mels, T=T)
                self._decode_ok += 1
            except Exception:
                spec = torch.zeros((1, n_mels, T), dtype=torch.float32)
                self._decode_fail += 1
            out_specs.append(spec)
        x_hat = torch.stack(out_specs, dim=0).to(self.device)
        return self.model(x_hat)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    NAN = float("nan")

    stage1_ckpt = torch.load(args.stage1_ckpt, map_location="cpu", weights_only=True)
    mel_mean, mel_std, mel_stats_eps = mel_norm_from_stage1_ckpt(stage1_ckpt)

    train_ds = DCASE2020Task2LogMelDataset(
        root=args.data_path,
        machine_type=args.machine_type,
        include_test=False,
        mel_mean=mel_mean,
        mel_std=mel_std,
        mel_stats_eps=mel_stats_eps,
    )
    test_ds = DCASE2020Task2TestDataset(
        root=args.data_path,
        machine_type=args.machine_type,
        target_T=train_ds.target_T,
        mel_mean=mel_mean,
        mel_std=mel_std,
        mel_stats_eps=mel_stats_eps,
    )
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
    model.load_state_dict(st2)
    model = model.to(device)

    out_path = Path(args.output) if args.output else (Path(args.stage2_ckpt).resolve().parent / "results" / "awgn_jpeg_results.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "machine_type",
                "method",
                "channel",
                "snr_db",
                "seed",
                "quality",
                "opus_kbps",
                "jscc_ckpt",
                "ber_curve",
                "channel_mode",
                "avg_cu_coarse",
                "avg_cu_fine",
                "avg_cu_total",
                "cu_unit",
                "decode_ok",
                "decode_fail",
                "decode_ok_rate",
                "clip_oor_c",
                "clip_oor_f",
                "machine_id",
                "auc",
                "pauc",
            ]
        )
        for snr_db in args.snr_db:
            for seed in args.seeds:
                wrapper = JPEGBaselineWrapper(
                    model=model,
                    device=device,
                    quality=args.quality,
                    use_channel=args.use_channel,
                    snr_db=float(snr_db),
                    seed=seed,
                    ber_curve_path=args.ber_curve,
                    protect_bytes=args.protect_bytes,
                    channel_mode=args.channel_mode,
                )
                evaluator = AnomalyEvaluator(
                    model=wrapper,
                    test_dataset=test_ds,
                    device=device,
                    pauc_max_fpr=args.pauc_max_fpr,
                    batch_size=args.batch_size,
                )
                res = evaluator.evaluate()
                ids = res.get(args.machine_type, {})
                cu = wrapper.avg_channel_uses_per_clip() if args.use_channel else 0.0
                ok, fail, okr = wrapper.decode_success_stats() if args.use_channel else (0, 0, 0.0)
                for mid, v in ids.items():
                    if not isinstance(v, dict):
                        continue
                    w.writerow(
                        [
                            args.machine_type,
                            "jpeg",
                            "ber_bitflip" if args.use_channel else "none",
                            snr_db,
                            seed,
                            args.quality,
                            NAN,
                            NAN,
                            args.ber_curve if args.use_channel else NAN,
                            args.channel_mode if args.use_channel else NAN,
                            NAN,
                            NAN,
                            f"{cu:.2f}",
                            "coded_bits_proxy",
                            ok,
                            fail,
                            f"{okr:.6f}",
                            NAN,
                            NAN,
                            mid,
                            v["auc"],
                            v["pauc"],
                        ]
                    )
                f.flush()
                print(
                    f"[{args.machine_type}] method=jpeg channel={'ber_bitflip' if args.use_channel else 'none'} "
                    f"quality={args.quality} snr={snr_db} seed={seed} "
                    f"cu_total={cu:.1f}/clip decode_ok={ok} decode_fail={fail} ok_rate={okr:.3f} "
                    f"clip_oor_c=nan clip_oor_f=nan avg={ids.get('average')}"
                )

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

