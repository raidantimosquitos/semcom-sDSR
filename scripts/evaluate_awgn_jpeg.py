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
from src.engine.evaluator import AnomalyEvaluator
from src.models.vq_vae.autoencoders import VQ_VAE_2Layer
from src.models.sDSR.s_dsr import sDSR, sDSRConfig
from src.utils.checkpoint_compat import migrate_vq_vae_state_dict
from src.utils.stage1_norm import load_norm_from_stage1_ckpt

from src.comm.bytes_bits import bytes_to_bits_lsb_first, bits_to_bytes_lsb_first
from src.comm.framing_crc import packetize, depacketize, transmit_frames_over_channel
from src.comm.ldpc_pyldpc import LDPCConfig, make_ldpc_code, ldpc_qpsk_awgn_roundtrip


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
    p.add_argument("--snr_db", type=float, nargs="+", default=[-5, 0, 5, 10, 15, 20])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--use_channel", action="store_true", help="If set, transmit JPEG bytes through LDPC+QPSK+AWGN.")
    p.add_argument("--frame_bytes", type=int, default=256, help="Payload bytes per CRC-protected frame (when --use_channel).")
    p.add_argument("--conceal", type=str, choices=["zeros"], default="zeros", help="Concealment on CRC fail (currently only zeros).")
    p.add_argument("--ldpc_n", type=int, default=512)
    p.add_argument("--ldpc_dv", type=int, default=2)
    p.add_argument("--ldpc_dc", type=int, default=4)
    p.add_argument("--ldpc_maxiter", type=int, default=100)
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
    def __init__(self, model: sDSR, *, device: torch.device, quality: int, use_channel: bool, snr_db: float, seed: int, ldpc: LDPCConfig, frame_bytes: int, conceal: str) -> None:
        super().__init__()
        self.model = model
        self.device = device
        self.quality = int(quality)
        self.use_channel = bool(use_channel)
        self.snr_db = float(snr_db)
        self.seed = int(seed)
        self.ldpc_cfg = ldpc
        self.ldpc_code = make_ldpc_code(ldpc)
        self.frame_bytes = int(frame_bytes)
        self.conceal = str(conceal)
        self._cu_total = 0
        self._n_total = 0
        self._frames_total = 0
        self._frames_failed = 0

    def avg_channel_uses_per_clip(self) -> float:
        return float(self._cu_total / self._n_total) if self._n_total else 0.0

    def frame_failure_stats(self) -> tuple[int, int, float]:
        n = int(self._frames_total)
        f = int(self._frames_failed)
        return n, f, (f / n) if n else 0.0

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
                frames = packetize(blob, self.frame_bytes)

                def txrx(fr: bytes) -> bytes:
                    bits = bytes_to_bits_lsb_first(fr)
                    rx_bits = ldpc_qpsk_awgn_roundtrip(
                        bits,
                        code=self.ldpc_code,
                        ldpc_maxiter=self.ldpc_cfg.maxiter,
                        ebn0_db=self.snr_db,
                        rng=rng,
                    )
                    # channel uses (QPSK): blocks*n /2
                    k, n0 = int(self.ldpc_code.k), int(self.ldpc_code.n)
                    blocks = (int(bits.size) + k - 1) // k
                    self._cu_total += (blocks * n0 + 1) // 2
                    return bits_to_bytes_lsb_first(rx_bits)

                rx_payloads, st = transmit_frames_over_channel(
                    frames, frame_payload_bytes=self.frame_bytes, channel_txrx=txrx, concealment=self.conceal
                )
                self._frames_total += int(st.n_frames)
                self._frames_failed += int(st.n_failed)
                self._n_total += 1
                blob = depacketize(rx_payloads, orig_len=len(blob))

            # Decode JPEG; if it fails (likely due to erasures), fall back to all-zeros spectrogram.
            try:
                spec = _jpeg_decode(blob, n_mels=n_mels, T=T)
            except Exception:
                spec = torch.zeros((1, n_mels, T), dtype=torch.float32)
            out_specs.append(spec)
        x_hat = torch.stack(out_specs, dim=0).to(self.device)
        return self.model(x_hat)


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

    ldpc_cfg = LDPCConfig(n=args.ldpc_n, d_v=args.ldpc_dv, d_c=args.ldpc_dc, maxiter=args.ldpc_maxiter, seed=0)

    out_path = Path(args.output) if args.output else (Path(args.stage2_ckpt).resolve().parent / "results" / "awgn_jpeg_results.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["machine_type", "method", "quality", "use_channel", "snr_db", "seed", "avg_cu_total", "machine_id", "auc", "pauc"])
        for snr_db in args.snr_db:
            for seed in args.seeds:
                wrapper = JPEGBaselineWrapper(
                    model=model,
                    device=device,
                    quality=args.quality,
                    use_channel=args.use_channel,
                    snr_db=float(snr_db),
                    seed=seed,
                    ldpc=ldpc_cfg,
                    frame_bytes=args.frame_bytes,
                    conceal=args.conceal,
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
                nfr, nff, fer = wrapper.frame_failure_stats() if args.use_channel else (0, 0, 0.0)
                for mid, v in ids.items():
                    if not isinstance(v, dict):
                        continue
                    w.writerow([args.machine_type, "jpeg", args.quality, int(args.use_channel), snr_db, seed, f"{cu:.2f}", mid, v["auc"], v["pauc"]])
                f.flush()
                print(f"[{args.machine_type}] jpeg Q={args.quality} use_channel={args.use_channel} snr={snr_db} seed={seed} frames={nfr} failed={nff} fer={fer:.3f} avg={ids.get('average')}")

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

