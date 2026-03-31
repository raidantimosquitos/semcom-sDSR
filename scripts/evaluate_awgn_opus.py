#!/usr/bin/env python3
"""
OPUS waveform baseline:
- Load raw wav from DCASE test set
- Encode to Opus (Ogg Opus) bytes via ffmpeg
- (Optional) transmit bytes through LDPC+QPSK+AWGN and decode
- Decode Opus back to wav via ffmpeg
- Recompute log-mel spectrogram and standardize/crop/pad to match dataset
- Run sDSR Stage2 evaluation (AUC/pAUC@0.1)

Requires:
- ffmpeg on PATH with libopus enabled
- torchaudio
- numpy, pyldpc (if --use_channel)
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
import torch.nn as nn

from src.data.dataset import MEL_TIME_CROP, DCASE2020Task2LogMelDataset, DCASE2020Task2TestDataset
from src.engine.evaluator import AnomalyEvaluator
from src.models.vq_vae.autoencoders import VQ_VAE_2Layer
from src.models.sDSR.s_dsr import sDSR, sDSRConfig
from src.utils.checkpoint_compat import migrate_vq_vae_state_dict
from src.utils.stage1_norm import load_norm_from_stage1_ckpt
from src.utils.audio import standardize_spectrogram

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
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--pauc_max_fpr", type=float, default=0.1)
    p.add_argument("--opus_kbps", type=int, default=12)
    p.add_argument("--snr_db", type=float, nargs="+", default=[-5, 0, 5, 10, 15, 20])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--use_channel", action="store_true")
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


def opus_encode_bytes_ffmpeg(wav: torch.Tensor, sr: int, kbps: int) -> bytes:
    """Encode wav -> Ogg Opus bytes using ffmpeg."""
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        in_wav = td / "in.wav"
        out_opus = td / "out.ogg"
        torchaudio.save(str(in_wav), wav.cpu(), sample_rate=sr)
        enc = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(in_wav),
                "-c:a",
                "libopus",
                "-b:a",
                f"{int(kbps)}k",
                str(out_opus),
            ],
            capture_output=True,
        )
        if enc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg opus encode failed: {enc.stderr.decode(errors='ignore')}"
            )
        return out_opus.read_bytes()


def opus_decode_bytes_ffmpeg(opus_bytes: bytes) -> tuple[torch.Tensor, int]:
    """Decode Ogg Opus bytes -> wav, sr using ffmpeg."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        in_opus = td / "in.ogg"
        out_wav = td / "out.wav"
        in_opus.write_bytes(opus_bytes)
        dec = subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(in_opus), str(out_wav)],
            capture_output=True,
        )
        if dec.returncode != 0:
            raise RuntimeError(
                f"ffmpeg opus decode failed: {dec.stderr.decode(errors='ignore')}"
            )
        wav_d, sr_d = torchaudio.load(str(out_wav))
        return wav_d, int(sr_d)


def wav_to_logmel(
    wav: torch.Tensor,
    sr: int,
    *,
    sample_rate: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 512,
    n_mels: int = 128,
    top_db: float = 80.0,
    target_T: int | None = None,
) -> torch.Tensor:
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=0.0,
        f_max=8000.0,
    )
    to_db = T.AmplitudeToDB(top_db=top_db)
    mel = mel_transform(wav)
    log_mel = to_db(mel).float()  # (1,n_mels,T)
    x = standardize_spectrogram(log_mel)
    x = x[..., :MEL_TIME_CROP]
    if target_T is not None and x.shape[-1] < target_T:
        x = F.pad(x, (0, target_T - x.shape[-1]), mode="constant", value=0.0)
    return x


class OpusBaselineWrapper(nn.Module):
    def __init__(self, model: sDSR, *, device: torch.device, kbps: int, use_channel: bool, snr_db: float, seed: int, ldpc: LDPCConfig, data_root: Path, machine_type: str, target_T: int) -> None:
        super().__init__()
        self.model = model
        self.device = device
        self.kbps = int(kbps)
        self.use_channel = bool(use_channel)
        self.snr_db = float(snr_db)
        self.seed = int(seed)
        self.ldpc_cfg = ldpc
        self.ldpc_code = make_ldpc_code(ldpc)
        self.data_root = Path(data_root)
        self.machine_type = machine_type
        self.target_T = int(target_T)
        self._cu_total = 0
        self._n_total = 0

    def avg_channel_uses_per_clip(self) -> float:
        return float(self._cu_total / self._n_total) if self._n_total else 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE: evaluator passes in spectrogram x, but for OPUS we need raw wav.
        # We therefore ignore x and re-load wavs from the underlying test dataset order.
        raise RuntimeError("OpusBaselineWrapper must be used with OpusTestDataset, not raw tensors.")


class OpusTestDataset(torch.utils.data.Dataset):
    """
    Produces (spectrogram_after_opus, label, machine_id) in the same order as DCASE2020Task2TestDataset,
    but with an OPUS encode/decode (and optional channel) applied to the waveform before feature extraction.
    """

    def __init__(self, base: DCASE2020Task2TestDataset, *, kbps: int, use_channel: bool, snr_db: float, seed: int, ldpc: LDPCConfig, target_T: int, frame_bytes: int, conceal: str) -> None:
        self.base = base
        self.kbps = int(kbps)
        self.use_channel = bool(use_channel)
        self.snr_db = float(snr_db)
        self.seed = int(seed)
        self.ldpc_cfg = ldpc
        self.ldpc_code = make_ldpc_code(ldpc)
        self.target_T = int(target_T)
        self.frame_bytes = int(frame_bytes)
        self.conceal = str(conceal)
        self._rng = np.random.default_rng(self.seed)
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

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        wav_path, label, machine_id = self.base.samples[idx]  # type: ignore[attr-defined]
        wav, sr = torchaudio.load(wav_path)
        opus_blob = opus_encode_bytes_ffmpeg(wav, sr, kbps=self.kbps)

        if self.use_channel:
            frames = packetize(opus_blob, self.frame_bytes)

            def txrx(fr: bytes) -> bytes:
                bits = bytes_to_bits_lsb_first(fr)
                rx_bits = ldpc_qpsk_awgn_roundtrip(
                    bits,
                    code=self.ldpc_code,
                    ldpc_maxiter=self.ldpc_cfg.maxiter,
                    ebn0_db=self.snr_db,
                    rng=self._rng,
                )
                # channel uses (QPSK): blocks*n / 2
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
            opus_blob = depacketize(rx_payloads, orig_len=len(opus_blob))

        try:
            wav_d, sr_d = opus_decode_bytes_ffmpeg(opus_blob)
        except Exception:
            # If decoding fails due to erasures, fall back to silence.
            wav_d, sr_d = torch.zeros((1, int(sr)), dtype=torch.float32), int(sr)
        x = wav_to_logmel(wav_d, sr_d, target_T=self.target_T)
        return x, label, machine_id


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

    out_path = Path(args.output) if args.output else (Path(args.stage2_ckpt).resolve().parent / "results" / "opus_results.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["machine_type", "method", "opus_kbps", "use_channel", "snr_db", "seed", "avg_cu_total", "machine_id", "auc", "pauc"])
        for snr_db in args.snr_db:
            for seed in args.seeds:
                ds = OpusTestDataset(
                    test_ds,
                    kbps=args.opus_kbps,
                    use_channel=args.use_channel,
                    snr_db=float(snr_db),
                    seed=seed,
                    ldpc=ldpc_cfg,
                    target_T=train_ds.target_T,
                    frame_bytes=args.frame_bytes,
                    conceal=args.conceal,
                )
                evaluator = AnomalyEvaluator(
                    model=model,
                    test_dataset=ds,
                    device=device,
                    pauc_max_fpr=args.pauc_max_fpr,
                    batch_size=args.batch_size,
                )
                res = evaluator.evaluate()
                ids = res.get(args.machine_type, {})
                cu = ds.avg_channel_uses_per_clip() if args.use_channel else 0.0
                nfr, nff, fer = ds.frame_failure_stats() if args.use_channel else (0, 0, 0.0)
                for mid, v in ids.items():
                    if not isinstance(v, dict):
                        continue
                    w.writerow([args.machine_type, "opus", args.opus_kbps, int(args.use_channel), snr_db, seed, f"{cu:.2f}", mid, v["auc"], v["pauc"]])
                f.flush()
                print(
                    f"[{args.machine_type}] opus {args.opus_kbps}kbps use_channel={args.use_channel} "
                    f"snr={snr_db} seed={seed} frames={nfr} failed={nff} fer={fer:.3f} cu_total={cu:.1f}/clip avg={ids.get('average')}"
                )

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

