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
import hashlib
import os
import shutil
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
from src.utils.audio import standardize_spectrogram
from src.utils.stage1_norm import load_norm_from_stage1_ckpt

from src.comm.bitflip_ber import load_ber_curve_csv, bitflip_bytes
from src.comm.ogg_payload import bitflip_ogg_payload_pages


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
    p.add_argument("--snr_db", type=float, nargs="+", default=[0, 2, 4, 6, 8, 10, 12, 15, 20])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--use_channel", action="store_true", help="If set, corrupt Opus bytes using BER(SNR) bit flips.")
    p.add_argument("--ber_curve", type=str, default=None, help="CSV with columns snr_db, ber_postfec (from calibrate_ldpc_bpsk_ber.py). Required when --use_channel.")
    p.add_argument("--channel_mode", type=str, choices=["ogg_pages", "prefix"], default="ogg_pages", help="Channel corruption mode: Ogg page-body masking with CRC rewrite (default) or legacy prefix protection.")
    p.add_argument("--protect_pages", type=int, default=2, help="For channel_mode=ogg_pages, protect the first N Ogg pages (typically OpusHead/OpusTags).")
    p.add_argument("--protect_bytes", type=int, default=128, help="For channel_mode=prefix, do not flip bits in the first N bytes.")
    p.add_argument("--ffmpeg_bin", type=str, default=None, help="Path to ffmpeg binary. Resolution order if omitted: $FFMPEG_BIN, /usr/bin/ffmpeg, PATH.")
    p.add_argument("--opus_cache_dir", type=str, default=None, help="If set, cache per-clip Ogg Opus bitstreams here to avoid re-encoding on every SNR/seed pass.")
    p.add_argument("--no_preencode", action="store_true", help="If set, do not pre-encode the full test set into --opus_cache_dir before evaluation.")
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


def resolve_ffmpeg_bin(ffmpeg_bin_arg: str | None) -> str:
    if ffmpeg_bin_arg:
        return str(ffmpeg_bin_arg)
    env_bin = os.environ.get("FFMPEG_BIN")
    if env_bin:
        return env_bin
    if Path("/usr/bin/ffmpeg").exists():
        return "/usr/bin/ffmpeg"
    path_bin = shutil.which("ffmpeg")
    if path_bin:
        return path_bin
    raise FileNotFoundError("ffmpeg not found. Set --ffmpeg_bin or FFMPEG_BIN.")


def opus_encode_bytes_ffmpeg(wav: torch.Tensor, sr: int, kbps: int, *, ffmpeg_bin: str) -> bytes:
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
                ffmpeg_bin,
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


def opus_decode_bytes_ffmpeg(opus_bytes: bytes, *, ffmpeg_bin: str) -> tuple[torch.Tensor, int]:
    """Decode Ogg Opus bytes -> wav, sr using ffmpeg."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        in_opus = td / "in.ogg"
        out_wav = td / "out.wav"
        in_opus.write_bytes(opus_bytes)
        dec = subprocess.run(
            [ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error", "-i", str(in_opus), str(out_wav)],
            capture_output=True,
        )
        if dec.returncode != 0:
            raise RuntimeError(
                f"ffmpeg opus decode failed: {dec.stderr.decode(errors='ignore')}"
            )
        wav_d, sr_d = torchaudio.load(str(out_wav))
        return wav_d, int(sr_d)

def _opus_cache_path(cache_dir: Path, wav_path: str, kbps: int) -> Path:
    h = hashlib.sha1(f"{kbps}|{wav_path}".encode("utf-8", errors="ignore")).hexdigest()
    return cache_dir / f"{h}.ogg"

def preencode_opus_testset(
    *,
    base: DCASE2020Task2TestDataset,
    cache_dir: Path,
    kbps: int,
    ffmpeg_bin: str,
) -> tuple[int, int]:
    """
    Pre-encode all test clips to Ogg Opus and store them in cache_dir.
    Returns (n_present, n_encoded).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    n_present = 0
    n_encoded = 0
    samples = getattr(base, "samples", [])
    for i, s in enumerate(samples):
        wav_path = s[0]
        out_p = _opus_cache_path(cache_dir, str(wav_path), kbps)
        if out_p.exists() and out_p.stat().st_size > 0:
            n_present += 1
            continue
        wav, sr = torchaudio.load(str(wav_path))
        opus_blob = opus_encode_bytes_ffmpeg(wav, sr, kbps=kbps, ffmpeg_bin=ffmpeg_bin)
        tmp = out_p.with_suffix(".tmp")
        tmp.write_bytes(opus_blob)
        tmp.replace(out_p)
        n_encoded += 1
        if (i + 1) % 100 == 0:
            print(f"[opus] pre-encode progress: {i+1}/{len(samples)} (new={n_encoded}, cached={n_present})")
    return n_present, n_encoded


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
    norm_mean: torch.Tensor | None = None,
    norm_std: torch.Tensor | None = None,
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
    log_mel = log_mel[..., :MEL_TIME_CROP]
    x = standardize_spectrogram(log_mel, mean=norm_mean, std=norm_std)
    if target_T is not None and x.shape[-1] < target_T:
        x = F.pad(x, (0, target_T - x.shape[-1]), mode="constant", value=0.0)
    return x


class OpusBaselineWrapper(nn.Module):
    def __init__(self, model: sDSR, *, device: torch.device, kbps: int, use_channel: bool, snr_db: float, seed: int, data_root: Path, machine_type: str, target_T: int) -> None:
        super().__init__()
        self.model = model
        self.device = device
        self.kbps = int(kbps)
        self.use_channel = bool(use_channel)
        self.snr_db = float(snr_db)
        self.seed = int(seed)
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

    def __init__(
        self,
        base: DCASE2020Task2TestDataset,
        *,
        kbps: int,
        use_channel: bool,
        snr_db: float,
        seed: int,
        ber_curve_path: str | None,
        protect_bytes: int,
        protect_pages: int,
        channel_mode: str,
        ffmpeg_bin: str,
        opus_cache_dir: str | None,
        target_T: int,
        norm_mean: torch.Tensor | None = None,
        norm_std: torch.Tensor | None = None,
    ) -> None:
        self.base = base
        # critical: evaluator keys results by dataset.machine_type
        self.machine_type = getattr(base, "machine_type", "unknown")
        self.kbps = int(kbps)
        self.use_channel = bool(use_channel)
        self.snr_db = float(snr_db)
        self.seed = int(seed)
        self.ber_curve = load_ber_curve_csv(ber_curve_path) if (use_channel and ber_curve_path) else None
        self.protect_bytes = int(protect_bytes)
        self.protect_pages = int(protect_pages)
        self.channel_mode = str(channel_mode)
        self.ffmpeg_bin = str(ffmpeg_bin)
        self.opus_cache_dir = Path(opus_cache_dir) if opus_cache_dir else None
        self.target_T = int(target_T)
        self.norm_mean = norm_mean.detach().cpu().clone() if norm_mean is not None else None
        self.norm_std = norm_std.detach().cpu().clone() if norm_std is not None else None
        self._rng = np.random.default_rng(self.seed)
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

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        wav_path, label, machine_id = self.base.samples[idx]  # type: ignore[attr-defined]
        opus_blob: bytes | None = None
        if self.opus_cache_dir is not None:
            p = _opus_cache_path(self.opus_cache_dir, str(wav_path), self.kbps)
            if p.exists() and p.stat().st_size > 0:
                opus_blob = p.read_bytes()

        if opus_blob is None:
            wav, sr = torchaudio.load(wav_path)
            opus_blob = opus_encode_bytes_ffmpeg(wav, sr, kbps=self.kbps, ffmpeg_bin=self.ffmpeg_bin)
            if self.opus_cache_dir is not None:
                self.opus_cache_dir.mkdir(parents=True, exist_ok=True)
                p = _opus_cache_path(self.opus_cache_dir, str(wav_path), self.kbps)
                tmp = p.with_suffix(".tmp")
                tmp.write_bytes(opus_blob)
                tmp.replace(p)
        else:
            # Only load sr for silence fallback sizing. Most DCASE clips are same sr, but we read header cheaply.
            _, sr = torchaudio.load(wav_path, num_frames=1)

        if self.use_channel:
            if self.ber_curve is None:
                raise RuntimeError("--use_channel requires --ber_curve")
            ber = self.ber_curve.ber_at(self.snr_db)
            seed = int(self.seed) ^ int(idx) ^ (int(self.snr_db * 100) & 0xFFFF)
            if self.channel_mode == "ogg_pages":
                opus_blob = bitflip_ogg_payload_pages(
                    opus_blob,
                    ber=ber,
                    protect_first_pages=self.protect_pages,
                    seed=seed,
                )
            else:
                opus_blob = bitflip_bytes(
                    opus_blob,
                    ber=ber,
                    protect_bytes=self.protect_bytes,
                    seed=seed,
                )
            # channel uses approximation for LDPC(1/2)+BPSK: coded_bits ~= 2*info_bits.
            self._cu_total += int(2 * len(opus_blob) * 8)
            self._n_total += 1

        try:
            wav_d, sr_d = opus_decode_bytes_ffmpeg(opus_blob, ffmpeg_bin=self.ffmpeg_bin)
            self._decode_ok += 1
        except Exception:
            # If decoding fails due to erasures, fall back to silence.
            wav_d, sr_d = torch.zeros((10, int(sr)), dtype=torch.float32), int(sr)
            self._decode_fail += 1
        x = wav_to_logmel(
            wav_d,
            sr_d,
            target_T=self.target_T,
            norm_mean=self.norm_mean,
            norm_std=self.norm_std,
        )
        return x, label, machine_id


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    NAN = float("nan")
    ffmpeg_bin = resolve_ffmpeg_bin(args.ffmpeg_bin)
    print(f"[opus] using ffmpeg binary: {ffmpeg_bin}")

    stage1_ckpt = torch.load(args.stage1_ckpt, map_location="cpu", weights_only=True)
    use_norm = bool(stage1_ckpt.get("spectrogram_standardize", True))
    norm_mean, norm_std = load_norm_from_stage1_ckpt(stage1_ckpt) if use_norm else (None, None)

    train_ds = DCASE2020Task2LogMelDataset(
        root=args.data_path,
        machine_type=args.machine_type,
        include_test=False,
        norm_mean=norm_mean,
        norm_std=norm_std,
        standardize=use_norm,
        compute_norm_stats=False,
    )
    test_ds = DCASE2020Task2TestDataset(
        root=args.data_path,
        machine_type=args.machine_type,
        target_T=train_ds.target_T,
        norm_mean=norm_mean,
        norm_std=norm_std,
        standardize=use_norm,
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

    out_path = Path(args.output) if args.output else (Path(args.stage2_ckpt).resolve().parent / "results" / "opus_results.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    opus_cache_dir = Path(args.opus_cache_dir) if args.opus_cache_dir else (out_path.parent / "opus_cache" / args.machine_type / f"{int(args.opus_kbps)}kbps")
    if not args.no_preencode:
        print(f"[opus] caching enabled, cache_dir={opus_cache_dir}")
        n_present, n_encoded = preencode_opus_testset(base=test_ds, cache_dir=opus_cache_dir, kbps=int(args.opus_kbps), ffmpeg_bin=ffmpeg_bin)
        print(f"[opus] pre-encode done: cached={n_present}, newly_encoded={n_encoded}, total={len(test_ds)}")

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
                ds = OpusTestDataset(
                    test_ds,
                    kbps=args.opus_kbps,
                    use_channel=args.use_channel,
                    snr_db=float(snr_db),
                    seed=seed,
                    ber_curve_path=args.ber_curve,
                    protect_bytes=args.protect_bytes,
                    protect_pages=args.protect_pages,
                    channel_mode=args.channel_mode,
                    ffmpeg_bin=ffmpeg_bin,
                    opus_cache_dir=str(opus_cache_dir),
                    target_T=train_ds.target_T,
                    norm_mean=norm_mean,
                    norm_std=norm_std,
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
                ok, fail, okr = ds.decode_success_stats()
                for mid, v in ids.items():
                    if not isinstance(v, dict):
                        continue
                    w.writerow(
                        [
                            args.machine_type,
                            "opus",
                            "ber_bitflip" if args.use_channel else "none",
                            snr_db,
                            seed,
                            NAN,
                            args.opus_kbps,
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
                    f"[{args.machine_type}] method=opus channel={'ber_bitflip' if args.use_channel else 'none'} "
                    f"opus_kbps={int(args.opus_kbps)} snr={snr_db} seed={seed} "
                    f"cu_total={cu:.1f}/clip decode_ok={ok} decode_fail={fail} ok_rate={okr:.3f} "
                    f"clip_oor_c=nan clip_oor_f=nan avg={ids.get('average')}"
                )

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

