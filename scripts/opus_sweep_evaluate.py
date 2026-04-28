#!/usr/bin/env python3
"""
OPUS sweep evaluation on DCASE2020 Task 2 (single machine_type).

For each bitrate in kbps (default: 8,12,16,24,32), this script:
  1) takes each test .wav clip
  2) OPUS-encodes it at the target bitrate (via ffmpeg + libopus)
  3) decodes back to PCM wav
  4) computes the standardized log-mel spectrogram (same shape as training)
  5) runs the normal sDSR evaluation pipeline (AUC / pAUC)

Usage:
  python3 scripts/opus_sweep_evaluate.py --stage1_ckpt ... --stage2_ckpt ... \\
    --data_path /path/to/dcase --machine_type fan

Notes:
  - Requires `ffmpeg` with libopus enabled.
  - This is intentionally simple (uses temp files; slower but easy to inspect).
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Tuple

import torch
import torchaudio
import torchaudio.functional as AF
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.data.dataset import DCASE2020Task2LogMelDataset, DCASE2020Task2TestDataset, MEL_TIME_CROP
from src.engine.evaluator import AnomalyEvaluator
from src.models.sDSR.s_dsr import sDSR, sDSRConfig
from src.models.vq_vae.autoencoders import VQ_VAE_2Layer


def _parse_bitrates(brs: list[int] | None) -> list[int]:
    if not brs:
        return [8, 12, 16, 24, 32]
    out: list[int] = []
    for b in brs:
        bi = int(b)
        if bi <= 0:
            raise ValueError(f"Bitrate must be > 0 kbps, got {bi}")
        out.append(bi)
    return out


def _ensure_ffmpeg() -> str:
    ff = shutil.which("ffmpeg")
    if not ff:
        raise RuntimeError("ffmpeg not found on PATH. Install ffmpeg (with libopus) to use this script.")
    return ff


def opus_roundtrip_wav(
    wav_path: str,
    bitrate_kbps: int,
    ffmpeg: str,
) -> tuple[torch.Tensor, int]:
    """
    OPUS encode/decode a wav file using ffmpeg, returning (waveform, sample_rate).
    """
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        opus_path = td_path / "clip.ogg"
        dec_path = td_path / "clip_dec.wav"

        # Encode: PCM wav -> Ogg Opus
        enc_cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(wav_path),
            "-c:a",
            "libopus",
            "-b:a",
            f"{int(bitrate_kbps)}k",
            "-vbr",
            "off",
            "-application",
            "audio",
            str(opus_path),
        ]
        try:
            subprocess.run(enc_cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            stderr = (e.stderr or "").strip()
            # Some ffmpeg builds do not support `-vbr` as an option at all (even for libopus).
            # In that case, fall back to bitrate-only encoding (often VBR) rather than failing.
            if "Unrecognized option 'vbr'" in stderr or "Option not found" in stderr:
                enc_cmd_fallback = [x for x in enc_cmd if x not in ("-vbr", "off")]
                subprocess.run(enc_cmd_fallback, check=True, capture_output=True, text=True)
            else:
                raise

        # Decode: Ogg Opus -> PCM wav
        dec_cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(opus_path),
            "-acodec",
            "pcm_s16le",
            str(dec_path),
        ]
        subprocess.run(dec_cmd, check=True, capture_output=True, text=True)

        wav, sr = torchaudio.load(str(dec_path))
        return wav, int(sr)


class OpusTestSpectrogramDataset(Dataset):
    """
    Wrap DCASE2020Task2TestDataset samples, but load audio through Opus roundtrip.

    Output matches DCASE2020Task2TestDataset.__getitem__:
      (standardized_log_mel, label, machine_id)
    where standardized_log_mel is (1, n_mels, T').
    """

    def __init__(self, base: DCASE2020Task2TestDataset, bitrate_kbps: int, ffmpeg: str) -> None:
        self.base = base
        self.bitrate_kbps = int(bitrate_kbps)
        self.ffmpeg = ffmpeg

        # pass-through attrs used by evaluator / logging
        self.machine_type = getattr(base, "machine_type", "unknown")
        self.machine_ids = getattr(base, "machine_ids", None)

        # reuse base transforms and settings
        self.sample_rate = int(base.sample_rate)
        self.mel_transform = base.mel_transform
        self.to_db = base.to_db
        self.target_T = base.target_T

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        wav_path, label, machine_id = self.base.samples[idx]

        wav, sr = opus_roundtrip_wav(wav_path, self.bitrate_kbps, self.ffmpeg)
        if sr != self.sample_rate:
            wav = AF.resample(wav, sr, self.sample_rate)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)

        mel = self.mel_transform(wav)
        log_mel = self.to_db(mel).float()  # (1, n_mels, T)
        log_mel = log_mel[..., :MEL_TIME_CROP]

        return log_mel, int(label), str(machine_id)


def build_s_dsr(
    n_mels: int,
    T: int,
    vq_vae: VQ_VAE_2Layer,
    embedding_dim: Tuple[int, int],
    hidden_channels: Tuple[int, int],
) -> sDSR:
    cfg = sDSRConfig(
        embedding_dim=embedding_dim,
        hidden_channels=hidden_channels,
        n_mels=n_mels,
        T=T,
    )
    return sDSR(vq_vae, cfg)


def _compute_train_score_stats(
    model: torch.nn.Module,
    train_ds: DCASE2020Task2LogMelDataset,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, tuple[float, float]], tuple[float, float]]:
    from collections import defaultdict

    model.eval()
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    by_id: dict[str, list[float]] = defaultdict(list)
    all_scores: list[float] = []

    with torch.no_grad():
        for batch in loader:
            x, _labels, machine_ids = batch
            x = x.to(device)
            m_out = model(x)
            logits = m_out[:, 1]
            flat = logits.view(m_out.shape[0], -1)
            sc_mean = flat.mean(dim=1).cpu()
            for i in range(x.shape[0]):
                mid = machine_ids[i] if isinstance(machine_ids[i], str) else str(machine_ids[i])
                s = sc_mean[i].item()
                by_id[mid].append(s)
                all_scores.append(s)

    train_score_stats: dict[str, tuple[float, float]] = {}
    for mid, scores in by_id.items():
        mean_val = sum(scores) / len(scores)
        var = sum((x - mean_val) ** 2 for x in scores) / len(scores) if len(scores) > 1 else 0.0
        std_val = var**0.5
        train_score_stats[mid] = (mean_val, std_val)

    if all_scores:
        global_mean = sum(all_scores) / len(all_scores)
        global_var = sum((x - global_mean) ** 2 for x in all_scores) / len(all_scores)
        global_std = global_var**0.5
        fallback = (global_mean, global_std)
    else:
        fallback = (0.0, 1.0)
    return train_score_stats, fallback


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate sDSR under OPUS compression (single machine_type).")
    p.add_argument("--stage1_ckpt", type=str, required=True)
    p.add_argument("--stage2_ckpt", type=str, required=True)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--machine_type", type=str, required=True)
    p.add_argument("--machine_id", type=str, default=None)
    p.add_argument("--bitrates", type=int, nargs="*", default=None, help="OPUS bitrates in kbps, e.g. --bitrates 8 12 16 24 32")
    p.add_argument("--pauc_max_fpr", type=float, default=0.1)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--no_score_norm", action="store_true")
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional CSV output path (default: <stage2_ckpt_parent>/results/opus_sweep.csv)",
    )
    return p.parse_args()


def _run(args: argparse.Namespace, tee: Callable[[str], None]) -> None:
    ffmpeg = _ensure_ffmpeg()

    stage1_ckpt = torch.load(args.stage1_ckpt, map_location="cpu", weights_only=True)

    train_ds = DCASE2020Task2LogMelDataset(
        root=args.data_path,
        machine_type=args.machine_type,
        machine_id=args.machine_id,
    )
    test_ds = DCASE2020Task2TestDataset(
        root=args.data_path,
        machine_type=args.machine_type,
        target_T=train_ds.target_T,
        machine_id=args.machine_id,
    )
    _, _, n_mels, T = train_ds.data.shape

    num_embeddings_coarse = stage1_ckpt["num_embeddings_coarse"]
    num_embeddings_fine = stage1_ckpt["num_embeddings_fine"]
    embedding_dim_coarse = stage1_ckpt["embedding_dim_coarse"]
    embedding_dim_fine = stage1_ckpt["embedding_dim_fine"]
    hidden_channels_coarse = stage1_ckpt["hidden_channels_coarse"]
    hidden_channels_fine = stage1_ckpt["hidden_channels_fine"]
    num_residual_layers = stage1_ckpt["num_residual_layers"]

    vq_vae = VQ_VAE_2Layer(
        hidden_channels=(hidden_channels_coarse, hidden_channels_fine),
        num_residual_layers=num_residual_layers,
        num_embeddings=(num_embeddings_coarse, num_embeddings_fine),
        embedding_dim=(embedding_dim_coarse, embedding_dim_fine),
        commitment_cost=0.25,
        decay=0.99,
    )

    state1 = dict(stage1_ckpt["model_state_dict"])
    vq_vae.load_state_dict(state1)

    model = build_s_dsr(
        n_mels,
        T,
        vq_vae=vq_vae,
        embedding_dim=(embedding_dim_coarse, embedding_dim_fine),
        hidden_channels=(hidden_channels_coarse, hidden_channels_fine),
    )

    stage2 = torch.load(args.stage2_ckpt, map_location="cpu", weights_only=True)
    state2 = dict(stage2["model_state_dict"])
    model.load_state_dict(state2)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_score_stats: dict[str, tuple[float, float]] | None = None
    train_score_stats_fallback: tuple[float, float] | None = None
    if not args.no_score_norm:
        train_score_stats, train_score_stats_fallback = _compute_train_score_stats(
            model, train_ds, device, args.batch_size
        )
        tee("Calibrated per-machine_id anomaly score stats (score normalization enabled).")

    brs = _parse_bitrates(args.bitrates)
    tee(f"OPUS bitrate sweep (kbps): {brs}")

    rows: list[tuple[int, str, float, float]] = []
    for br in brs:
        opus_test = OpusTestSpectrogramDataset(test_ds, bitrate_kbps=br, ffmpeg=ffmpeg)
        evaluator = AnomalyEvaluator(
            model=model,
            test_dataset=opus_test,
            device=args.device,
            pauc_max_fpr=args.pauc_max_fpr,
            batch_size=args.batch_size,
            train_score_stats=train_score_stats,
            train_score_stats_fallback=train_score_stats_fallback,
        )
        results = evaluator.evaluate()
        ids = results.get(opus_test.machine_type, {})
        avg = ids.get("average", {"auc": float("nan"), "pauc": float("nan")})
        auc = float(avg.get("auc", float("nan")))
        pauc = float(avg.get("pauc", float("nan")))
        tee(f"{br:>2d} kbps: average AUC={auc:.4f} pAUC={pauc:.4f}")
        rows.append((br, opus_test.machine_type, auc, pauc))

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(args.stage2_ckpt).resolve().parent / "results" / "opus_sweep.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bitrate_kbps", "machine_type", "avg_AUC", "avg_pAUC"])
        for r in rows:
            w.writerow([r[0], r[1], f"{r[2]:.6f}", f"{r[3]:.6f}"])
    tee(f"Saved sweep CSV to {out_path}")


def main() -> None:
    args = parse_args()
    results_dir = Path(args.stage2_ckpt).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "opus_sweep.log"
    log_file = open(log_path, "w", encoding="utf-8")

    def tee(msg: str) -> None:
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    try:
        _run(args, tee)
    finally:
        log_file.close()


if __name__ == "__main__":
    main()

