#!/usr/bin/env python3
"""
Benchmark wav-to-anomaly-score latency for JPEG, OPUS, and VQ-VAE-2 latent pipelines.

Uses a small set of example WAV files (not the full DCASE dataset). Multiple seeds
mainly affect BER bitflip when --use_channel is enabled; with a clean channel, seeds
are redundant but kept for a stable CLI.

Example:
  python3 -m scripts.benchmark_detection_latency \\
    --stage1_ckpt checkpoints/stage1/fan/best.pt \\
    --stage2_ckpt checkpoints/stage2/fan/best.pt \\
    --wav dataset/.../fan/test/normal_id_00_00000000.wav \\
    --warmup 10 --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch

from src.benchmark.pipelines import make_pipeline_config, run_pipeline
from src.benchmark.timing import StageTimes, aggregate_times, sync_device
from src.models.sDSR.s_dsr import sDSR, sDSRConfig
from src.models.vq_vae.autoencoders import VQ_VAE_2Layer
from src.utils.audio import amplitude_to_db_power, make_mel_spectrogram

from scripts.evaluate_awgn_opus import resolve_ffmpeg_bin


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark detection latency (JPEG / OPUS / latent).")
    p.add_argument("--stage1_ckpt", type=str, required=True)
    p.add_argument("--stage2_ckpt", type=str, required=True)
    p.add_argument("--wav", type=str, nargs="*", default=None, help="Example WAV path(s).")
    p.add_argument("--data_path", type=str, default=None, help="Used with --machine_type if --wav omitted.")
    p.add_argument("--machine_type", type=str, default=None)
    p.add_argument(
        "--methods",
        type=str,
        default="latent,jpeg,opus",
        help="Comma-separated: latent,jpeg,opus",
    )
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--tx_device",
        type=str,
        default="cpu",
        help="TX semantic encoder device (affects latent/VQ encoder). Default: cpu.",
    )
    p.add_argument("--jpeg_quality", type=int, default=70)
    p.add_argument("--opus_kbps", type=int, default=64)
    p.add_argument("--ffmpeg_bin", type=str, default=None)
    p.add_argument("--use_channel", action="store_true")
    p.add_argument("--ber_curve", type=str, default=None)
    p.add_argument("--snr_db", type=float, default=20.0)
    p.add_argument("--jpeg_channel_mode", type=str, choices=["jpeg_entropy", "prefix"], default="jpeg_entropy")
    p.add_argument("--jpeg_protect_bytes", type=int, default=8)
    p.add_argument("--opus_channel_mode", type=str, choices=["ogg_pages", "prefix"], default="ogg_pages")
    p.add_argument("--opus_protect_pages", type=int, default=2)
    p.add_argument("--opus_protect_bytes", type=int, default=128)
    p.add_argument("--bits_coarse", type=int, default=None)
    p.add_argument("--bits_fine", type=int, default=None)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--summary_json", type=str, default=None)
    return p.parse_args()


def build_s_dsr(
    n_mels: int,
    T: int,
    vq_vae: VQ_VAE_2Layer,
    embedding_dim: tuple[int, int],
    hidden_channels: tuple[int, int],
    num_residual_layers: int,
) -> sDSR:
    cfg = sDSRConfig(
        embedding_dim=embedding_dim,
        hidden_channels=hidden_channels,
        num_residual_layers=num_residual_layers,
        n_mels=n_mels,
        T=T,
    )
    return sDSR(vq_vae, cfg)


def resolve_wav_paths(args: argparse.Namespace) -> list[Path]:
    if args.wav:
        paths = [Path(w).resolve() for w in args.wav]
        for p in paths:
            if not p.is_file():
                raise FileNotFoundError(f"WAV not found: {p}")
        return paths
    if args.data_path is None or args.machine_type is None:
        raise ValueError("Provide --wav or both --data_path and --machine_type")
    test_dir = Path(args.data_path) / args.machine_type / "test"
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    wavs = sorted(test_dir.glob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"No WAV files in {test_dir}")
    return [wavs[0].resolve()]


def _build_vq_vae_from_stage1(stage1_ckpt: dict) -> VQ_VAE_2Layer:
    vq_vae = VQ_VAE_2Layer(
        hidden_channels=(stage1_ckpt["hidden_channels_coarse"], stage1_ckpt["hidden_channels_fine"]),
        num_residual_layers=stage1_ckpt["num_residual_layers"],
        num_embeddings=(stage1_ckpt["num_embeddings_coarse"], stage1_ckpt["num_embeddings_fine"]),
        embedding_dim=(stage1_ckpt["embedding_dim_coarse"], stage1_ckpt["embedding_dim_fine"]),
        commitment_cost=0.25,
        decay=0.99,
    )
    vq_vae.load_state_dict(dict(stage1_ckpt["model_state_dict"]))
    return vq_vae


def load_models(
    args: argparse.Namespace, rx_device: torch.device, tx_device: torch.device
) -> tuple[sDSR, VQ_VAE_2Layer, VQ_VAE_2Layer, int, int]:
    stage1_ckpt = torch.load(args.stage1_ckpt, map_location="cpu", weights_only=True)
    n_mels = int(stage1_ckpt["n_mels"])
    target_T = int(stage1_ckpt["target_T"])

    vq_vae_rx = _build_vq_vae_from_stage1(stage1_ckpt).to(rx_device).eval()
    if tx_device == rx_device:
        vq_vae_tx = vq_vae_rx
    else:
        vq_vae_tx = _build_vq_vae_from_stage1(stage1_ckpt).to(tx_device).eval()

    model = build_s_dsr(
        n_mels,
        target_T,
        vq_vae=vq_vae_rx,
        embedding_dim=(stage1_ckpt["embedding_dim_coarse"], stage1_ckpt["embedding_dim_fine"]),
        hidden_channels=(stage1_ckpt["hidden_channels_coarse"], stage1_ckpt["hidden_channels_fine"]),
        num_residual_layers=stage1_ckpt["num_residual_layers"],
    )
    stage2 = torch.load(args.stage2_ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(dict(stage2["model_state_dict"]))
    model = model.to(rx_device).eval()
    return model, vq_vae_tx, vq_vae_rx, n_mels, target_T


def row_from_result(
    *,
    method: str,
    wav_path: Path,
    seed: int,
    repeat: int,
    snr_db: float,
    use_channel: bool,
    result,
) -> dict:
    # Use definition-aligned breakdown emitted by pipelines.py (seconds).
    tx_load = float(result.extra.get("tx_t_load_wav", 0.0))
    tx_spec = float(result.extra.get("tx_t_compute_spectrogram", 0.0))
    tx_payload = float(result.extra.get("tx_t_to_payload", 0.0))

    rx_decode_to_latents = float(result.extra.get("rx_t_decode_to_latents", 0.0))
    rx_gen = float(result.extra.get("rx_t_general_dec", 0.0))
    rx_obj = float(result.extra.get("rx_t_object_dec", 0.0))
    rx_anom = float(result.extra.get("rx_t_anom_det", 0.0))
    rx_score = float(result.extra.get("rx_t_score", 0.0))

    tx_total = tx_load + tx_spec + tx_payload
    rx_total = rx_decode_to_latents + max(rx_gen, rx_obj) + rx_anom + rx_score
    return {
        "method": method,
        "wav_path": str(wav_path),
        "seed": seed,
        "repeat": repeat,
        "snr_db": snr_db,
        "use_channel": int(use_channel),
        "tx_t_load_wav": f"{tx_load * 1000.0:.4f}",
        "tx_t_compute_spectrogram": f"{tx_spec * 1000.0:.4f}",
        "tx_t_to_payload": f"{tx_payload * 1000.0:.4f}",
        "tx_total": f"{tx_total * 1000.0:.4f}",
        "t_decode_to_latents": f"{rx_decode_to_latents * 1000.0:.4f}",
        "t_general_dec": f"{rx_gen * 1000.0:.4f}",
        "t_object_dec": f"{rx_obj * 1000.0:.4f}",
        "t_anom_det": f"{rx_anom * 1000.0:.4f}",
        "t_score_ms": f"{rx_score * 1000.0:.4f}",
        "rx_total": f"{rx_total * 1000.0:.4f}",
        "payload_bytes": result.payload_bytes,
        "decode_ok": int(result.decode_ok),
        "anomaly_score": f"{result.anomaly_score:.6f}",
    }


def main() -> None:
    args = parse_args()
    rx_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tx_device = torch.device(args.tx_device if (args.tx_device != "cuda" or torch.cuda.is_available()) else "cpu")
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m not in ("latent", "jpeg", "opus"):
            raise ValueError(f"Unknown method: {m}")

    if args.use_channel and not args.ber_curve:
        raise ValueError("--use_channel requires --ber_curve")

    wav_paths = resolve_wav_paths(args)
    model, vq_vae_tx, vq_vae_rx, n_mels, target_T = load_models(args, rx_device, tx_device)

    if "opus" in methods:
        resolve_ffmpeg_bin(args.ffmpeg_bin)

    mel_transform = make_mel_spectrogram()
    to_db = amplitude_to_db_power()

    out_path = (
        Path(args.output)
        if args.output
        else Path(args.stage2_ckpt).resolve().parent / "results" / "latency_benchmark.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "method",
        "wav_path",
        "seed",
        "repeat",
        "snr_db",
        "use_channel",
        "tx_t_load_wav",
        "tx_t_compute_spectrogram",
        "tx_t_to_payload",
        "tx_total",
        "t_decode_to_latents",
        "t_general_dec",
        "t_object_dec",
        "t_anom_det",
        "t_score_ms",
        "rx_total",
        "payload_bytes",
        "decode_ok",
        "anomaly_score",
    ]

    all_rows: list[dict] = []
    by_method_wav: dict[tuple[str, str], list[StageTimes]] = defaultdict(list)
    by_method: dict[str, list[StageTimes]] = defaultdict(list)

    for method in methods:
        for wav_path in wav_paths:
            for _ in range(args.warmup):
                cfg = make_pipeline_config(
                    tx_device=tx_device,
                    device=rx_device,
                    n_mels=n_mels,
                    target_T=target_T,
                    seed=0,
                    use_channel=args.use_channel,
                    snr_db=args.snr_db,
                    ber_curve_path=args.ber_curve,
                    jpeg_quality=args.jpeg_quality,
                    jpeg_channel_mode=args.jpeg_channel_mode,
                    jpeg_protect_bytes=args.jpeg_protect_bytes,
                    opus_kbps=args.opus_kbps,
                    opus_channel_mode=args.opus_channel_mode,
                    opus_protect_pages=args.opus_protect_pages,
                    opus_protect_bytes=args.opus_protect_bytes,
                    ffmpeg_bin=args.ffmpeg_bin,
                    bits_coarse=args.bits_coarse,
                    bits_fine=args.bits_fine,
                )
                run_pipeline(
                    method,
                    wav_path,
                    model=model,
                    vq_vae_tx=vq_vae_tx,
                    vq_vae_rx=vq_vae_rx,
                    cfg=cfg,
                    mel_transform=mel_transform,
                    to_db=to_db,
                )
                sync_device(rx_device)

            for seed in args.seeds:
                for rep in range(args.repeats):
                    cfg = make_pipeline_config(
                        tx_device=tx_device,
                        device=rx_device,
                        n_mels=n_mels,
                        target_T=target_T,
                        seed=seed,
                        use_channel=args.use_channel,
                        snr_db=args.snr_db,
                        ber_curve_path=args.ber_curve,
                        jpeg_quality=args.jpeg_quality,
                        jpeg_channel_mode=args.jpeg_channel_mode,
                        jpeg_protect_bytes=args.jpeg_protect_bytes,
                        opus_kbps=args.opus_kbps,
                        opus_channel_mode=args.opus_channel_mode,
                        opus_protect_pages=args.opus_protect_pages,
                        opus_protect_bytes=args.opus_protect_bytes,
                        ffmpeg_bin=args.ffmpeg_bin,
                        bits_coarse=args.bits_coarse,
                        bits_fine=args.bits_fine,
                    )
                    result = run_pipeline(
                        method,
                        wav_path,
                        model=model,
                        vq_vae_tx=vq_vae_tx,
                        vq_vae_rx=vq_vae_rx,
                        cfg=cfg,
                        mel_transform=mel_transform,
                        to_db=to_db,
                    )
                    sync_device(rx_device)

                    row = row_from_result(
                        method=method,
                        wav_path=wav_path,
                        seed=seed,
                        repeat=rep,
                        snr_db=args.snr_db,
                        use_channel=args.use_channel,
                        result=result,
                    )
                    all_rows.append(row)
                    by_method_wav[(method, str(wav_path))].append(result.times)
                    by_method[method].append(result.times)

                    stage_sum_ms = result.times.stage_sum() * 1000.0
                    e2e_ms = result.times.t_e2e * 1000.0
                    drift = abs(e2e_ms - stage_sum_ms) / max(e2e_ms, 1e-6)
                    if drift > 0.05:
                        print(
                            f"[warn] {method} {wav_path.name} seed={seed} rep={rep}: "
                            f"stage sum drift {drift * 100:.1f}%"
                        )

            print(f"[done] warmup={args.warmup} method={method} wav={wav_path.name}")

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)

    summary: dict = {
        "rx_device": str(rx_device),
        "tx_device": str(tx_device),
        "n_wavs": len(wav_paths),
        "methods": methods,
        "warmup": args.warmup,
        "seeds": args.seeds,
        "repeats": args.repeats,
        "use_channel": args.use_channel,
        "snr_db": args.snr_db,
        "per_method_wav": {},
        "per_method": {},
    }

    print("\n=== TX/RX breakdown (definition-aligned; medians over runs) ===")
    print("TX_VQVAE2   : load_wav + mel + enc_vq_vae (encode_to_indices)")
    print("RX_VQVAE2   : indices_to_quantized + decode_vqvae + anomaly_score")
    print("TX_JPEG     : load_wav + mel + enc_to_JPEG")
    print("RX_JPEG     : decode_JPEG + enc_vq_vae + decode_vqvae + anomaly_score")
    print("TX_OPUS     : load_wav + compress_to_OPUS")
    print("RX_OPUS     : decompress_OPUS + mel + enc_vq_vae + decode_vqvae + anomaly_score")
    print("Totals use max(general_dec, object_dec) to reflect parallel decode.\n")

    print(f"\nSaved per-run CSV: {out_path}")

    if args.summary_json:
        json_path = Path(args.summary_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(summary, jf, indent=2)
        print(f"Saved summary JSON: {json_path}")


if __name__ == "__main__":
    main()
