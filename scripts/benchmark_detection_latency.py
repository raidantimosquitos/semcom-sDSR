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

from src.benchmark.pipelines import load_wav_and_mel, make_pipeline_config, run_pipeline
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


def load_models(args: argparse.Namespace, device: torch.device) -> tuple[sDSR, VQ_VAE_2Layer, int, int]:
    stage1_ckpt = torch.load(args.stage1_ckpt, map_location="cpu", weights_only=True)
    n_mels = int(stage1_ckpt["n_mels"])
    target_T = int(stage1_ckpt["target_T"])

    vq_vae = VQ_VAE_2Layer(
        hidden_channels=(stage1_ckpt["hidden_channels_coarse"], stage1_ckpt["hidden_channels_fine"]),
        num_residual_layers=stage1_ckpt["num_residual_layers"],
        num_embeddings=(stage1_ckpt["num_embeddings_coarse"], stage1_ckpt["num_embeddings_fine"]),
        embedding_dim=(stage1_ckpt["embedding_dim_coarse"], stage1_ckpt["embedding_dim_fine"]),
        commitment_cost=0.25,
        decay=0.99,
    )
    vq_vae.load_state_dict(dict(stage1_ckpt["model_state_dict"]))

    model = build_s_dsr(
        n_mels,
        target_T,
        vq_vae=vq_vae,
        embedding_dim=(stage1_ckpt["embedding_dim_coarse"], stage1_ckpt["embedding_dim_fine"]),
        hidden_channels=(stage1_ckpt["hidden_channels_coarse"], stage1_ckpt["hidden_channels_fine"]),
        num_residual_layers=stage1_ckpt["num_residual_layers"],
    )
    stage2 = torch.load(args.stage2_ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(dict(stage2["model_state_dict"]))
    model = model.to(device).eval()
    vq_vae = vq_vae.to(device).eval()
    return model, vq_vae, n_mels, target_T


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
    ms = result.times.to_ms_dict()
    return {
        "method": method,
        "wav_path": str(wav_path),
        "seed": seed,
        "repeat": repeat,
        "snr_db": snr_db,
        "use_channel": int(use_channel),
        "t_load_wav_ms": f"{ms['t_load_wav']:.4f}",
        "t_mel_ms": f"{ms['t_mel']:.4f}",
        "t_tx_codec_ms": f"{ms['t_tx_codec']:.4f}",
        "t_channel_ms": f"{ms['t_channel']:.4f}",
        "t_rx_codec_ms": f"{ms['t_rx_codec']:.4f}",
        "t_detector_ms": f"{ms['t_detector']:.4f}",
        "t_score_ms": f"{ms['t_score']:.4f}",
        "t_e2e_ms": f"{ms['t_e2e']:.4f}",
        "payload_bytes": result.payload_bytes,
        "decode_ok": int(result.decode_ok),
        "anomaly_score": f"{result.anomaly_score:.6f}",
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m not in ("latent", "jpeg", "opus"):
            raise ValueError(f"Unknown method: {m}")

    if args.use_channel and not args.ber_curve:
        raise ValueError("--use_channel requires --ber_curve")

    wav_paths = resolve_wav_paths(args)
    model, vq_vae, n_mels, target_T = load_models(args, device)

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
        "t_load_wav_ms",
        "t_mel_ms",
        "t_tx_codec_ms",
        "t_channel_ms",
        "t_rx_codec_ms",
        "t_detector_ms",
        "t_score_ms",
        "t_e2e_ms",
        "payload_bytes",
        "decode_ok",
        "anomaly_score",
    ]

    all_rows: list[dict] = []
    by_method_wav: dict[tuple[str, str], list[StageTimes]] = defaultdict(list)
    by_method: dict[str, list[StageTimes]] = defaultdict(list)

    for method in methods:
        for wav_path in wav_paths:
            preloaded = load_wav_and_mel(
                wav_path,
                sample_rate=16_000,
                target_T=target_T,
                mel_transform=mel_transform,
                to_db=to_db,
            )

            for _ in range(args.warmup):
                cfg = make_pipeline_config(
                    device=device,
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
                    vq_vae=vq_vae,
                    cfg=cfg,
                    mel_transform=mel_transform,
                    to_db=to_db,
                    preloaded=preloaded,
                )
                sync_device(device)

            for seed in args.seeds:
                for rep in range(args.repeats):
                    cfg = make_pipeline_config(
                        device=device,
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
                        vq_vae=vq_vae,
                        cfg=cfg,
                        mel_transform=mel_transform,
                        to_db=to_db,
                        preloaded=preloaded,
                    )
                    sync_device(device)

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
        "device": str(device),
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

    print("\n=== Latency summary (median ms) ===")
    for (method, wav_str), samples in sorted(by_method_wav.items()):
        agg = aggregate_times(samples)
        summary["per_method_wav"][f"{method}|{wav_str}"] = {
            k: {stat: v * 1000.0 for stat, v in stats.items()}
            for k, stats in agg.items()
        }
        e2e_med = agg.get("t_e2e", {}).get("median", 0.0) * 1000.0
        det_med = agg.get("t_detector", {}).get("median", 0.0) * 1000.0
        tx_med = agg.get("t_tx_codec", {}).get("median", 0.0) * 1000.0
        print(
            f"  {method} | {Path(wav_str).name}: e2e={e2e_med:.2f} ms "
            f"(tx_codec={tx_med:.2f}, detector={det_med:.2f}) n={len(samples)}"
        )

    for method, samples in sorted(by_method.items()):
        agg = aggregate_times(samples)
        summary["per_method"][method] = {
            k: {stat: v * 1000.0 for stat, v in stats.items()}
            for k, stats in agg.items()
        }
        parts = [f"  {method} (all wavs):"]
        for stage in ("t_load_wav", "t_mel", "t_tx_codec", "t_rx_codec", "t_detector", "t_e2e"):
            med = agg.get(stage, {}).get("median", 0.0) * 1000.0
            parts.append(f"{stage}={med:.2f}")
        print(" ".join(parts) + f" ms n={len(samples)}")

    print(f"\nSaved per-run CSV: {out_path}")

    if args.summary_json:
        json_path = Path(args.summary_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(summary, jf, indent=2)
        print(f"Saved summary JSON: {json_path}")


if __name__ == "__main__":
    main()
