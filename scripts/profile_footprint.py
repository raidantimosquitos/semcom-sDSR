#!/usr/bin/env python3
"""
Profile inference footprint for sDSR deployment splits (TX / RX).

This script is designed for thesis reporting:
  - Parameter counts (trainable + total)
  - State-dict size on disk (MB)
  - Peak CUDA memory during a forward (MB) if using GPU
  - Latency (ms) with warmup + repeats
  - FLOPs (approx) using torch.profiler (if supported by your PyTorch build)

Typical deployment split in this repo:
  - TX (sender): VQ-VAE encoder + quantizer -> indices  (see VQ_VAE_2Layer.encode_to_indices)
  - RX (receiver): indices -> embeddings -> general decoder + object-specific decoder
    (optionally + anomaly detection head if you deploy detection on receiver)

Example:
  python3 -m scripts.profile_footprint \
    --stage1_ckpt checkpoints/stage1/fan/best.pt \
    --stage2_ckpt checkpoints/stage2/fan/best.pt \
    --device cuda \
    --wav dataset/.../fan/test/normal_id_00_00000000.wav

TX encoder is always profiled on CPU at batch size 1 (edge). RX modules use --device and --batch.
"""

from __future__ import annotations

import argparse
import copy
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio

from src.benchmark.pipelines import load_wav, wav_to_mel
from src.models.sDSR.s_dsr import sDSR, sDSRConfig
from src.models.vq_vae.autoencoders import VQ_VAE_2Layer
from src.utils.audio import amplitude_to_db_power, make_mel_spectrogram


@dataclass
class Footprint:
    params_total: int
    params_trainable: int
    state_dict_mb: float
    peak_cuda_mb: float | None
    latency_ms: float
    flops: float | None


class VQVAE_TX_Encoder(torch.nn.Module):
    """
    TX-side subset of VQ_VAE_2Layer used by encode_to_indices().

    Includes:
      - _encoder_fine, _encoder_coarse
      - _pre_vq_conv_coarse, _vq_coarse
      - _upscale_coarse (used to condition fine)
      - _pre_vq_conv_fine, _vq_fine

    Excludes:
      - _decoder_fine (RX-side)
    """

    def __init__(self, vq: VQ_VAE_2Layer) -> None:
        super().__init__()
        self._encoder_fine = vq._encoder_fine
        self._encoder_coarse = vq._encoder_coarse
        self._pre_vq_conv_coarse = vq._pre_vq_conv_coarse
        self._vq_coarse = vq._vq_coarse
        self._upscale_coarse = vq._upscale_coarse
        self._pre_vq_conv_fine = vq._pre_vq_conv_fine
        self._vq_fine = vq._vq_fine

    def encode_to_indices(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Same computation as VQ_VAE_2Layer.encode_to_indices, but using subset modules.
        f_fine = self._encoder_fine(x)
        f_coarse = self._encoder_coarse(f_fine)
        z_coarse = self._pre_vq_conv_coarse(f_coarse)
        idx_coarse_flat = self._vq_coarse.get_indices(z_coarse)
        _, quantized_coarse, _, _ = self._vq_coarse(z_coarse)
        quantized_coarse_up = self._upscale_coarse(quantized_coarse)
        feat_fine = torch.cat([f_fine, quantized_coarse_up], dim=1)
        z_fine = self._pre_vq_conv_fine(feat_fine)
        idx_fine_flat = self._vq_fine.get_indices(z_fine)
        B, _, H_coarse, W_coarse = z_coarse.shape
        _, _, H_fine, W_fine = z_fine.shape
        indices_coarse = idx_coarse_flat.view(B, H_coarse, W_coarse)
        indices_fine = idx_fine_flat.view(B, H_fine, W_fine)
        return indices_coarse, indices_fine


class VQVAE_RX_GeneralDecoder(torch.nn.Module):
    """
    RX-side subset of VQ_VAE_2Layer used by decode_general().

    Includes:
      - _upscale_coarse, _decoder_fine

    Excludes:
      - encoders, quantizers
    """

    def __init__(self, vq: VQ_VAE_2Layer) -> None:
        super().__init__()
        self._upscale_coarse = vq._upscale_coarse
        self._decoder_fine = vq._decoder_fine

    def decode_general(self, q_fine: torch.Tensor, q_coarse: torch.Tensor) -> torch.Tensor:
        quantized_coarse_up = self._upscale_coarse(q_coarse)
        quant_joined = torch.cat([quantized_coarse_up, q_fine], dim=1)
        return self._decoder_fine(quant_joined)


def _num_params(m: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return int(total), int(trainable)


def _state_dict_size_mb(state_dict: dict) -> float:
    # Serialize to a temporary buffer on disk to include tensor storage.
    # (torch.save to BytesIO can be expensive on large checkpoints; disk is fine.)
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp = f.name
    try:
        torch.save(state_dict, tmp)
        return float(os.path.getsize(tmp)) / (1024.0 * 1024.0)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def _sync(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _measure_latency(
    fn,
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> float:
    for _ in range(warmup):
        fn()
        _sync(device)
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
        _sync(device)
    dt = time.perf_counter() - t0
    return (dt / max(1, repeats)) * 1000.0


def _measure_peak_cuda_mb(fn, *, device: torch.device) -> float | None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    torch.cuda.reset_peak_memory_stats(device)
    fn()
    _sync(device)
    return float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)


def _measure_flops_with_profiler(fn, *, device: torch.device) -> float | None:
    """
    Returns total FLOPs reported by torch.profiler (may be unavailable depending on torch build).
    """
    try:
        from torch.profiler import profile, ProfilerActivity
    except Exception:
        return None

    acts = [ProfilerActivity.CPU]
    if device.type == "cuda" and torch.cuda.is_available():
        acts.append(ProfilerActivity.CUDA)

    try:
        with profile(activities=acts, with_flops=True, record_shapes=False) as prof:
            fn()
            _sync(device)
        # Sum FLOPs across events if present
        total = 0.0
        for e in prof.key_averages():
            fl = getattr(e, "flops", None)
            if fl is not None:
                total += float(fl)
        return total if total > 0 else None
    except Exception:
        return None


def _resolve_wav_for_mel(
    wav_arg: str | None,
    *,
    sample_rate: int,
    clip_seconds: float,
) -> tuple[Path, bool]:
    """
    Return (wav_path, is_temporary).

    If ``wav_arg`` is set, use that file. Otherwise create a temporary mono WAV
    of length ``clip_seconds`` for reproducible load+I/O timing.
    """
    if wav_arg:
        path = Path(wav_arg).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"WAV not found: {path}")
        return path, False

    n_samples = int(round(sample_rate * clip_seconds))
    wav = torch.zeros(1, n_samples)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    torchaudio.save(str(tmp_path), wav, sample_rate)
    return tmp_path, True


def profile_mel_frontend(
    *,
    wav_path: Path,
    target_T: int,
    sample_rate: int,
    warmup: int,
    repeats: int,
) -> tuple[Footprint, Footprint, Footprint]:
    """
    Profile edge mel pipeline on CPU: load wav, mel+db+crop/pad, end-to-end.

    Returns (load_wav, mel_compute, load_plus_mel).
    """
    device = torch.device("cpu")
    mel_transform = make_mel_spectrogram(sample_rate=sample_rate)
    to_db = amplitude_to_db_power()

    wav_holder: list[torch.Tensor] = []

    def _load() -> None:
        wav, _sr = load_wav(wav_path, sample_rate=sample_rate)
        wav_holder.clear()
        wav_holder.append(wav)

    def _mel_from_holder() -> None:
        if not wav_holder:
            _load()
        _ = wav_to_mel(
            wav_holder[0],
            target_T=target_T,
            mel_transform=mel_transform,
            to_db=to_db,
        )

    def _e2e() -> None:
        wav, _sr = load_wav(wav_path, sample_rate=sample_rate)
        _ = wav_to_mel(
            wav,
            target_T=target_T,
            mel_transform=mel_transform,
            to_db=to_db,
        )

    print("\n=== TX edge frontend (CPU): WAV -> log-mel ===")
    print(f"  wav_path         : {wav_path}")
    print(f"  sample_rate      : {sample_rate} Hz")
    print(f"  target_T (mel)   : {target_T}")

    load_fp = profile_runtime_only(
        "TX load WAV (mono, resample to 16 kHz if needed)",
        _load,
        device=device,
        warmup=warmup,
        repeats=repeats,
    )
    # Prime wav_holder once before mel-only timings.
    _load()
    mel_fp = profile_runtime_only(
        "TX log-mel (MelSpectrogram + dB + crop/pad)",
        _mel_from_holder,
        device=device,
        warmup=warmup,
        repeats=repeats,
    )
    e2e_fp = profile_runtime_only(
        "TX load WAV + log-mel (end-to-end frontend)",
        _e2e,
        device=device,
        warmup=warmup,
        repeats=repeats,
    )
    print(
        f"\n  TX frontend sum (load + mel, approximate): "
        f"{load_fp.latency_ms + mel_fp.latency_ms:.3f} ms "
        f"(e2e measured: {e2e_fp.latency_ms:.3f} ms)"
    )
    return load_fp, mel_fp, e2e_fp


def build_models(args: argparse.Namespace) -> tuple[sDSR, VQ_VAE_2Layer, int, int]:
    """Load checkpoints on CPU; caller moves modules to RX device after TX profiling."""
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
    vq_vae.eval()

    cfg = sDSRConfig(
        embedding_dim=(stage1_ckpt["embedding_dim_coarse"], stage1_ckpt["embedding_dim_fine"]),
        hidden_channels=(stage1_ckpt["hidden_channels_coarse"], stage1_ckpt["hidden_channels_fine"]),
        num_residual_layers=stage1_ckpt["num_residual_layers"],
        n_mels=n_mels,
        T=target_T,
    )
    model = sDSR(vq_vae, cfg)
    stage2 = torch.load(args.stage2_ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(dict(stage2["model_state_dict"]))
    model.eval()
    return model, vq_vae, n_mels, target_T


def profile_runtime_only(
    name: str,
    fn,
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> Footprint:
    """Latency/FLOPs only (no params / state_dict), e.g. mel frontend or load I/O."""
    peak = _measure_peak_cuda_mb(fn, device=device)
    lat = _measure_latency(fn, device=device, warmup=warmup, repeats=repeats)
    flops = _measure_flops_with_profiler(fn, device=device)
    print(f"\n[{name}]")
    print(f"  device           : {device}")
    if peak is not None:
        print(f"  peak_cuda_mem    : {peak:.2f} MB")
    print(f"  latency          : {lat:.3f} ms (avg over {repeats}, warmup {warmup})")
    if flops is not None:
        print(f"  flops            : {flops:.3e}")
    else:
        print("  flops            : (not available in this torch build)")
    return Footprint(
        params_total=0,
        params_trainable=0,
        state_dict_mb=0.0,
        peak_cuda_mb=peak,
        latency_ms=lat,
        flops=flops,
    )


def profile_module(
    name: str,
    module: torch.nn.Module,
    state_dict: dict,
    fn,
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> Footprint:
    p_total, p_train = _num_params(module)
    sd_mb = _state_dict_size_mb(state_dict)
    peak = _measure_peak_cuda_mb(fn, device=device)
    lat = _measure_latency(fn, device=device, warmup=warmup, repeats=repeats)
    flops = _measure_flops_with_profiler(fn, device=device)
    print(f"\n[{name}]")
    print(f"  device           : {device}")
    print(f"  params_total     : {p_total:,}")
    print(f"  params_trainable : {p_train:,}")
    print(f"  state_dict_size  : {sd_mb:.2f} MB")
    if peak is not None:
        print(f"  peak_cuda_mem    : {peak:.2f} MB")
    print(f"  latency          : {lat:.3f} ms (avg over {repeats}, warmup {warmup})")
    if flops is not None:
        print(f"  flops            : {flops:.3e}")
    else:
        print("  flops            : (not available in this torch build)")
    return Footprint(
        params_total=p_total,
        params_trainable=p_train,
        state_dict_mb=sd_mb,
        peak_cuda_mb=peak,
        latency_ms=lat,
        flops=flops,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile computational footprint for sDSR TX/RX splits.")
    p.add_argument("--stage1_ckpt", type=str, required=True)
    p.add_argument("--stage2_ckpt", type=str, required=True)
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for RX modules (general/object/detector).",
    )
    p.add_argument(
        "--tx_device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for TX semantic encoder (encode_to_indices).",
    )
    p.add_argument("--batch", type=int, default=1, help="Batch size for RX profiling only.")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument(
        "--wav",
        type=str,
        default=None,
        help="Example 10 s mono WAV for mel frontend timing. If omitted, a synthetic 10 s clip is created.",
    )
    p.add_argument("--sample_rate", type=int, default=16_000)
    p.add_argument(
        "--clip_seconds",
        type=float,
        default=10.0,
        help="Length of synthetic WAV when --wav is not set.",
    )
    p.add_argument(
        "--skip_mel",
        action="store_true",
        help="Skip WAV load + log-mel frontend profiling.",
    )
    p.add_argument(
        "--include_detector",
        action="store_true",
        help="Also profile receiver anomaly detection head (x_specific/x_general -> m_out).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rx_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tx_device = torch.device(args.tx_device if (args.tx_device != "cuda" or torch.cuda.is_available()) else "cpu")
    print(f"RX device: {rx_device}")
    print(f"TX device: {tx_device} (encoder batch size fixed to 1)")

    model, vq_vae, n_mels, target_T = build_models(args)

    tmp_wav: Path | None = None
    if not args.skip_mel:
        wav_path, is_tmp = _resolve_wav_for_mel(
            args.wav,
            sample_rate=int(args.sample_rate),
            clip_seconds=float(args.clip_seconds),
        )
        if is_tmp:
            tmp_wav = wav_path
        try:
            profile_mel_frontend(
                wav_path=wav_path,
                target_T=target_T,
                sample_rate=int(args.sample_rate),
                warmup=args.warmup,
                repeats=args.repeats,
            )
        finally:
            if tmp_wav is not None:
                try:
                    os.remove(tmp_wav)
                except OSError:
                    pass

    # -------- TX: encoder-only on TX device, batch=1 (edge).
    # IMPORTANT: VQVAE_TX_Encoder aliases VQ-VAE submodules. We keep TX and RX weights
    # separate to prevent CPU<->CUDA device-mismatch when RX weights are moved.
    vq_vae_tx = copy.deepcopy(vq_vae).to(tx_device).eval()
    vq_tx = VQVAE_TX_Encoder(vq_vae_tx).eval()
    x_tx = torch.randn(1, 1, n_mels, target_T, device=tx_device)

    def tx_fn():
        with torch.inference_mode():
            _ = vq_tx.encode_to_indices(x_tx)

    print(f"\n=== TX edge semantic encoder ({tx_device}, batch=1) ===")
    tx_fp = profile_module(
        "TX encoder-only (VQ encoders + quantizers -> indices)",
        vq_tx,
        vq_tx.state_dict(),
        tx_fn,
        device=tx_device,
        warmup=args.warmup,
        repeats=args.repeats,
    )

    # Definition-aligned TX/RX timing summary (latent only).
    # TX: t_load_wav, t_spectrogram, t_encode_to_indices
    # RX: t_decode_to_vq_vae_latents, t_general_dec, t_object_dec, t_anom_det, t_score
    # Total_RX uses max(general, object) to reflect parallel decode.
    if not args.skip_mel:
        wav_path, _is_tmp = _resolve_wav_for_mel(
            args.wav,
            sample_rate=int(args.sample_rate),
            clip_seconds=float(args.clip_seconds),
        )
        mel_transform = make_mel_spectrogram(sample_rate=int(args.sample_rate))
        to_db = amplitude_to_db_power()

        # TX load + spectrogram (CPU)
        tx_load_ms = _measure_latency(
            lambda: load_wav(wav_path, sample_rate=int(args.sample_rate)),
            device=torch.device("cpu"),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        wav0, _ = load_wav(wav_path, sample_rate=int(args.sample_rate))
        tx_spec_ms = _measure_latency(
            lambda: wav_to_mel(
                wav0, target_T=target_T, mel_transform=mel_transform, to_db=to_db
            ),
            device=torch.device("cpu"),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        mel0 = wav_to_mel(wav0, target_T=target_T, mel_transform=mel_transform, to_db=to_db)
        x_mel_tx = mel0.unsqueeze(0).to(tx_device)
        tx_payload_ms = _measure_latency(
            lambda: vq_tx.encode_to_indices(x_mel_tx),
            device=tx_device,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        tx_total_ms = tx_load_ms + tx_spec_ms + tx_payload_ms

        with torch.inference_mode():
            idx_c_tx, idx_f_tx = vq_tx.encode_to_indices(x_mel_tx)

        # Move shared weights to RX device for receiver profiling.
        if rx_device.type != "cpu":
            vq_vae = vq_vae.to(rx_device)
            model = model.to(rx_device)
        idx_c_rx = idx_c_tx.to(rx_device)
        idx_f_rx = idx_f_tx.to(rx_device)

        rx_decode_latents_ms = _measure_latency(
            lambda: vq_vae.indices_to_quantized(idx_c_rx, idx_f_rx),
            device=rx_device,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        with torch.inference_mode():
            q_fine, q_coarse = vq_vae.indices_to_quantized(idx_c_rx, idx_f_rx)

        # Reuse existing profiled modules for decoder/head timings by measuring runtime only here.
        vq_rx_gen = VQVAE_RX_GeneralDecoder(vq_vae).eval()
        obj_dec = model._object_decoder
        det = model._anomaly_detection

        rx_general_ms = _measure_latency(
            lambda: vq_rx_gen.decode_general(q_fine, q_coarse),
            device=rx_device,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        rx_object_ms = _measure_latency(
            lambda: obj_dec(q_coarse, q_fine, vq_vae._vq_coarse, vq_vae._vq_fine, return_aux=False),
            device=rx_device,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        with torch.inference_mode():
            x_g = vq_rx_gen.decode_general(q_fine, q_coarse)
            x_s = obj_dec(q_coarse, q_fine, vq_vae._vq_coarse, vq_vae._vq_fine, return_aux=False)

        rx_anom_ms = _measure_latency(
            lambda: det(x_s.detach(), x_g.detach()),
            device=rx_device,
            warmup=args.warmup,
            repeats=args.repeats,
        )

        def _score_fn():
            with torch.inference_mode():
                m_out = det(x_s.detach(), x_g.detach())
                probs = torch.softmax(m_out, dim=1)
                _ = probs[:, 1].reshape(m_out.shape[0], -1).mean(dim=1).item()

        rx_score_ms = _measure_latency(
            _score_fn,
            device=rx_device,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        rx_total_ms = rx_decode_latents_ms + max(rx_general_ms, rx_object_ms) + rx_anom_ms + rx_score_ms

        print("\n=== Latent TX/RX (definition-aligned) ===")
        print(f"TX: tx_t_load_wav             : {tx_load_ms:.3f} ms")
        print(f"TX: tx_t_compute_spectrogram  : {tx_spec_ms:.3f} ms")
        print(f"TX: tx_t_to_payload           : {tx_payload_ms:.3f} ms  (encode_to_indices)")
        print(f"TX: tx_total                  : {tx_total_ms:.3f} ms")
        print(f"RX: t_decode_to_latents       : {rx_decode_latents_ms:.3f} ms  (indices_to_quantized)")
        print(f"RX: t_general_dec             : {rx_general_ms:.3f} ms")
        print(f"RX: t_object_dec              : {rx_object_ms:.3f} ms")
        print(f"RX: t_anom_det                : {rx_anom_ms:.3f} ms")
        print(f"RX: t_score_ms                : {rx_score_ms:.3f} ms")
        print(f"RX: rx_total                  : {rx_total_ms:.3f} ms  (uses max(general, object))")

    # Move shared VQ-VAE + sDSR to RX device for receiver profiling.
    if rx_device.type != "cpu":
        vq_vae = vq_vae.to(rx_device)
        model = model.to(rx_device)

    B = int(args.batch)
    x_rx = torch.randn(B, 1, n_mels, target_T, device=rx_device)

    # Prepare rx inputs (indices + quantized tensors).
    with torch.inference_mode():
        idx_c, idx_f = vq_vae.encode_to_indices(x_rx)
        q_fine, q_coarse = vq_vae.indices_to_quantized(idx_c, idx_f)

    print(f"\n=== RX modules ({rx_device}, batch={B}) ===")

    # -------- RX: general decoder-only
    vq_rx_gen = VQVAE_RX_GeneralDecoder(vq_vae).eval()

    def rx_general_fn():
        with torch.inference_mode():
            _ = vq_rx_gen.decode_general(q_fine, q_coarse)

    profile_module(
        "RX general decoder-only (upscale + DecoderFine)",
        vq_rx_gen,
        vq_rx_gen.state_dict(),
        rx_general_fn,
        device=rx_device,
        warmup=args.warmup,
        repeats=args.repeats,
    )

    # -------- RX: object-specific decoder
    obj_dec = model._object_decoder

    def rx_object_fn():
        with torch.inference_mode():
            _ = obj_dec(q_coarse, q_fine, vq_vae._vq_coarse, vq_vae._vq_fine, return_aux=False)

    profile_module(
        "RX object-specific decoder (ObjectSpecificDecoder)",
        obj_dec,
        obj_dec.state_dict(),
        rx_object_fn,
        device=rx_device,
        warmup=args.warmup,
        repeats=args.repeats,
    )

    if args.include_detector:
        det = model._anomaly_detection
        with torch.inference_mode():
            x_g = vq_vae.decode_general(q_fine, q_coarse)
            x_s = obj_dec(q_coarse, q_fine, vq_vae._vq_coarse, vq_vae._vq_fine, return_aux=False)

        def rx_det_fn():
            with torch.inference_mode():
                _ = det(x_s, x_g)

        profile_module(
            "RX anomaly detector (AnomalyDetectionModule)",
            det,
            det.state_dict(),
            rx_det_fn,
            device=rx_device,
            warmup=args.warmup,
            repeats=args.repeats,
        )

    if not args.skip_mel:
        print(
            "\n=== TX edge inference budget (CPU, batch=1, approximate) ===\n"
            f"  Use load+mel e2e + encoder latency from sections above "
            f"(encoder: {tx_fp.latency_ms:.3f} ms)."
        )


if __name__ == "__main__":
    main()

