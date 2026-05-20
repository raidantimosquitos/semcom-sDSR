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
    --device cuda --batch 1
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

import torch

from src.models.sDSR.s_dsr import sDSR, sDSRConfig
from src.models.vq_vae.autoencoders import VQ_VAE_2Layer


@dataclass
class Footprint:
    params_total: int
    params_trainable: int
    state_dict_mb: float
    peak_cuda_mb: float | None
    latency_ms: float
    flops: float | None


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


def build_models(args: argparse.Namespace, device: torch.device) -> tuple[sDSR, VQ_VAE_2Layer, int, int]:
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
    vq_vae = vq_vae.to(device).eval()

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
    model = model.to(device).eval()
    return model, vq_vae, n_mels, target_T


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
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument(
        "--include_detector",
        action="store_true",
        help="Also profile receiver anomaly detection head (x_specific/x_general -> m_out).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, vq_vae, n_mels, target_T = build_models(args, device)

    B = int(args.batch)
    x = torch.randn(B, 1, n_mels, target_T, device=device)

    # -------- TX: encode_to_indices (includes quantization + index extraction)
    def tx_fn():
        with torch.inference_mode():
            _ = vq_vae.encode_to_indices(x)

    profile_module(
        "TX encoder (VQ_VAE_2Layer.encode_to_indices)",
        vq_vae,
        vq_vae.state_dict(),
        tx_fn,
        device=device,
        warmup=args.warmup,
        repeats=args.repeats,
    )

    # Prepare rx inputs (indices + quantized tensors).
    with torch.inference_mode():
        idx_c, idx_f = vq_vae.encode_to_indices(x)
        q_fine, q_coarse = vq_vae.indices_to_quantized(idx_c, idx_f)

    # -------- RX: general decoder
    def rx_general_fn():
        with torch.inference_mode():
            _ = vq_vae.decode_general(q_fine, q_coarse)

    profile_module(
        "RX general decoder (VQ_VAE_2Layer.decode_general)",
        vq_vae,
        vq_vae.state_dict(),
        rx_general_fn,
        device=device,
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
        device=device,
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
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
        )


if __name__ == "__main__":
    main()

