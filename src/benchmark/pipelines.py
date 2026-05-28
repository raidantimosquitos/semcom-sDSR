"""Staged detection-latency pipelines (JPEG, OPUS, VQ-VAE-2 latents)."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from src.benchmark.timing import PipelineResult, StageTimer, sync_device
from src.comm.bitflip_ber import BERCycle, bitflip_bytes, load_ber_curve_csv
from src.comm.jpeg_payload import bitflip_jpeg_entropy_payload
from src.comm.ogg_payload import bitflip_ogg_payload_pages
from src.data.dataset import MEL_TIME_CROP
from src.models.sDSR.s_dsr import sDSR
from src.models.vq_vae.autoencoders import VQ_VAE_2Layer
from src.utils.audio import amplitude_to_db_power, make_mel_spectrogram, mel_db_to_finite

from scripts.evaluate_awgn_jpeg import _jpeg_decode, _jpeg_encode
from scripts.evaluate_awgn_opus import (
    opus_decode_bytes_ffmpeg,
    opus_encode_bytes_ffmpeg,
    resolve_ffmpeg_bin,
    wav_to_logmel,
)


def _bits_required(K: int) -> int:
    if K <= 1:
        return 1
    return int(math.ceil(math.log2(float(K))))


def _pack_symbols_fixed_lsb(symbols: np.ndarray, bits_per_symbol: int) -> tuple[np.ndarray, int]:
    symbols = np.asarray(symbols, dtype=np.int64).reshape(-1)
    B = int(bits_per_symbol)
    bitmat = ((symbols[:, None] >> np.arange(B, dtype=np.int64)[None, :]) & 1).astype(np.uint8)
    bits = bitmat.reshape(-1)
    n_bits = int(bits.size)
    payload = np.packbits(bits, bitorder="little")
    return payload.astype(np.uint8), n_bits


def _unpack_symbols_fixed_lsb(
    payload_bytes: np.ndarray, n_bits: int, n_symbols: int, bits_per_symbol: int
) -> np.ndarray:
    payload_bytes = np.asarray(payload_bytes, dtype=np.uint8).reshape(-1)
    B = int(bits_per_symbol)
    bits = np.unpackbits(payload_bytes, bitorder="little")[:n_bits].astype(np.uint8)
    bitmat = bits.reshape(int(n_symbols), B).astype(np.int64)
    weights = (1 << np.arange(B, dtype=np.int64))[None, :]
    return (bitmat * weights).sum(axis=1).astype(np.int64)


@dataclass
class PipelineConfig:
    tx_device: torch.device
    device: torch.device
    n_mels: int
    target_T: int
    sample_rate: int = 16_000
    seed: int = 0
    use_channel: bool = False
    snr_db: float = 20.0
    ber_curve: BERCycle | None = None
    jpeg_quality: int = 50
    jpeg_channel_mode: str = "jpeg_entropy"
    jpeg_protect_bytes: int = 8
    opus_kbps: int = 12
    opus_channel_mode: str = "ogg_pages"
    opus_protect_pages: int = 2
    opus_protect_bytes: int = 128
    ffmpeg_bin: str | None = None
    bits_coarse: int | None = None
    bits_fine: int | None = None


def load_wav(
    wav_path: str | Path,
    *,
    sample_rate: int,
) -> tuple[torch.Tensor, int]:
    """Load and resample wav to mono (CPU). Returns (1, n_samples), sr."""
    wav, sr = torchaudio.load(str(wav_path))
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
        sr = sample_rate
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    return wav, sr


def wav_to_mel(
    wav: torch.Tensor,
    *,
    target_T: int,
    mel_transform: nn.Module,
    to_db: nn.Module,
) -> torch.Tensor:
    """Log-mel with crop/pad (matches :class:`DCASE2020Task2TestDataset`)."""
    mel = mel_transform(wav)
    log_mel = mel_db_to_finite(to_db(mel).float())
    log_mel = log_mel[..., :MEL_TIME_CROP]
    T = log_mel.shape[-1]
    if target_T is not None and T < target_T:
        log_mel = F.pad(log_mel, (0, target_T - T), mode="constant", value=0.0)
    return log_mel


def load_wav_and_mel(
    wav_path: str | Path,
    *,
    sample_rate: int,
    target_T: int,
    mel_transform: nn.Module | None = None,
    to_db: nn.Module | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    mel_transform = mel_transform or make_mel_spectrogram(sample_rate=sample_rate)
    to_db = to_db or amplitude_to_db_power()
    wav, sr = load_wav(wav_path, sample_rate=sample_rate)
    mel = wav_to_mel(wav, target_T=target_T, mel_transform=mel_transform, to_db=to_db)
    return wav, mel, sr


def _anomaly_score(m_out: torch.Tensor) -> float:
    probs = torch.softmax(m_out, dim=1)
    return float(probs[:, 1].reshape(m_out.shape[0], -1).mean(dim=1).item())


def _run_sdsr_from_mel_with_breakdown(
    model: sDSR, x_mel: torch.Tensor, *, device: torch.device
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Run sDSR forward in the same semantics as model.forward(x), but return a timing breakdown.

    Returns:
        (m_out, {"t_enc_vqvae":.., "t_dec_general":.., "t_dec_object":.., "t_anom_det":..})
    """
    breakdown: dict[str, float] = {}
    with torch.inference_mode():
        sync_device(device)
        t0 = time.perf_counter()
        q_fine, q_coarse = model._vq_vae.encode(x_mel)  # type: ignore[attr-defined]
        sync_device(device)
        breakdown["t_enc_vqvae"] = time.perf_counter() - t0

        t1 = time.perf_counter()
        x_general = model._vq_vae.decode_general(q_fine, q_coarse)  # type: ignore[attr-defined]
        sync_device(device)
        breakdown["t_dec_general"] = time.perf_counter() - t1

        t2 = time.perf_counter()
        x_specific = model._object_decoder(  # type: ignore[attr-defined]
            q_coarse,
            q_fine,
            model._vq_vae._vq_coarse,  # type: ignore[attr-defined]
            model._vq_vae._vq_fine,  # type: ignore[attr-defined]
            return_aux=False,
        )
        sync_device(device)
        breakdown["t_dec_object"] = time.perf_counter() - t2

        t3 = time.perf_counter()
        m_out = model._anomaly_detection(  # type: ignore[attr-defined]
            x_specific.detach(), x_general.detach()
        )
        sync_device(device)
        breakdown["t_anom_det"] = time.perf_counter() - t3

    return m_out, breakdown


def _run_sdsr_from_quantized_with_breakdown(
    model: sDSR, *, q_fine: torch.Tensor, q_coarse: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Run sDSR receiver path from quantized latents (same semantics as forward_from_quantized),
    but return a timing breakdown.
    """
    breakdown: dict[str, float] = {}
    with torch.inference_mode():
        sync_device(device)
        t1 = time.perf_counter()
        x_general = model._vq_vae.decode_general(q_fine, q_coarse)  # type: ignore[attr-defined]
        sync_device(device)
        breakdown["t_dec_general"] = time.perf_counter() - t1

        t2 = time.perf_counter()
        x_specific = model._object_decoder(  # type: ignore[attr-defined]
            q_coarse,
            q_fine,
            model._vq_vae._vq_coarse,  # type: ignore[attr-defined]
            model._vq_vae._vq_fine,  # type: ignore[attr-defined]
            return_aux=False,
        )
        sync_device(device)
        breakdown["t_dec_object"] = time.perf_counter() - t2

        t3 = time.perf_counter()
        m_out = model._anomaly_detection(  # type: ignore[attr-defined]
            x_specific.detach(), x_general.detach()
        )
        sync_device(device)
        breakdown["t_anom_det"] = time.perf_counter() - t3

    return m_out, breakdown


def run_latent_pipeline(
    wav_path: str | Path,
    *,
    model: sDSR,
    vq_vae_tx: VQ_VAE_2Layer,
    vq_vae_rx: VQ_VAE_2Layer,
    cfg: PipelineConfig,
    mel_transform: nn.Module | None = None,
    to_db: nn.Module | None = None,
    preloaded: tuple[torch.Tensor, torch.Tensor, int] | None = None,
) -> PipelineResult:
    timer = StageTimer()
    tx_device = cfg.tx_device
    rx_device = cfg.device

    mel_transform = mel_transform or make_mel_spectrogram(sample_rate=cfg.sample_rate)
    to_db = to_db or amplitude_to_db_power()

    timer.start("t_load_wav")
    if preloaded is not None:
        wav, mel, _sr = preloaded
        timer.stop("t_load_wav")
        timer.start("t_mel")
        timer.stop("t_mel")
    else:
        wav, sr = load_wav(wav_path, sample_rate=cfg.sample_rate)
        timer.stop("t_load_wav")
        timer.start("t_mel")
        mel = wav_to_mel(wav, target_T=cfg.target_T, mel_transform=mel_transform, to_db=to_db)
        timer.stop("t_mel")

    x_tx = mel.unsqueeze(0).to(tx_device)
    sync_device(tx_device)

    Kc = int(vq_vae_tx.num_embeddings_coarse)
    Kf = int(vq_vae_tx.num_embeddings_fine)
    bits_c = int(cfg.bits_coarse) if cfg.bits_coarse is not None else _bits_required(Kc)
    bits_f = int(cfg.bits_fine) if cfg.bits_fine is not None else _bits_required(Kf)

    timer.start("t_tx_codec")
    with torch.inference_mode():
        idx_c_t, idx_f_t = vq_vae_tx.encode_to_indices(x_tx)
        sync_device(tx_device)
        idx_c = idx_c_t.detach().cpu().numpy().astype(np.int64).reshape(-1)
        idx_f = idx_f_t.detach().cpu().numpy().astype(np.int64).reshape(-1)
        b_c, nbc = _pack_symbols_fixed_lsb(idx_c, bits_c)
        b_f, nbf = _pack_symbols_fixed_lsb(idx_f, bits_f)
    timer.stop("t_tx_codec")

    payload_c = bytes(np.asarray(b_c, dtype=np.uint8).tobytes())
    payload_f = bytes(np.asarray(b_f, dtype=np.uint8).tobytes())
    payload_bytes = len(payload_c) + len(payload_f)

    timer.start("t_channel")
    if cfg.use_channel:
        if cfg.ber_curve is None:
            raise RuntimeError("--use_channel requires --ber_curve")
        ber = float(cfg.ber_curve.ber_at(cfg.snr_db))
        seed_mix = int(cfg.seed) ^ (int(cfg.snr_db * 100) & 0xFFFF)
        rx_payload_c = bitflip_bytes(payload_c, ber=ber, protect_bytes=0, seed=seed_mix ^ 0xC0A5E)
        rx_payload_f = bitflip_bytes(payload_f, ber=ber, protect_bytes=0, seed=seed_mix ^ 0xF1A9E)
    else:
        rx_payload_c, rx_payload_f = payload_c, payload_f
    timer.stop("t_channel")

    timer.start("t_rx_codec")
    rx_b_c = np.frombuffer(rx_payload_c, dtype=np.uint8)
    rx_b_f = np.frombuffer(rx_payload_f, dtype=np.uint8)
    n_c, n_f = idx_c.size, idx_f.size
    dec_c = _unpack_symbols_fixed_lsb(rx_b_c, nbc, n_c, bits_c)
    dec_f = _unpack_symbols_fixed_lsb(rx_b_f, nbf, n_f, bits_f)
    dec_c = np.clip(dec_c, 0, Kc - 1)
    dec_f = np.clip(dec_f, 0, Kf - 1)
    Hc, Wc = idx_c_t.shape[1], idx_c_t.shape[2]
    Hf, Wf = idx_f_t.shape[1], idx_f_t.shape[2]
    rx_idx_c_t = torch.from_numpy(dec_c.reshape(1, Hc, Wc)).long().to(rx_device)
    rx_idx_f_t = torch.from_numpy(dec_f.reshape(1, Hf, Wf)).long().to(rx_device)
    with torch.inference_mode():
        q_fine, q_coarse = vq_vae_rx.indices_to_quantized(rx_idx_c_t, rx_idx_f_t)
    sync_device(rx_device)
    timer.stop("t_rx_codec")

    timer.start("t_detector")
    m_out, rx_breakdown = _run_sdsr_from_quantized_with_breakdown(
        model, q_fine=q_fine, q_coarse=q_coarse, device=rx_device
    )
    timer.stop("t_detector")

    timer.start("t_score")
    score = _anomaly_score(m_out)
    timer.stop("t_score")

    extra = dict(rx_breakdown)
    extra["tx_device_is_cuda"] = 1.0 if tx_device.type == "cuda" else 0.0
    extra["rx_device_is_cuda"] = 1.0 if rx_device.type == "cuda" else 0.0
    return PipelineResult(
        times=timer.times,
        payload_bytes=payload_bytes,
        decode_ok=True,
        anomaly_score=score,
        extra=extra,
    )


def run_jpeg_pipeline(
    wav_path: str | Path,
    *,
    model: sDSR,
    cfg: PipelineConfig,
    mel_transform: nn.Module | None = None,
    to_db: nn.Module | None = None,
    preloaded: tuple[torch.Tensor, torch.Tensor, int] | None = None,
) -> PipelineResult:
    timer = StageTimer()
    device = cfg.device

    mel_transform = mel_transform or make_mel_spectrogram(sample_rate=cfg.sample_rate)
    to_db = to_db or amplitude_to_db_power()

    timer.start("t_load_wav")
    if preloaded is not None:
        _wav, mel, _sr = preloaded
        timer.stop("t_load_wav")
        timer.start("t_mel")
        timer.stop("t_mel")
    else:
        _wav, sr = load_wav(wav_path, sample_rate=cfg.sample_rate)
        timer.stop("t_load_wav")
        timer.start("t_mel")
        mel = wav_to_mel(_wav, target_T=cfg.target_T, mel_transform=mel_transform, to_db=to_db)
        timer.stop("t_mel")

    x = mel.unsqueeze(0)

    timer.start("t_tx_codec")
    blob = _jpeg_encode(x[0], quality=cfg.jpeg_quality)
    timer.stop("t_tx_codec")
    payload_bytes = len(blob)

    timer.start("t_channel")
    if cfg.use_channel:
        if cfg.ber_curve is None:
            raise RuntimeError("--use_channel requires --ber_curve")
        ber = float(cfg.ber_curve.ber_at(cfg.snr_db))
        seed = int(cfg.seed) ^ (int(cfg.snr_db * 100) & 0xFFFF)
        if cfg.jpeg_channel_mode == "jpeg_entropy":
            blob = bitflip_jpeg_entropy_payload(
                blob, ber=ber, prefix_protect_bytes=cfg.jpeg_protect_bytes, seed=seed,
            )
        else:
            blob = bitflip_bytes(blob, ber=ber, protect_bytes=cfg.jpeg_protect_bytes, seed=seed)
    timer.stop("t_channel")

    decode_ok = True
    timer.start("t_rx_codec")
    try:
        spec = _jpeg_decode(blob, n_mels=cfg.n_mels, T=cfg.target_T)
    except Exception:
        spec = torch.zeros((1, cfg.n_mels, cfg.target_T), dtype=torch.float32)
        decode_ok = False
    x_hat = spec.unsqueeze(0).to(device)
    timer.stop("t_rx_codec")

    timer.start("t_detector")
    m_out, rx_breakdown = _run_sdsr_from_mel_with_breakdown(model, x_hat, device=device)
    timer.stop("t_detector")

    timer.start("t_score")
    score = _anomaly_score(m_out)
    timer.stop("t_score")

    return PipelineResult(
        times=timer.times,
        payload_bytes=payload_bytes,
        decode_ok=decode_ok,
        anomaly_score=score,
        extra=rx_breakdown,
    )


def run_opus_pipeline(
    wav_path: str | Path,
    *,
    model: sDSR,
    cfg: PipelineConfig,
    mel_transform: nn.Module | None = None,
    to_db: nn.Module | None = None,
    preloaded: tuple[torch.Tensor, torch.Tensor, int] | None = None,
) -> PipelineResult:
    timer = StageTimer()
    device = cfg.device
    ffmpeg_bin = cfg.ffmpeg_bin or resolve_ffmpeg_bin(None)
    mel_transform = mel_transform or make_mel_spectrogram(sample_rate=cfg.sample_rate)
    to_db = to_db or amplitude_to_db_power()

    timer.start("t_load_wav")
    if preloaded is not None:
        wav, _mel, sr = preloaded
        timer.stop("t_load_wav")
    else:
        wav, sr = load_wav(wav_path, sample_rate=cfg.sample_rate)
        timer.stop("t_load_wav")

    timer.start("t_tx_codec")
    opus_blob = opus_encode_bytes_ffmpeg(wav, sr, kbps=cfg.opus_kbps, ffmpeg_bin=ffmpeg_bin)
    timer.stop("t_tx_codec")
    payload_bytes = len(opus_blob)

    timer.start("t_channel")
    if cfg.use_channel:
        if cfg.ber_curve is None:
            raise RuntimeError("--use_channel requires --ber_curve")
        ber = float(cfg.ber_curve.ber_at(cfg.snr_db))
        seed = int(cfg.seed) ^ (int(cfg.snr_db * 100) & 0xFFFF)
        if cfg.opus_channel_mode == "ogg_pages":
            opus_blob = bitflip_ogg_payload_pages(
                opus_blob, ber=ber, protect_first_pages=cfg.opus_protect_pages, seed=seed,
            )
        else:
            opus_blob = bitflip_bytes(
                opus_blob, ber=ber, protect_bytes=cfg.opus_protect_bytes, seed=seed,
            )
    timer.stop("t_channel")

    decode_ok = True
    t_rx0 = time.perf_counter()
    try:
        wav_d, sr_d = opus_decode_bytes_ffmpeg(opus_blob, ffmpeg_bin=ffmpeg_bin)
        timer.add("t_rx_codec", time.perf_counter() - t_rx0)
        t_mel0 = time.perf_counter()
        x_hat = wav_to_logmel(
            wav_d, sr_d, mel_transform, to_db,
            sample_rate=cfg.sample_rate, target_T=cfg.target_T,
        )
        timer.add("t_mel", time.perf_counter() - t_mel0)
    except Exception:
        timer.add("t_rx_codec", time.perf_counter() - t_rx0)
        x_hat = torch.zeros((1, cfg.n_mels, cfg.target_T), dtype=torch.float32)
        decode_ok = False
    x_hat = x_hat.unsqueeze(0).to(device)

    timer.start("t_detector")
    m_out, rx_breakdown = _run_sdsr_from_mel_with_breakdown(model, x_hat, device=device)
    timer.stop("t_detector")

    timer.start("t_score")
    score = _anomaly_score(m_out)
    timer.stop("t_score")

    return PipelineResult(
        times=timer.times,
        payload_bytes=payload_bytes,
        decode_ok=decode_ok,
        anomaly_score=score,
        extra=rx_breakdown,
    )


def run_pipeline(
    method: str,
    wav_path: str | Path,
    *,
    model: sDSR,
    vq_vae_tx: VQ_VAE_2Layer | None,
    vq_vae_rx: VQ_VAE_2Layer | None,
    cfg: PipelineConfig,
    mel_transform: nn.Module | None = None,
    to_db: nn.Module | None = None,
    preloaded: tuple[torch.Tensor, torch.Tensor, int] | None = None,
) -> PipelineResult:
    if method == "latent":
        if vq_vae_tx is None or vq_vae_rx is None:
            raise ValueError("vq_vae_tx and vq_vae_rx required for latent pipeline")
        return run_latent_pipeline(
            wav_path, model=model, vq_vae_tx=vq_vae_tx, vq_vae_rx=vq_vae_rx, cfg=cfg,
            mel_transform=mel_transform, to_db=to_db, preloaded=preloaded,
        )
    if method == "jpeg":
        return run_jpeg_pipeline(
            wav_path, model=model, cfg=cfg,
            mel_transform=mel_transform, to_db=to_db, preloaded=preloaded,
        )
    if method == "opus":
        return run_opus_pipeline(
            wav_path, model=model, cfg=cfg,
            mel_transform=mel_transform, to_db=to_db, preloaded=preloaded,
        )
    raise ValueError(f"Unknown method: {method}")


def make_pipeline_config(
    *,
    tx_device: torch.device,
    device: torch.device,
    n_mels: int,
    target_T: int,
    seed: int,
    use_channel: bool,
    snr_db: float,
    ber_curve_path: str | None,
    jpeg_quality: int,
    jpeg_channel_mode: str,
    jpeg_protect_bytes: int,
    opus_kbps: int,
    opus_channel_mode: str,
    opus_protect_pages: int,
    opus_protect_bytes: int,
    ffmpeg_bin: str | None,
    bits_coarse: int | None,
    bits_fine: int | None,
) -> PipelineConfig:
    ber_curve = load_ber_curve_csv(ber_curve_path) if (use_channel and ber_curve_path) else None
    return PipelineConfig(
        tx_device=tx_device,
        device=device,
        n_mels=n_mels,
        target_T=target_T,
        seed=seed,
        use_channel=use_channel,
        snr_db=snr_db,
        ber_curve=ber_curve,
        jpeg_quality=jpeg_quality,
        jpeg_channel_mode=jpeg_channel_mode,
        jpeg_protect_bytes=jpeg_protect_bytes,
        opus_kbps=opus_kbps,
        opus_channel_mode=opus_channel_mode,
        opus_protect_pages=opus_protect_pages,
        opus_protect_bytes=opus_protect_bytes,
        ffmpeg_bin=ffmpeg_bin,
        bits_coarse=bits_coarse,
        bits_fine=bits_fine,
    )
