#!/usr/bin/env python3
"""
Evaluate sDSR under a BER-calibrated BSC channel on Stage-1 latent index maps.

Pipeline:
1) Deterministic fixed-length packing of coarse/fine indices (no entropy coding)
2) Bit flips with probability BER(SNR) from a calibration CSV (BSC approximation)
3) Fixed-length unpacking back to indices, then Stage-2 evaluation
"""

from __future__ import annotations

import argparse
import csv
import math
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

def _bits_required(K: int) -> int:
    if K <= 1:
        return 1
    return int(math.ceil(math.log2(float(K))))


def _pack_symbols_fixed_lsb(symbols: np.ndarray, bits_per_symbol: int) -> tuple[np.ndarray, int]:
    """
    Pack integer symbols into bytes using LSB-first bit order per symbol.
    Returns (payload_bytes uint8 array, n_bits valid).
    """
    symbols = np.asarray(symbols, dtype=np.int64).reshape(-1)
    B = int(bits_per_symbol)
    if B <= 0:
        raise ValueError(f"bits_per_symbol must be > 0, got {B}")
    # (N,B) bits, LSB-first along B
    bitmat = ((symbols[:, None] >> np.arange(B, dtype=np.int64)[None, :]) & 1).astype(np.uint8)
    bits = bitmat.reshape(-1)
    n_bits = int(bits.size)
    payload = np.packbits(bits, bitorder="little")
    return payload.astype(np.uint8), n_bits


def _unpack_symbols_fixed_lsb(payload_bytes: np.ndarray, n_bits: int, n_symbols: int, bits_per_symbol: int) -> np.ndarray:
    payload_bytes = np.asarray(payload_bytes, dtype=np.uint8).reshape(-1)
    B = int(bits_per_symbol)
    if n_bits != n_symbols * B:
        raise ValueError(f"n_bits mismatch: got {n_bits}, expected {n_symbols * B}")
    bits = np.unpackbits(payload_bytes, bitorder="little")[:n_bits].astype(np.uint8)
    bitmat = bits.reshape(int(n_symbols), B).astype(np.int64)
    weights = (1 << np.arange(B, dtype=np.int64))[None, :]
    syms = (bitmat * weights).sum(axis=1)
    return syms.astype(np.int64)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--stage1_ckpt", type=str, required=True)
    p.add_argument("--stage2_ckpt", type=str, required=True)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--machine_type", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--pauc_max_fpr", type=float, default=0.1)
    p.add_argument("--snr_db", type=float, nargs="+", default=[0, 2, 4, 6, 8, 10, 12, 15, 20])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--method", type=str, choices=["bitstream"], default="bitstream")
    p.add_argument("--bits_coarse", type=int, default=None, help="Fixed-length bits/index for coarse map.")
    p.add_argument("--bits_fine", type=int, default=None, help="Fixed-length bits/index for fine map.")
    p.add_argument("--ber_curve", type=str, required=True, help="CSV with columns snr_db, ber_postfec from calibrate_ldpc_bpsk_ber.py.")

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


class AWGNIndexChannelWrapper(nn.Module):
    """
    Wrap an sDSR model to inject an index-map channel before inference.
    Forward signature matches evaluator: forward(x) -> m_out.
    """

    def __init__(
        self,
        model: sDSR,
        vq_vae: VQ_VAE_2Layer,
        *,
        device: torch.device,
        bits_coarse: int | None,
        bits_fine: int | None,
        ber_curve_path: str,
        seed: int,
        snr_db: float,
    ) -> None:
        super().__init__()
        self.model = model
        self.vq_vae = vq_vae
        self.device = device
        self._bits_arg_c = bits_coarse
        self._bits_arg_f = bits_fine
        self.ber_curve = load_ber_curve_csv(ber_curve_path)
        self.seed = seed
        self.snr_db = snr_db

        # Fixed-length symbol widths (computed from codebook sizes unless provided).
        Kc = int(self.vq_vae.num_embeddings_coarse)
        Kf = int(self.vq_vae.num_embeddings_fine)
        self._bits_c = int(self._bits_arg_c) if self._bits_arg_c is not None else int(_bits_required(Kc))
        self._bits_f = int(self._bits_arg_f) if self._bits_arg_f is not None else int(_bits_required(Kf))

        # Accounting: BSC on coded bitstream proxy => 1 bit/channel use.
        self._cu_coarse_total = 0
        self._cu_fine_total = 0
        self._n_samples_total = 0
        self._clip_oor_coarse = 0
        self._clip_oor_fine = 0

    def avg_channel_uses_per_clip(self) -> tuple[float, float, float]:
        if self._n_samples_total <= 0:
            return (0.0, 0.0, 0.0)
        c = self._cu_coarse_total / self._n_samples_total
        f = self._cu_fine_total / self._n_samples_total
        return (float(c), float(f), float(c + f))

    def fixedlen_clip_counts(self) -> tuple[int, int]:
        """(coarse_out_of_range, fine_out_of_range) counts before clipping."""
        return (int(self._clip_oor_coarse), int(self._clip_oor_fine))

    def _encode_indices_to_bits(self, idx_c: np.ndarray, idx_f: np.ndarray) -> tuple[np.ndarray, int, np.ndarray, int]:
        b_c, nbc = _pack_symbols_fixed_lsb(idx_c, self._bits_c)
        b_f, nbf = _pack_symbols_fixed_lsb(idx_f, self._bits_f)
        return b_c, nbc, b_f, nbf

    def _decode_bits_to_indices(self, b_c: np.ndarray, nbc: int, b_f: np.ndarray, nbf: int, n_c: int, n_f: int) -> tuple[np.ndarray, np.ndarray]:
        idx_c = _unpack_symbols_fixed_lsb(b_c, nbc, n_c, self._bits_c)
        idx_f = _unpack_symbols_fixed_lsb(b_f, nbf, n_f, self._bits_f)
        # Clip to valid codebook range (important if bits_per_symbol > ceil(log2(K)) or errors occur).
        Kc = int(self.vq_vae.num_embeddings_coarse)
        Kf = int(self.vq_vae.num_embeddings_fine)
        oor_c = int(np.count_nonzero((idx_c < 0) | (idx_c >= Kc)))
        oor_f = int(np.count_nonzero((idx_f < 0) | (idx_f >= Kf)))
        self._clip_oor_coarse += oor_c
        self._clip_oor_fine += oor_f
        if oor_c:
            idx_c = np.clip(idx_c, 0, Kc - 1)
        if oor_f:
            idx_f = np.clip(idx_f, 0, Kf - 1)
        return idx_c, idx_f

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        self.vq_vae.eval()
        ber = float(self.ber_curve.ber_at(self.snr_db))

        with torch.inference_mode():
            idx_c_t, idx_f_t = self.vq_vae.encode_to_indices(x.to(self.device))
            # flatten to symbols for entropy coding
            idx_c = idx_c_t.detach().cpu().numpy().astype(np.int64).reshape(x.shape[0], -1)
            idx_f = idx_f_t.detach().cpu().numpy().astype(np.int64).reshape(x.shape[0], -1)

            rx_idx_c = []
            rx_idx_f = []
            cu_coarse = 0
            cu_fine = 0
            for i in range(x.shape[0]):
                b_c, nbc, b_f, nbf = self._encode_indices_to_bits(idx_c[i], idx_f[i])
                payload_c = bytes(np.asarray(b_c, dtype=np.uint8).tobytes())
                payload_f = bytes(np.asarray(b_f, dtype=np.uint8).tobytes())
                rx_payload_c = bitflip_bytes(
                    payload_c,
                    ber=ber,
                    protect_bytes=0,
                    seed=int(self.seed) ^ int(i) ^ (int(self.snr_db * 100) & 0xFFFF) ^ 0xC0A5E,
                )
                rx_payload_f = bitflip_bytes(
                    payload_f,
                    ber=ber,
                    protect_bytes=0,
                    seed=int(self.seed) ^ int(i) ^ (int(self.snr_db * 100) & 0xFFFF) ^ 0xF1A9E,
                )
                rx_b_c = np.frombuffer(rx_payload_c, dtype=np.uint8)
                rx_b_f = np.frombuffer(rx_payload_f, dtype=np.uint8)

                dec_c, dec_f = self._decode_bits_to_indices(rx_b_c, nbc, rx_b_f, nbf, idx_c.shape[1], idx_f.shape[1])
                rx_idx_c.append(dec_c)
                rx_idx_f.append(dec_f)
                cu_coarse += int(nbc)
                cu_fine += int(nbf)

            rx_idx_c = np.stack(rx_idx_c, axis=0)
            rx_idx_f = np.stack(rx_idx_f, axis=0)
            self._cu_coarse_total += int(cu_coarse)
            self._cu_fine_total += int(cu_fine)
            self._n_samples_total += int(x.shape[0])

            # reshape back to (B,H,W)
            B, Hc, Wc = idx_c_t.shape
            _, Hf, Wf = idx_f_t.shape
            rx_idx_c_t = torch.from_numpy(rx_idx_c.reshape(B, Hc, Wc)).long().to(self.device)
            rx_idx_f_t = torch.from_numpy(rx_idx_f.reshape(B, Hf, Wf)).long().to(self.device)

            q_fine, q_coarse = self.vq_vae.indices_to_quantized(rx_idx_c_t, rx_idx_f_t)
            # NOTE: indices_to_quantized returns (q_fine, q_coarse) in this repo
            out = self.model.forward_from_quantized(q_fine=q_fine, q_coarse=q_coarse)
            # Some model variants may return (m_out, aux1, aux2); evaluator expects a Tensor.
            m_out: torch.Tensor = out[0] if isinstance(out, tuple) else out
            return m_out


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
    vq_vae = vq_vae.to(device)

    out_path = Path(args.output) if args.output else (Path(args.stage2_ckpt).resolve().parent / "results" / "awgn_results.csv")
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
                wrapper = AWGNIndexChannelWrapper(
                    model=model,
                    vq_vae=vq_vae,
                    device=device,
                    bits_coarse=args.bits_coarse,
                    bits_fine=args.bits_fine,
                    ber_curve_path=args.ber_curve,
                    seed=seed,
                    snr_db=float(snr_db),
                )
                evaluator = AnomalyEvaluator(
                    model=wrapper,
                    test_dataset=test_ds,
                    device=device,
                    pauc_max_fpr=args.pauc_max_fpr,
                    batch_size=args.batch_size,
                    train_score_stats=None,
                    train_score_stats_fallback=None,
                )
                res = evaluator.evaluate()
                cu_c, cu_f, cu_t = wrapper.avg_channel_uses_per_clip()
                oor_c, oor_f = wrapper.fixedlen_clip_counts()
                ids = res.get(args.machine_type, {})
                for mid, v in ids.items():
                    if not isinstance(v, dict):
                        continue
                    w.writerow(
                        [
                            args.machine_type,
                            args.method,
                            "bsc_ber_curve",
                            snr_db,
                            seed,
                            NAN,
                            NAN,
                            NAN,
                            args.ber_curve,
                            NAN,
                            f"{cu_c:.2f}",
                            f"{cu_f:.2f}",
                            f"{cu_t:.2f}",
                            "bits",
                            NAN,
                            NAN,
                            NAN,
                            oor_c,
                            oor_f,
                            mid,
                            v["auc"],
                            v["pauc"],
                        ]
                    )
                f.flush()
                print(
                    f"[{args.machine_type}] method={args.method} channel=bsc_ber_curve "
                    f"snr={snr_db} seed={seed} "
                    f"cu_total={cu_t:.1f}/clip decode_ok=nan decode_fail=nan ok_rate=nan "
                    f"clip_oor_c={oor_c} clip_oor_f={oor_f} avg={ids.get('average')}"
                )

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

