#!/usr/bin/env python3
"""
Evaluate sDSR under an AWGN channel on the Stage-1 latent index maps.

Two families:
1) bitstream-first: (optional Huffman) -> LDPC (pyldpc) -> QPSK -> AWGN -> decode -> indices
2) JSCC: (placeholder in this script; use training script to produce a JSCC model)

This script focuses on **per machine_type** evaluation (no joint multi-type sweep).
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

from src.data.dataset import DCASE2020Task2LogMelDataset, DCASE2020Task2TestDataset
from src.engine.evaluator import AnomalyEvaluator
from src.models.vq_vae.autoencoders import VQ_VAE_2Layer
from src.models.sDSR.s_dsr import sDSR, sDSRConfig
from src.utils.stage1_norm import load_norm_from_stage1_ckpt
from src.utils.checkpoint_compat import migrate_vq_vae_state_dict

from src.comm.huffman import build_huffman_from_counts, decode_symbols, encode_symbols
from src.comm.ldpc_pyldpc import (
    LDPCCode,
    LDPCConfig,
    ldpc_qpsk_awgn_roundtrip,
    make_ldpc_code,
)

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
    p.add_argument("--snr_db", type=float, nargs="+", default=[-5, 0, 5, 10, 15, 20])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])

    p.add_argument("--method", type=str, choices=["bitstream"], default="bitstream")
    p.add_argument("--entropy", type=str, choices=["none", "huffman"], default="huffman")
    p.add_argument("--bits_coarse", type=int, default=None, help="Fixed-length bits/index for coarse if entropy=none.")
    p.add_argument("--bits_fine", type=int, default=None, help="Fixed-length bits/index for fine if entropy=none.")

    p.add_argument("--uep", type=str, choices=["B1", "B2", "B3"], default="B2")
    p.add_argument("--ldpc_n", type=int, default=512)
    p.add_argument("--ldpc_dv", type=int, default=2)
    p.add_argument("--ldpc_dc", type=int, default=4)
    p.add_argument("--ldpc_maxiter", type=int, default=100)

    p.add_argument("--output", type=str, default=None)
    p.add_argument(
        "--smoke_test",
        action="store_true",
        help="Run a quick fixed-length (entropy=none) LDPC+QPSK round-trip sanity check and exit.",
    )
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


@dataclass(frozen=True)
class UEPPoint:
    Rc: float
    Rf: float


UEP_POINTS: dict[str, UEPPoint] = {
    "B1": UEPPoint(Rc=1 / 2, Rf=2 / 3),
    "B2": UEPPoint(Rc=1 / 3, Rf=1 / 2),
    "B3": UEPPoint(Rc=1 / 4, Rf=1 / 3),
}


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
        train_ds: DCASE2020Task2LogMelDataset,
        device: torch.device,
        entropy: str,
        bits_coarse: int | None,
        bits_fine: int | None,
        uep: UEPPoint,
        ldpc_cfg: LDPCConfig,
        seed: int,
        snr_db: float,
    ) -> None:
        super().__init__()
        self.model = model
        self.vq_vae = vq_vae
        self.device = device
        self.entropy = entropy
        self._bits_arg_c = bits_coarse
        self._bits_arg_f = bits_fine
        self.uep = uep
        self.ldpc_cfg = ldpc_cfg
        self.seed = seed
        self.snr_db = snr_db

        # Build Huffman codes from TRAIN distribution (per machine_type)
        # (Using encode_to_indices, so this is consistent with transmitted symbols.)
        if entropy == "huffman":
            counts_c, counts_f = self._count_indices(train_ds)
            self.huff_c = build_huffman_from_counts(counts_c)
            self.huff_f = build_huffman_from_counts(counts_f)
        else:
            self.huff_c = None
            self.huff_f = None

        # Fixed-length symbol widths for entropy=none (computed from codebook sizes unless provided).
        self._bits_c = None
        self._bits_f = None
        if entropy == "none":
            Kc = int(self.vq_vae.num_embeddings_coarse)
            Kf = int(self.vq_vae.num_embeddings_fine)
            self._bits_c = int(self._bits_arg_c) if self._bits_arg_c is not None else int(_bits_required(Kc))
            self._bits_f = int(self._bits_arg_f) if self._bits_arg_f is not None else int(_bits_required(Kf))

        # Two LDPC codes for UEP (coarse/fine) to realize (Rc, Rf) points.
        # Regular LDPC has approximate rate ~ 1 - d_v/d_c; we map the target points
        # to common (d_v, d_c) pairs and generate separate codes.
        self.ldpc_code_c, self.ldpc_code_f = self._make_uep_codes(ldpc_cfg, uep)

        # Accounting: channel-use counters (QPSK => 2 coded bits / channel use).
        self._cu_coarse_total = 0
        self._cu_fine_total = 0
        self._n_samples_total = 0
        self._huff_fail_coarse = 0
        self._huff_fail_fine = 0
        self._huff_fail_any = 0
        self._clip_oor_coarse = 0
        self._clip_oor_fine = 0

    def avg_channel_uses_per_clip(self) -> tuple[float, float, float]:
        if self._n_samples_total <= 0:
            return (0.0, 0.0, 0.0)
        c = self._cu_coarse_total / self._n_samples_total
        f = self._cu_fine_total / self._n_samples_total
        return (float(c), float(f), float(c + f))

    def huffman_failure_counts(self) -> tuple[int, int, int]:
        """(coarse_failures, fine_failures, any_failures) across all processed samples."""
        return (int(self._huff_fail_coarse), int(self._huff_fail_fine), int(self._huff_fail_any))

    def fixedlen_clip_counts(self) -> tuple[int, int]:
        """(coarse_out_of_range, fine_out_of_range) counts before clipping (entropy=none)."""
        return (int(self._clip_oor_coarse), int(self._clip_oor_fine))

    def _make_uep_codes(self, base: LDPCConfig, uep: UEPPoint) -> tuple[LDPCCode, LDPCCode]:
        # Map target rates to (d_v, d_c) pairs (rate ≈ 1 - d_v/d_c).
        # These are standard, fast-to-generate configs under pyldpc.
        def _round_up_to_multiple(n: int, m: int) -> int:
            if m <= 0:
                raise ValueError(f"m must be > 0, got {m}")
            if n <= 0:
                raise ValueError(f"n must be > 0, got {n}")
            return ((n + m - 1) // m) * m

        def cfg_for_rate(R: float) -> LDPCConfig:
            if abs(R - 1 / 2) < 1e-6:
                dc = 4
                return LDPCConfig(n=_round_up_to_multiple(base.n, dc), d_v=2, d_c=dc, maxiter=base.maxiter, seed=base.seed)
            if abs(R - 2 / 3) < 1e-6:
                dc = 6
                return LDPCConfig(n=_round_up_to_multiple(base.n, dc), d_v=2, d_c=dc, maxiter=base.maxiter, seed=base.seed)
            if abs(R - 1 / 3) < 1e-6:
                # Important: pyldpc requires (n % d_c == 0) for a regular LDPC matrix.
                # With default n=512, d_c=3 would fail (512 % 3 != 0).
                dc = 3
                return LDPCConfig(n=_round_up_to_multiple(base.n, dc), d_v=2, d_c=dc, maxiter=base.maxiter, seed=base.seed)
            if abs(R - 1 / 4) < 1e-6:
                dc = 4
                return LDPCConfig(n=_round_up_to_multiple(base.n, dc), d_v=3, d_c=dc, maxiter=base.maxiter, seed=base.seed)
            raise ValueError(f"Unsupported target LDPC rate: {R}")

        code_c = make_ldpc_code(cfg_for_rate(uep.Rc))
        code_f = make_ldpc_code(cfg_for_rate(uep.Rf))
        return code_c, code_f

    def _count_indices(self, train_ds: DCASE2020Task2LogMelDataset) -> tuple[np.ndarray, np.ndarray]:
        loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=False, num_workers=0)
        # infer K from vq_vae
        Kc = int(self.vq_vae.num_embeddings_coarse)
        Kf = int(self.vq_vae.num_embeddings_fine)
        counts_c = np.zeros(Kc, dtype=np.int64)
        counts_f = np.zeros(Kf, dtype=np.int64)
        self.vq_vae.eval()
        with torch.inference_mode():
            for x, _lbl, _mid in loader:
                x = x.to(self.device)
                idx_c, idx_f = self.vq_vae.encode_to_indices(x)
                flat_c = idx_c.reshape(-1).detach().cpu().numpy().astype(np.int64)
                flat_f = idx_f.reshape(-1).detach().cpu().numpy().astype(np.int64)
                counts_c += np.bincount(flat_c, minlength=Kc)
                counts_f += np.bincount(flat_f, minlength=Kf)
        return counts_c, counts_f

    def _encode_indices_to_bits(self, idx_c: np.ndarray, idx_f: np.ndarray) -> tuple[np.ndarray, int, np.ndarray, int]:
        if self.entropy == "huffman":
            assert self.huff_c is not None and self.huff_f is not None
            b_c, nbc = encode_symbols(self.huff_c, idx_c)
            b_f, nbf = encode_symbols(self.huff_f, idx_f)
            return b_c, nbc, b_f, nbf
        if self.entropy == "none":
            assert self._bits_c is not None and self._bits_f is not None
            b_c, nbc = _pack_symbols_fixed_lsb(idx_c, self._bits_c)
            b_f, nbf = _pack_symbols_fixed_lsb(idx_f, self._bits_f)
            return b_c, nbc, b_f, nbf
        raise ValueError(f"Unsupported entropy mode: {self.entropy}")

    def _decode_bits_to_indices(self, b_c: np.ndarray, nbc: int, b_f: np.ndarray, nbf: int, n_c: int, n_f: int) -> tuple[np.ndarray, np.ndarray]:
        if self.entropy == "none":
            assert self._bits_c is not None and self._bits_f is not None
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

        assert self.huff_c is not None and self.huff_f is not None
        fail_any = False
        try:
            idx_c = decode_symbols(self.huff_c, b_c, nbc, n_c)
        except ValueError:
            # Huffman is not error-resilient: a single bit error can desync the stream.
            # For sweep robustness, treat this as a decode failure and fall back to a safe default.
            idx_c = np.zeros(n_c, dtype=np.int64)
            self._huff_fail_coarse += 1
            fail_any = True
        try:
            idx_f = decode_symbols(self.huff_f, b_f, nbf, n_f)
        except ValueError:
            idx_f = np.zeros(n_f, dtype=np.int64)
            self._huff_fail_fine += 1
            fail_any = True
        if fail_any:
            self._huff_fail_any += 1
        return idx_c, idx_f

    def _ldpc_channel(self, bits: np.ndarray, *, code: LDPCCode, ebn0_db: float, rng: np.random.Generator) -> np.ndarray:
        return ldpc_qpsk_awgn_roundtrip(
            bits,
            code=code,
            ldpc_maxiter=self.ldpc_cfg.maxiter,
            ebn0_db=ebn0_db,
            rng=rng,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        self.vq_vae.eval()
        rng = np.random.default_rng(self.seed)

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

                bits_c = np.unpackbits(b_c, bitorder="little")[:nbc].astype(np.uint8)
                bits_f = np.unpackbits(b_f, bitorder="little")[:nbf].astype(np.uint8)

                rx_bits_c = self._ldpc_channel(bits_c, code=self.ldpc_code_c, ebn0_db=float(self.snr_db), rng=rng)
                rx_bits_f = self._ldpc_channel(bits_f, code=self.ldpc_code_f, ebn0_db=float(self.snr_db), rng=rng)

                # Channel uses for this sample under QPSK:
                # coded_bits ≈ ceil(N/k)*n, then /2 for QPSK
                k_c, n_c = int(self.ldpc_code_c.k), int(self.ldpc_code_c.n)
                k_f, n_f = int(self.ldpc_code_f.k), int(self.ldpc_code_f.n)
                blocks_c = (int(bits_c.size) + k_c - 1) // k_c
                blocks_f = (int(bits_f.size) + k_f - 1) // k_f
                cu_coarse += (blocks_c * n_c + 1) // 2
                cu_fine += (blocks_f * n_f + 1) // 2

                rx_b_c = np.packbits(rx_bits_c, bitorder="little")
                rx_b_f = np.packbits(rx_bits_f, bitorder="little")

                dec_c, dec_f = self._decode_bits_to_indices(rx_b_c, nbc, rx_b_f, nbf, idx_c.shape[1], idx_f.shape[1])
                rx_idx_c.append(dec_c)
                rx_idx_f.append(dec_f)

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

    if args.smoke_test:
        # Minimal sanity: fixed-length packing + LDPC round-trip at very high SNR should be lossless.
        if args.entropy != "none":
            raise ValueError("--smoke_test expects --entropy none")
        rng = np.random.default_rng(0)
        uep = UEP_POINTS[args.uep]
        base = LDPCConfig(n=args.ldpc_n, d_v=args.ldpc_dv, d_c=args.ldpc_dc, maxiter=args.ldpc_maxiter, seed=0)

        # Match the same UEP mapping logic used by the wrapper.
        def round_up(n: int, m: int) -> int:
            return ((n + m - 1) // m) * m

        def cfg_for_rate(R: float) -> LDPCConfig:
            if abs(R - 1 / 2) < 1e-6:
                dc = 4
                return LDPCConfig(n=round_up(base.n, dc), d_v=2, d_c=dc, maxiter=base.maxiter, seed=base.seed)
            if abs(R - 2 / 3) < 1e-6:
                dc = 6
                return LDPCConfig(n=round_up(base.n, dc), d_v=2, d_c=dc, maxiter=base.maxiter, seed=base.seed)
            if abs(R - 1 / 3) < 1e-6:
                dc = 3
                return LDPCConfig(n=round_up(base.n, dc), d_v=2, d_c=dc, maxiter=base.maxiter, seed=base.seed)
            if abs(R - 1 / 4) < 1e-6:
                dc = 4
                return LDPCConfig(n=round_up(base.n, dc), d_v=3, d_c=dc, maxiter=base.maxiter, seed=base.seed)
            raise ValueError(f"Unsupported target LDPC rate: {R}")

        code_c = make_ldpc_code(cfg_for_rate(uep.Rc))
        code_f = make_ldpc_code(cfg_for_rate(uep.Rf))

        Kc, Kf = 1024, 4096
        bits_c = int(args.bits_coarse) if args.bits_coarse is not None else _bits_required(Kc)
        bits_f = int(args.bits_fine) if args.bits_fine is not None else _bits_required(Kf)

        idx_c = rng.integers(0, Kc, size=16 * 40, dtype=np.int64)
        idx_f = rng.integers(0, Kf, size=32 * 80, dtype=np.int64)
        b_c, nbc = _pack_symbols_fixed_lsb(idx_c, bits_c)
        b_f, nbf = _pack_symbols_fixed_lsb(idx_f, bits_f)
        bits_c_vec = np.unpackbits(b_c, bitorder="little")[:nbc].astype(np.uint8)
        bits_f_vec = np.unpackbits(b_f, bitorder="little")[:nbf].astype(np.uint8)

        # Very high SNR (effectively noiseless)
        rx_bits_c = ldpc_qpsk_awgn_roundtrip(bits_c_vec, code=code_c, ldpc_maxiter=code_c.k, ebn0_db=60.0, rng=rng)
        rx_bits_f = ldpc_qpsk_awgn_roundtrip(bits_f_vec, code=code_f, ldpc_maxiter=code_f.k, ebn0_db=60.0, rng=rng)
        rx_b_c = np.packbits(rx_bits_c, bitorder="little")
        rx_b_f = np.packbits(rx_bits_f, bitorder="little")
        dec_c = _unpack_symbols_fixed_lsb(rx_b_c, nbc, idx_c.size, bits_c)
        dec_f = _unpack_symbols_fixed_lsb(rx_b_f, nbf, idx_f.size, bits_f)

        if not np.array_equal(idx_c, dec_c):
            raise RuntimeError("Smoke test failed: coarse indices mismatch")
        if not np.array_equal(idx_f, dec_f):
            raise RuntimeError("Smoke test failed: fine indices mismatch")
        print("Smoke test passed: fixed-length pack/unpack + LDPC round-trip is lossless at high SNR.")
        return

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    stage1_ckpt = torch.load(args.stage1_ckpt, map_location="cpu", weights_only=True)
    _norm_mean, _norm_std = load_norm_from_stage1_ckpt(stage1_ckpt)

    train_ds = DCASE2020Task2LogMelDataset(
        root=args.data_path,
        machine_type=args.machine_type,
        include_test=False,
    )
    test_ds = DCASE2020Task2TestDataset(
        root=args.data_path,
        machine_type=args.machine_type,
        target_T=train_ds.target_T,
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
    vq_vae = vq_vae.to(device)

    uep = UEP_POINTS[args.uep]
    ldpc_cfg = LDPCConfig(
        n=args.ldpc_n, d_v=args.ldpc_dv, d_c=args.ldpc_dc, maxiter=args.ldpc_maxiter, seed=0
    )

    out_path = Path(args.output) if args.output else (Path(args.stage2_ckpt).resolve().parent / "results" / "awgn_results.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["machine_type", "method", "entropy", "uep", "snr_db", "seed", "avg_cu_coarse", "avg_cu_fine", "avg_cu_total", "machine_id", "auc", "pauc"])

        for snr_db in args.snr_db:
            for seed in args.seeds:
                wrapper = AWGNIndexChannelWrapper(
                    model=model,
                    vq_vae=vq_vae,
                    train_ds=train_ds,
                    device=device,
                    entropy=args.entropy,
                    bits_coarse=args.bits_coarse,
                    bits_fine=args.bits_fine,
                    uep=uep,
                    ldpc_cfg=ldpc_cfg,
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
                hf_c, hf_f, hf_any = wrapper.huffman_failure_counts()
                oor_c, oor_f = wrapper.fixedlen_clip_counts()
                ids = res.get(args.machine_type, {})
                for mid, v in ids.items():
                    if not isinstance(v, dict):
                        continue
                    w.writerow([args.machine_type, args.method, args.entropy, args.uep, snr_db, seed, f"{cu_c:.2f}", f"{cu_f:.2f}", f"{cu_t:.2f}", mid, v["auc"], v["pauc"]])
                f.flush()
                print(
                    f"[{args.machine_type}] method={args.method} entropy={args.entropy} uep={args.uep} "
                    f"snr={snr_db} seed={seed} cu_total={cu_t:.1f}/clip "
                    f"huff_fail_any={hf_any} clip_oor_c={oor_c} clip_oor_f={oor_f} avg={ids.get('average')}"
                )

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

