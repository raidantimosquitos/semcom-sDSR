#!/usr/bin/env python3
"""
Decode received bitstream (after FEC decode in GNURadio) and evaluate AUC/pAUC
per machine_id using the pretrained General Decoder + Object-specific Decoder.
Runs on the receiver computer.

Usage:
  python scripts/receive_and_evaluate.py --stage1_ckpt ... --stage2_ckpt ... \\
    --data_path /path/to/dcase --machine_type fan --input_bitstream decoded.bin

  One FEC-decoded file per clip (same stems as test wavs, from transmit_indices.py --output_dir):
  python scripts/receive_and_evaluate.py ... --input_bitstream_dir /path/to/decoded_bins/
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import math
import re
import torch

from src.data.dataset import (
    DCASE2020Task2LogMelDataset,
    DCASE2020Task2TestDataset,
)
from src.engine.evaluator import _partial_auc
from src.models.vq_vae.autoencoders import VQ_VAE_2Layer
from src.models.sDSR.s_dsr import sDSR, sDSRConfig
from src.utils.stage1_norm import load_norm_from_stage1_ckpt
from src.utils.bitstream import (
    frame_size_bytes,
    read_frame_file,
    read_frames_from_raw_stream,
    unpack_frame_to_indices,
)

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Decode bitstream and evaluate AUC/pAUC per machine_id.")
    p.add_argument("--stage1_ckpt", type=str, required=True, help="Stage 1 checkpoint")
    p.add_argument("--stage2_ckpt", type=str, required=True, help="Stage 2 checkpoint")
    p.add_argument("--data_path", type=str, required=True, help="Path to DCASE dataset root")
    p.add_argument("--machine_type", type=str, default="fan")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--input_bitstream",
        type=str,
        default=None,
        help="Single raw file containing concatenated frames (no headers)",
    )
    g.add_argument(
        "--input_bitstream_dir",
        type=str,
        default=None,
        help="Directory of one .bin per clip (same stem as test wav, e.g. normal_id_01_00000000.bin)",
    )
    p.add_argument("--output", type=str, default=None, help="CSV results path")
    p.add_argument("--pauc_max_fpr", type=float, default=0.1)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()

_STEM_RE = re.compile(r"^(normal|anomaly)_(id_\d{2})_.*$")


def _parse_label_and_id_from_stem(stem: str) -> tuple[int, str]:
    """
    Expected stems (from DCASE2020 Task2): '{normal|anomaly}_id_XX_YYYYYYYY'
    Returns: (label, machine_id) where label is 0 (normal) or 1 (anomaly).
    """
    m = _STEM_RE.match(stem)
    if not m:
        raise ValueError(f"Unrecognized bitstream filename stem '{stem}'. Expected like 'normal_id_00_00000000'.")
    label = 0 if m.group(1) == "normal" else 1
    machine_id = m.group(2)
    return label, machine_id


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load Stage 1 checkpoint for norm stats and model weights
    ckpt1 = torch.load(args.stage1_ckpt, map_location="cpu", weights_only=True)
    norm_mean, norm_std = load_norm_from_stage1_ckpt(ckpt1)
    train_ds = DCASE2020Task2LogMelDataset(
        root=args.data_path,
        machine_type=args.machine_type,
    )
    _, _, n_mels, T = train_ds.data.shape
    test_ds = DCASE2020Task2TestDataset(
        root=args.data_path,
        machine_type=args.machine_type,
        target_T=train_ds.target_T,
    )

    # Load Stage 1 + Stage 2 (full sDSR); arch params from checkpoint
    from src.utils.checkpoint_compat import migrate_vq_vae_state_dict
    num_embeddings_coarse = ckpt1.get("num_embeddings_coarse", ckpt1.get("num_embeddings_top"))
    num_embeddings_fine = ckpt1.get("num_embeddings_fine", ckpt1.get("num_embeddings_bot"))
    embedding_dim = ckpt1["embedding_dim"]
    hidden_channels = ckpt1["hidden_channels"]
    num_residual_layers = ckpt1.get("num_residual_layers", 2)
    vq_vae = VQ_VAE_2Layer(
        hidden_channels=hidden_channels,
        num_residual_layers=num_residual_layers,
        num_embeddings=(num_embeddings_coarse, num_embeddings_fine),
        embedding_dim=embedding_dim,
        commitment_cost=0.25,
        decay=0.99,
    )
    state1 = dict(ckpt1["model_state_dict"])
    migrate_vq_vae_state_dict(state1)
    vq_vae.load_state_dict(state1)

    bits_coarse = int(math.log2(num_embeddings_coarse))
    bits_fine = int(math.log2(num_embeddings_fine))
    # Fine x2/x4 down, coarse x8/x16 down from spectrogram
    H_fine = max(1, n_mels // 2)
    W_fine = max(1, T // 4)
    H_coarse = max(1, n_mels // 8)
    W_coarse = max(1, T // 16)
    frame_sz = frame_size_bytes(H_coarse, W_coarse, H_fine, W_fine, bits_coarse, bits_fine)

    if args.input_bitstream_dir is not None:
        frames: list[bytes] = []
        labels: list[int] = []
        machine_ids: list[str] = []
        in_dir = Path(args.input_bitstream_dir)
        bin_files = sorted(p for p in in_dir.iterdir() if p.is_file() and p.suffix == ".bin")
        if not bin_files:
            raise FileNotFoundError(f"No .bin files found under {in_dir}")
        for p in bin_files:
            stem = p.stem
            label, machine_id = _parse_label_and_id_from_stem(stem)
            frames.append(read_frame_file(str(p), expected_frame_size=frame_sz))
            labels.append(label)
            machine_ids.append(machine_id)
        n_eval = len(frames)
    else:
        frames = read_frames_from_raw_stream(args.input_bitstream, frame_sz)
        if len(frames) != len(test_ds):
            print(f"Warning: bitstream has {len(frames)} frames, test set has {len(test_ds)}. Using min for alignment.")
        n_eval = min(len(frames), len(test_ds))
    
    cfg = sDSRConfig(
        embedding_dim=embedding_dim,
        hidden_channels=hidden_channels,
        n_mels=n_mels,
        T=T,
    )
    model = sDSR(vq_vae, cfg)
    ckpt2 = torch.load(args.stage2_ckpt, map_location="cpu", weights_only=True)
    state2 = dict(ckpt2["model_state_dict"])
    migrate_vq_vae_state_dict(state2)
    model.load_state_dict(state2)
    model = model.to(device)
    model.eval()

    # Per-clip (score, label, machine_id) in test_ds order
    scores_by_id: dict[str, list[tuple[float, int]]] = defaultdict(list)

    with torch.no_grad():
        for i in range(n_eval):
            if args.input_bitstream_dir is not None:
                label = labels[i]
                machine_id = machine_ids[i]
            else:
                _, label, machine_id = test_ds[i]
            frame = frames[i]
            indices_coarse, indices_fine = unpack_frame_to_indices(
                frame, H_coarse, W_coarse, H_fine, W_fine,
                bits_coarse=bits_coarse, bits_fine=bits_fine, device=device,
            )
            q_fine, q_coarse = vq_vae.indices_to_quantized(indices_coarse, indices_fine)
            out = model.forward_from_quantized(q_fine, q_coarse)
            m_out: torch.Tensor = out[0] if isinstance(out, tuple) else out
            # Anomaly score: mean over anomaly channel (same as AnomalyEvaluator)
            logits = m_out[:, 1]
            score = logits.view(-1).mean().item()
            mid = str(machine_id)
            scores_by_id[mid].append((score, label))

    # AUC / pAUC per machine_id
    machine_type = test_ds.machine_type
    result: dict[str, dict[str, dict[str, float]]] = {machine_type: {}}
    for mid in sorted(scores_by_id.keys()):
        pairs = scores_by_id[mid]
        y_true = [p[1] for p in pairs]
        y_score = [p[0] for p in pairs]
        auc_val = float(roc_auc_score(y_true, y_score)) if roc_auc_score else float("nan")
        pauc_val = float(_partial_auc(y_true, y_score, args.pauc_max_fpr))
        result[machine_type][mid] = {"auc": auc_val, "pauc": pauc_val}

    ids = [k for k in result[machine_type].keys()]
    n = len(ids)
    result[machine_type]["average"] = {
        "auc": sum(result[machine_type][mid]["auc"] for mid in ids) / n if n else float("nan"),
        "pauc": sum(result[machine_type][mid]["pauc"] for mid in ids) / n if n else float("nan"),
    }

    for mid, v in result[machine_type].items():
        print(f"  {mid}: AUC={v['auc']:.4f} pAUC={v['pauc']:.4f}")

    out_path = args.output
    if out_path is None:
        if args.input_bitstream_dir is not None:
            out_path = str(Path(args.input_bitstream_dir) / "receive_results.csv")
        else:
            out_path = str(Path(args.input_bitstream).parent / "receive_results.csv")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["machine_type", "machine_id", "AUC", "pAUC"])
        for mt, ids in result.items():
            for k, v in ids.items():
                if isinstance(v, dict):
                    w.writerow([mt, k, f"{v['auc']:.4f}", f"{v['pauc']:.4f}"])
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
