#!/usr/bin/env python3
"""
JPEG sweep evaluation on DCASE2020 Task 2 (single machine_type).

For each Q in {10,20,...,90} (or a user-provided list), this script:
  1) loads the DCASE test spectrograms for one machine_type
  2) JPEG-compresses each spectrogram as a grayscale image (in-memory)
  3) decodes back to a spectrogram tensor
  4) runs the normal sDSR evaluation pipeline (AUC / pAUC per machine_id)

Usage:
  python scripts/jpeg_sweep_evaluate.py --stage1_ckpt ... --stage2_ckpt ... \\
    --data_path /path/to/dcase --machine_type fan
"""

from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from src.data.dataset import DCASE2020Task2LogMelDataset, DCASE2020Task2TestDataset
from src.engine.evaluator import AnomalyEvaluator
from src.models.sDSR.s_dsr import sDSR, sDSRConfig
from src.models.vq_vae.autoencoders import VQ_VAE_2Layer


def _parse_q_list(qs: Sequence[int] | None) -> list[int]:
    if qs is None or len(qs) == 0:
        return list(range(10, 100, 10))
    out: list[int] = []
    for q in qs:
        q_int = int(q)
        if q_int < 1 or q_int > 95:
            raise ValueError(f"JPEG quality must be in [1,95], got {q_int}")
        out.append(q_int)
    return out


class JPEGSpectrogramDataset(Dataset):
    """
    Wrap (x, label, machine_id) dataset and apply grayscale JPEG compression to x.

    x is expected shape: (1, n_mels, T) float32.
    """

    def __init__(self, base: Any, quality: int, clip: float = 4.0) -> None:
        self.base = base
        self.quality = int(quality)
        self.clip = float(clip)
        # pass-through attrs used by evaluator / logging
        self.machine_type = getattr(base, "machine_type", "unknown")
        self.machine_ids = getattr(base, "machine_ids", None)

        try:
            import PIL  # noqa: F401
            import numpy as np  # noqa: F401
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "JPEG sweep requires Pillow and numpy. Install with: pip install pillow numpy"
            ) from e

    def __len__(self) -> int:
        return len(self.base)

    def _encode_decode_jpeg(self, x: torch.Tensor) -> torch.Tensor:
        import numpy as np
        from PIL import Image

        if x.dim() != 3 or x.shape[0] != 1:
            raise ValueError(f"Expected x shape (1, n_mels, T), got {tuple(x.shape)}")

        # Map standardized float spectrogram to uint8 for grayscale JPEG.
        # Using a fixed symmetric clamp keeps the mapping consistent across samples.
        c = self.clip
        x2 = x.detach().cpu().float().clamp(-c, c)
        x01 = (x2 + c) / (2.0 * c)  # [0,1]
        img_u8 = (x01.squeeze(0).numpy() * 255.0).round().astype(np.uint8)  # (n_mels, T)

        img = Image.fromarray(img_u8, mode="L")
        buf = io.BytesIO()
        img.save(
            buf,
            format="JPEG",
            quality=self.quality,
            optimize=True,
            progressive=False,
            subsampling=0,
        )
        buf.seek(0)
        img_dec = Image.open(buf).convert("L")
        arr = np.asarray(img_dec).astype(np.float32) / 255.0  # (n_mels, T) in [0,1]
        x_rec = (arr * (2.0 * c) - c).astype(np.float32)  # [-c, c]
        out = torch.from_numpy(x_rec).unsqueeze(0)  # (1, n_mels, T)
        return out

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        x, label, machine_id = self.base[idx]
        x_jpeg = self._encode_decode_jpeg(x)
        return x_jpeg, int(label), str(machine_id)


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
    p = argparse.ArgumentParser(description="Evaluate sDSR under JPEG compression (single machine_type).")
    p.add_argument("--stage1_ckpt", type=str, required=True)
    p.add_argument("--stage2_ckpt", type=str, required=True)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--machine_type", type=str, required=True)
    p.add_argument("--machine_id", type=str, default=None, help="Optional: evaluate only on one machine_id")
    p.add_argument("--pauc_max_fpr", type=float, default=0.1)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--no_score_norm", action="store_true")
    p.add_argument("--clip", type=float, default=4.0, help="Clamp range for mapping spectrogram -> uint8 and back ([-clip, clip]).")
    p.add_argument("--qs", type=int, nargs="*", default=None, help="JPEG quality factors, e.g. --qs 10 20 30 ...")
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional CSV output path (default: <stage2_ckpt_parent>/results/jpeg_sweep.csv)",
    )
    return p.parse_args()


def _run(args: argparse.Namespace, tee: Callable[[str], None]) -> None:
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

    # Stage 1 arch from checkpoint
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

    qs = _parse_q_list(args.qs)
    tee(f"JPEG Q sweep: {qs}")
    tee(f"Mapping clamp: [-{args.clip:g}, {args.clip:g}] -> uint8 -> back")

    rows: list[tuple[int, str, float, float]] = []
    for q in qs:
        jpeg_test = JPEGSpectrogramDataset(test_ds, quality=q, clip=args.clip)
        evaluator = AnomalyEvaluator(
            model=model,
            test_dataset=jpeg_test,
            device=args.device,
            pauc_max_fpr=args.pauc_max_fpr,
            batch_size=args.batch_size,
            train_score_stats=train_score_stats,
            train_score_stats_fallback=train_score_stats_fallback,
        )
        results = evaluator.evaluate()
        ids = results.get(jpeg_test.machine_type, {})
        avg = ids.get("average", {"auc": float("nan"), "pauc": float("nan")})
        auc = float(avg.get("auc", float("nan")))
        pauc = float(avg.get("pauc", float("nan")))
        tee(f"Q={q:02d}: average AUC={auc:.4f} pAUC={pauc:.4f}")
        rows.append((q, jpeg_test.machine_type, auc, pauc))

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(args.stage2_ckpt).resolve().parent / "results" / "jpeg_sweep.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Q", "machine_type", "avg_AUC", "avg_pAUC"])
        for r in rows:
            w.writerow([r[0], r[1], f"{r[2]:.6f}", f"{r[3]:.6f}"])
    tee(f"Saved sweep CSV to {out_path}")


def main() -> None:
    args = parse_args()

    results_dir = Path(args.stage2_ckpt).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "jpeg_sweep.log"
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

