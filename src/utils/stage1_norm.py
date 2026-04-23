from __future__ import annotations

from typing import Any

import torch


def load_norm_from_stage1_ckpt(
    stage1_ckpt: dict[str, Any],
    *,
    machine_type: str | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """
    Load spectrogram normalization stats from a Stage 1 checkpoint.

    Returns (mean, std) as tensors (typically scalar) or (None, None) if absent.

    Supports two checkpoint schemas:
    - Global stats: keys ``norm_mean`` and ``norm_std``
    - Per-type stats: key ``norm_stats`` mapping machine_type -> {"mean": ..., "std": ...}
    """
    if "norm_mean" in stage1_ckpt and "norm_std" in stage1_ckpt:
        mean = stage1_ckpt["norm_mean"]
        std = stage1_ckpt["norm_std"]
        if isinstance(mean, torch.Tensor) and isinstance(std, torch.Tensor):
            return mean, std
        return torch.as_tensor(mean), torch.as_tensor(std)

    norm_stats = stage1_ckpt.get("norm_stats")
    if isinstance(norm_stats, dict) and norm_stats:
        key = machine_type
        if key is None:
            # Fallback: pick the first entry deterministically.
            key = sorted(norm_stats.keys())[0]
        entry = norm_stats.get(key)
        if isinstance(entry, dict) and "mean" in entry and "std" in entry:
            mean = entry["mean"]
            std = entry["std"]
            if isinstance(mean, torch.Tensor) and isinstance(std, torch.Tensor):
                return mean, std
            return torch.as_tensor(mean), torch.as_tensor(std)

    return None, None

