"""
Spectrogram standardization tensors stored in Stage 1 checkpoints.

Stage 1 fits (or loads) norm_mean / norm_std on the unpadded mel crop; downstream
must use the same tensors as training (typically from the same stage1_ckpt).
"""

from __future__ import annotations

from typing import Any

import torch


def load_norm_from_stage1_ckpt(
    ckpt: dict[str, Any],
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    mean = ckpt.get("norm_mean")
    std = ckpt.get("norm_std")
    if mean is not None and std is not None:
        return mean, std
    # Legacy keys (if any)
    mean = ckpt.get("mean")
    std = ckpt.get("std")
    if mean is not None and std is not None:
        return mean, std
    return None, None
