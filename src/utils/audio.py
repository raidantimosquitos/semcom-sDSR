import torch
import torchaudio
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pathlib import Path
import numpy as np
import torchaudio.transforms as T
import re
from typing import Callable

_EPS = 1e-8


def pad_time_replicate(x: torch.Tensor, target_T: int) -> torch.Tensor:
    """
    Replicate-pad (or truncate) along time axis to target_T.

    Args:
        x: (C, n_mels, T) or (B, C, n_mels, T)
        target_T: desired time length

    Returns:
        Tensor with same leading dims and time length target_T.
    """
    if x.dim() == 3:
        C, n_mels, T = x.shape
        if T == target_T:
            return x
        if T > target_T:
            return x[..., :target_T]
        pad = target_T - T
        last = x[..., -1:].expand(C, n_mels, pad)
        return torch.cat([x, last], dim=-1)

    if x.dim() == 4:
        B, C, n_mels, T = x.shape
        if T == target_T:
            return x
        if T > target_T:
            return x[..., :target_T]
        pad = target_T - T
        last = x[..., -1:].expand(B, C, n_mels, pad)
        return torch.cat([x, last], dim=-1)

    raise ValueError(f"pad_time_replicate expects 3D or 4D tensor, got shape={tuple(x.shape)}")


def _delta_regression_5frame(x: torch.Tensor) -> torch.Tensor:
    """
    5-frame regression delta (window=2) along time dimension.

    Uses: Δ_t = (1*(x_{t+1}-x_{t-1}) + 2*(x_{t+2}-x_{t-2})) / 10
    Edge handling: replicate-pad 2 frames on both sides.

    Args:
        x: (B, 1, n_mels, T) or (B, n_mels, T) or (1, n_mels, T)

    Returns:
        delta with same shape as x.
    """
    orig_dim = x.dim()
    if orig_dim == 3:
        x = x.unsqueeze(0)  # (1, C, n_mels, T) where C is 1
    if x.dim() == 3:
        # (B, n_mels, T) -> (B, 1, n_mels, T)
        x = x.unsqueeze(1)
    if x.dim() != 4:
        raise ValueError(f"_delta_regression_5frame expects 3D/4D tensor, got shape={tuple(x.shape)}")

    B, C, n_mels, T = x.shape
    if C != 1:
        # keep implementation simple/explicit; callers stack after
        raise ValueError(f"_delta_regression_5frame expects C=1, got C={C}")

    left = x[..., :1].expand(B, C, n_mels, 2)
    right = x[..., -1:].expand(B, C, n_mels, 2)
    xpad = torch.cat([left, x, right], dim=-1)  # (B,1,n_mels,T+4)

    xm2 = xpad[..., 0:T]
    xm1 = xpad[..., 1:T + 1]
    xp1 = xpad[..., 3:T + 3]
    xp2 = xpad[..., 4:T + 4]
    delta = ((xp1 - xm1) + 2.0 * (xp2 - xm2)) / 10.0

    if orig_dim == 3:
        return delta.squeeze(0)
    return delta


def log_mel_to_delta3(log_mel_db: torch.Tensor) -> torch.Tensor:
    """
    Convert log-mel dB spectrogram to 3-channel (logmel, delta, delta-delta).

    Assumes time length is already padded/truncated to the final target_T.

    Args:
        log_mel_db: (1, n_mels, T) or (B, 1, n_mels, T)

    Returns:
        (3, n_mels, T) or (B, 3, n_mels, T)
    """
    if log_mel_db.dim() == 3:
        x = log_mel_db.unsqueeze(0)  # (1,1,n_mels,T)
        squeeze_b = True
    elif log_mel_db.dim() == 4:
        x = log_mel_db
        squeeze_b = False
    else:
        raise ValueError(f"log_mel_to_delta3 expects 3D/4D tensor, got shape={tuple(log_mel_db.shape)}")

    if x.shape[1] != 1:
        raise ValueError(f"log_mel_to_delta3 expects channel dim=1, got shape={tuple(x.shape)}")

    delta = _delta_regression_5frame(x)         # (B,1,n_mels,T)
    delta2 = _delta_regression_5frame(delta)   # (B,1,n_mels,T)
    out = torch.cat([x, delta, delta2], dim=1)  # (B,3,n_mels,T)
    return out.squeeze(0) if squeeze_b else out


def load_mel_for_dir(
    audio_dir: Path,
    sample_rate: int,
    mel_transform: T.MelSpectrogram,
    to_db: Callable[[torch.Tensor], torch.Tensor],
    filename_re: re.Pattern[str],
) -> tuple[list[torch.Tensor], list[str]]:
    files = sorted(audio_dir.glob("*.wav"))
    assert files, f"No .wav files found in {audio_dir}"
    spectrograms: list[torch.Tensor] = []
    machine_id_strs: list[str] = []
    for path in files:
        m = filename_re.match(path.name)
        mid = m.group(1) if m else "id_00"
        machine_id_strs.append(mid)
        wav, sr = torchaudio.load(path)
        if sr != sample_rate:
            wav = torchaudio.functional.resample(wav, sr, sample_rate)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        mel = mel_transform(wav)
        log_mel_db = to_db(mel).float()  # (1, n_mels, T)
        spectrograms.append(log_mel_db)
        
    return spectrograms, machine_id_strs
    
def log_mel_to_rgb(log_mel: torch.Tensor, cmap: colors.Colormap | None = None) -> torch.Tensor:
    """Convert log-mel spectrogram (1 or 2D) to RGB tensor (3, n_mels, T) in [0, 1]."""
    if cmap is None:
        cmap = plt.get_cmap("jet")
    log_mel_np = log_mel.squeeze().cpu().numpy()
    lo, hi = float(log_mel_np.min()), float(log_mel_np.max())
    span = hi - lo
    if span <= 0:
        span = 1.0
    log_mel_norm = (log_mel_np - lo) / span
    rgba: np.ndarray = np.asarray(cmap(log_mel_norm))
    rgb = rgba[..., :3]
    return torch.from_numpy(rgb.copy()).float().permute(2, 0, 1)