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

def load_mel_for_dir(
    audio_dir: Path,
    sample_rate: int,
    mel_transform: T.MelSpectrogram,
    to_db: Callable[[torch.Tensor], torch.Tensor],
    filename_re: re.Pattern[str],
    *,
    norm_mean: torch.Tensor | None = None,
    norm_std: torch.Tensor | None = None,
    standardize: bool = True,
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
        # Safety: different corpora may contain silence/degenerate audio producing
        # NaN/±inf after log or dB conversion in some torchaudio builds.
        # Keep everything finite so a single "poison" clip can't destabilize Stage 1.
        log_mel_db = torch.nan_to_num(log_mel_db, nan=0.0, posinf=0.0, neginf=-80.0)
        if standardize:
            log_mel_db = standardize_spectrogram(log_mel_db, mean=norm_mean, std=norm_std)
        spectrograms.append(log_mel_db)
        
    return spectrograms, machine_id_strs

def standardize_spectrogram(
    mel_db: torch.Tensor,
    *,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
) -> torch.Tensor:
    # mel_db: (1, n_mels, T) in dB, already floor-clipped
    if mean is None or std is None:
        mean = mel_db.mean(dim=(1, 2), keepdim=True)
        std = mel_db.std(dim=(1, 2), keepdim=True)
    return (mel_db - mean) / (std + _EPS)

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