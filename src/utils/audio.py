import re
from pathlib import Path
from typing import Callable

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T

def make_mel_spectrogram(
    *,
    sample_rate: int = 16_000,
    n_fft: int = 1024,
    hop_length: int = 512,
    n_mels: int = 128,
    f_min: float = 0.0,
    f_max: float = 8_000.0,
) -> T.MelSpectrogram:
    return T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
    )


def amplitude_to_db_power(*, top_db: float = 80.0) -> T.AmplitudeToDB:
    """Power mel to dB with fixed dynamic-range clamp via ``top_db``."""
    return T.AmplitudeToDB(stype="power", top_db=top_db)


def mel_db_to_finite(log_mel_db: torch.Tensor, *, top_db: float = 80.0) -> torch.Tensor:
    """Replace NaN/Inf and clamp to [-top_db, top_db] dB."""
    x = torch.nan_to_num(log_mel_db, nan=0.0, posinf=top_db, neginf=-top_db)
    return x.clamp(min=-top_db, max=top_db)


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
        log_mel_db = mel_db_to_finite(log_mel_db)
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
