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

def log_mel_db_to_01(log_mel_db: torch.Tensor, *, top_db: float = 80.0) -> torch.Tensor:
    """
    Map log-mel in dB to [0, 1] deterministically assuming the dB range is [-top_db, 0].

    This matches the common "ref=max, top_db=80" style used in many DCASE pipelines:
    0 dB -> 1, -top_db dB -> 0 (values outside are clamped).
    """
    if top_db <= 0:
        raise ValueError(f"top_db must be > 0, got {top_db}")
    x = (log_mel_db + float(top_db)) / float(top_db)
    return x.clamp_(0.0, 1.0)

def load_mel_for_dir(
    audio_dir: Path,
    sample_rate: int,
    mel_transform: T.MelSpectrogram,
    to_db: Callable[[torch.Tensor], torch.Tensor],
    filename_re: re.Pattern[str],
    *,
    map_to_01: bool = True,
) -> tuple[list[torch.Tensor], list[str]]:
    files = sorted(audio_dir.glob("*.wav"))
    assert files, f"No .wav files found in {audio_dir}"
    spectrograms: list[torch.Tensor] = []
    machine_id_strs: list[str] = []
    top_db = float(getattr(to_db, "top_db", 80.0) or 80.0)
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
        log_mel_db = torch.nan_to_num(log_mel_db, nan=0.0, posinf=0.0, neginf=-top_db)

        if map_to_01:
            log_mel_db = log_mel_db_to_01(log_mel_db, top_db=top_db)

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