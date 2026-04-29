"""
Streaming global per-mel mean/std over DCASE2020 Task 2 layouts (matches dataset cropping).
"""

from __future__ import annotations

import re
from pathlib import Path

import torch
import torchaudio
import torchaudio.transforms as T

from ..utils.audio import amplitude_to_db_power, make_mel_spectrogram, mel_db_to_finite
from .dataset import MEL_TIME_CROP

_FILENAME_RE_TRAIN = re.compile(r"^normal_(id_\d+)_\d+\.wav$", re.IGNORECASE)
_FILENAME_RE_TEST = re.compile(r"^(?:normal|anomaly)_(id_\d+)_\d+\.wav$", re.IGNORECASE)


def _iter_wav_paths_for_stats(
    roots: list[str | Path],
    machine_types: list[str],
    *,
    include_test_for_stats: bool,
) -> list[tuple[Path, re.Pattern[str]]]:
    """Return (wav_path, filename_re) pairs used only for statistics."""
    pairs: list[tuple[Path, re.Pattern[str]]] = []
    for root in roots:
        root_path = Path(root)
        for mt in sorted(machine_types):
            train_dir = root_path / mt / "train"
            if train_dir.is_dir():
                for p in sorted(train_dir.glob("*.wav")):
                    if _FILENAME_RE_TRAIN.match(p.name):
                        pairs.append((p, _FILENAME_RE_TRAIN))
            if include_test_for_stats:
                test_dir = root_path / mt / "test"
                if test_dir.is_dir():
                    for p in sorted(test_dir.glob("*.wav")):
                        if _FILENAME_RE_TEST.match(p.name):
                            pairs.append((p, _FILENAME_RE_TEST))
    return pairs


def compute_global_mel_mean_std(
    roots: list[str | Path],
    machine_types: list[str],
    *,
    include_test_for_stats: bool = False,
    sample_rate: int = 16_000,
    n_fft: int = 1024,
    hop_length: int = 512,
    n_mels: int = 128,
    f_min: float = 0.0,
    f_max: float = 8_000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per-mel-bin mean and std over all selected clips, aggregated over time (first ``MEL_TIME_CROP`` bins only).

    Returns:
        mean, std each shape ``(1, n_mels, 1)`` float32 on CPU.
    """
    mel_t = make_mel_spectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
    )
    to_db = amplitude_to_db_power()

    paths = _iter_wav_paths_for_stats(
        roots, machine_types, include_test_for_stats=include_test_for_stats
    )
    if not paths:
        raise FileNotFoundError(
            f"No wav files found for mel stats under roots={roots!r} "
            f"machine_types={machine_types!r} include_test_for_stats={include_test_for_stats}"
        )

    sum_m = torch.zeros(n_mels, dtype=torch.float64)
    sumsq_m = torch.zeros(n_mels, dtype=torch.float64)
    count = 0

    for wav_path, filename_re in paths:
        if not filename_re.match(wav_path.name):
            continue
        wav, sr = torchaudio.load(str(wav_path))
        if sr != sample_rate:
            wav = torchaudio.functional.resample(wav, sr, sample_rate)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        mel = mel_t(wav)
        x = mel_db_to_finite(to_db(mel).float()).squeeze(0)  # (n_mels, T)
        x = x[:, :MEL_TIME_CROP]
        if x.shape[1] == 0:
            continue
        sum_m += x.sum(dim=1).double()
        sumsq_m += x.pow(2).sum(dim=1).double()
        count += int(x.shape[1])

    if count <= 0:
        raise RuntimeError("Mel stats accumulation got zero time samples.")

    mean_m = sum_m / float(count)
    ex2 = sumsq_m / float(count)
    var_m = (ex2 - mean_m * mean_m).clamp_min(0.0)
    std_m = var_m.sqrt()

    mean = mean_m.float().view(1, n_mels, 1)
    std = std_m.float().view(1, n_mels, 1)
    return mean, std
