import random
import re
import math
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader, Dataset

from ..utils.anomalies import AnomalyMapGenerator


class DCASE2020Task2LogMelDataset(Dataset):
    """
    Training dataset: pre-computes log-mel spectrograms into RAM.
    Single machine type or multiple (machine_types); machine_id parsed from filenames (normal_id_XX_....wav).
    All training samples are normal (no anomalies), so label is always 0.

    Normalization (mean, std) and target_T: for a single machine_type they are computed on that type;
    for multiple machine_types, global per-mel normalization is used: mean/std are computed over all
    types' data combined (one value per mel bin), then applied to the concatenated data. For multiple
    types, all spectrograms are truncated to the minimum length across types (aligned to 16 frames);
    for a single type, T is padded to a multiple of 16.

    __getitem__ returns (spectrogram, label, machine_id) with label 0 = normal.

    Expected layout: root/{machine_type}/train/
        normal_id_01_00000000.wav, normal_id_02_00000000.wav, ...
    """

    _FILENAME_RE = re.compile(r"^normal_(id_\d+)_\d+\.wav$", re.IGNORECASE)

    def __init__(
        self,
        root: str,
        machine_type: str | None = None,
        machine_types: list[str] | None = None,
        sample_rate:   int   = 16_000,
        n_fft:         int   = 1024,
        hop_length:    int   = 512,
        n_mels:        int   = 128,
        f_min:         float = 0.0,
        f_max:         float = 8_000.0,
        top_db:        float = 80.0,
        normalize:     bool  = True,
    ):
        if machine_types is not None:
            self._init_multi(root, machine_types, sample_rate, n_fft, hop_length, n_mels, f_min, f_max, top_db, normalize)
        elif machine_type is not None:
            self._init_single(root, machine_type, sample_rate, n_fft, hop_length, n_mels, f_min, f_max, top_db, normalize)
        else:
            raise ValueError("Provide either machine_type or machine_types")

    def _load_mel_for_dir(
        self,
        audio_dir: Path,
        sample_rate: int,
    ) -> tuple[list[torch.Tensor], list[str]]:
        files = sorted(audio_dir.glob("*.wav"))
        assert files, f"No .wav files found in {audio_dir}"
        spectrograms: list[torch.Tensor] = []
        machine_id_strs: list[str] = []
        for path in files:
            m = self._FILENAME_RE.match(path.name)
            mid = m.group(1) if m else "id_00"
            machine_id_strs.append(mid)
            wav, sr = torchaudio.load(path)
            if sr != sample_rate:
                wav = torchaudio.functional.resample(wav, sr, sample_rate)
            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)
            mel = self.mel_transform(wav)
            log_mel = self.to_db(mel)
            spectrograms.append(log_mel)
        return spectrograms, machine_id_strs

    def _init_single(
        self,
        root: str,
        machine_type: str,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        n_mels: int,
        f_min: float,
        f_max: float,
        top_db: float,
        normalize: bool,
    ) -> None:
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
        )
        self.to_db = T.AmplitudeToDB(top_db=top_db)
        self.machine_type = machine_type

        audio_dir = Path(root) / machine_type / "train"
        spectrograms, machine_id_strs = self._load_mel_for_dir(audio_dir, sample_rate)

        self.data = torch.stack(spectrograms)
        self.machine_ids = sorted(set(machine_id_strs))
        self._machine_id_strs = machine_id_strs

        if normalize:
            self.mean = self.data.mean(dim=(0, 2, 3), keepdim=True)
            self.std  = self.data.std(dim=(0, 2, 3),  keepdim=True)
            self.data = (self.data - self.mean) / (self.std + 1e-8)
        else:
            shape = (1, 1, self.data.shape[2], 1)
            self.mean = torch.zeros(shape, dtype=self.data.dtype, device=self.data.device)
            self.std  = torch.ones(shape, dtype=self.data.dtype, device=self.data.device)

        target_T = math.ceil(self.data.shape[-1] / 16) * 16
        self.target_T = target_T
        if self.data.shape[-1] != target_T:
            pad = target_T - self.data.shape[-1]
            self.data = F.pad(self.data, (0, pad))
            print(f"Padded T: {self.data.shape[-1] - pad} → {self.data.shape[-1]} (target: {target_T})")

        print(
            f"DCASE2020Task2LogMelDataset: {machine_type} | {len(self.data)} spectrograms, "
            f"shape {tuple(self.data.shape)} | IDs: {self.machine_ids} | "
            f"{self.data.nbytes / 1e9:.2f} GB in RAM"
        )

    def _init_multi(
        self,
        root: str,
        machine_types: list[str],
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        n_mels: int,
        f_min: float,
        f_max: float,
        top_db: float,
        normalize: bool,
    ) -> None:
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
        )
        self.to_db = T.AmplitudeToDB(top_db=top_db)
        self.machine_type = "+".join(sorted(machine_types))

        root_path = Path(root)
        all_spectrograms: list[torch.Tensor] = []
        all_machine_id_strs: list[str] = []
        min_T = 0

        for mt in sorted(machine_types):
            audio_dir = root_path / mt / "train"
            spectrograms, machine_id_strs = self._load_mel_for_dir(audio_dir, sample_rate)
            data_mt = torch.stack(spectrograms)
            t_len = data_mt.shape[-1]
            min_T = t_len if min_T is None else min(min_T, t_len)
            all_spectrograms.append(data_mt)
            all_machine_id_strs.extend(machine_id_strs)

        target_T = max(16, (min_T // 16) * 16)
        self.target_T = target_T
        truncated = [data_mt[..., :target_T] for data_mt in all_spectrograms]
        self.data = torch.cat(truncated, dim=0)
        self._machine_id_strs = all_machine_id_strs
        self.machine_ids = sorted(set(self._machine_id_strs))

        if normalize:
            self.mean = self.data.mean(dim=(0, 2, 3), keepdim=True)
            self.std = self.data.std(dim=(0, 2, 3), keepdim=True)
            self.data = (self.data - self.mean) / (self.std + 1e-8)
        else:
            shape = (1, 1, self.data.shape[2], 1)
            self.mean = torch.zeros(shape, dtype=self.data.dtype, device=self.data.device)
            self.std = torch.ones(shape, dtype=self.data.dtype, device=self.data.device)

        print(
            f"DCASE2020Task2LogMelDataset: {self.machine_type} | {len(self.data)} spectrograms (global per-mel norm), "
            f"shape {tuple(self.data.shape)} | IDs: {self.machine_ids} | "
            f"{self.data.nbytes / 1e9:.2f} GB in RAM"
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        # label 0 = normal (all training samples are normal)
        return self.data[idx], 0, self._machine_id_strs[idx]

    def _normalize(self, data: torch.Tensor) -> torch.Tensor:
        return (data - self.mean) / (self.std + 1e-8)
    
    def _denormalize(self, data: torch.Tensor) -> torch.Tensor:
        return data * (self.std + 1e-8) + self.mean


class AudDSRAnomTrainDataset(Dataset):
    """
    Stage-2 training dataset: wraps a normal spectrogram dataset and adds
    synthetic anomaly masks at dataset level (DSR-style).

    Each __getitem__ returns a dict: image (spectrogram), anomaly_mask,
    has_anomaly, label, machine_id. With probability zero_mask_prob the mask
    is zero (normal); otherwise a mask is generated with the chosen strategy
    (perlin, audio_specific, or both). The model uses the mask for codebook
    replacement in feature space.
    """

    def __init__(
        self,
        base_dataset: DCASE2020Task2LogMelDataset,
        strategy: Literal["perlin", "audio_specific", "both"] = "both",
        zero_mask_prob: float = 0.5,
    ) -> None:
        self.base = base_dataset
        self.strategy = strategy
        self.zero_mask_prob = zero_mask_prob
        self.machine_type = getattr(base_dataset, "machine_type", None)
        # Spectrogram space: (n_mels, T); mask shape (1, 1, n_mels, T)
        _, _, n_mels, T = base_dataset.data.shape
        self.n_mels = n_mels
        self.T = T
        spectrogram_shape = (n_mels, T)
        self._mask_generator = AnomalyMapGenerator(
            strategy=strategy,
            spectrogram_shape=spectrogram_shape,
            q_shape=spectrogram_shape,
            n_mels=n_mels,
            T=T,
            zero_mask_prob=0.0,
        )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int | str]:
        spectrogram, label, machine_id = self.base[idx]
        n_mels, T = self.n_mels, self.T
        if random.random() < self.zero_mask_prob:
            mask = torch.zeros(1, 1, n_mels, T, dtype=torch.float32)
            has_anomaly = 0.0
        else:
            mask = self._mask_generator.generate(
                1, device="cpu", force_anomaly=True
            )
            has_anomaly = 1.0
        return {
            "image": spectrogram,
            "anomaly_mask": mask.squeeze(0),
            "has_anomaly": torch.tensor(has_anomaly, dtype=torch.float32),
            "label": label,
            "machine_id": machine_id,
        }


class DCASE2020Task2TestDataset(Dataset):
    """
    Test dataset: normal and anomalous samples per machine ID.
    Uses mean, std, and target_T from the training dataset for consistent normalization and length.

    __getitem__ returns (spectrogram, label, machine_id):
        label 0 = normal, 1 = anomalous (from filename).

    Expected layout: root/{machine_type}/test/
        normal_id_01_00000000.wav   -> label 0, machine_id id_01
        anomaly_id_01_00000000.wav  -> label 1, machine_id id_01
        ...
    Filename format: {normal|anomaly}_id_{XX}_{number}.wav
    """

    _FILENAME_RE = re.compile(r"^(normal|anomaly)_(id_\d+)_\d+\.wav$", re.IGNORECASE)

    def __init__(
        self,
        root: str,
        machine_type: str,
        sample_rate: int = 16_000,
        n_fft: int = 1024,
        hop_length: int = 512,
        n_mels: int = 128,
        f_min: float = 0.0,
        f_max: float = 8_000.0,
        top_db: float = 80.0,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
        target_T: int | None = None,
    ):
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
        )
        self.to_db = T.AmplitudeToDB(top_db=top_db)
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.mean = mean
        self.std = std
        self.target_T = target_T

        base = Path(root) / machine_type / "test"
        if not base.exists():
            raise FileNotFoundError(f"Test directory not found: {base}")

        self.samples: list[tuple[str, int, str]] = []
        machine_ids: set[str] = set()

        for wav_path in sorted(base.glob("*.wav")):
            m = self._FILENAME_RE.match(wav_path.name)
            if m:
                label = 0 if m.group(1).lower() == "normal" else 1
                mid = m.group(2)
                machine_ids.add(mid)
                self.samples.append((str(wav_path), label, mid))


        self.machine_ids = sorted(machine_ids)
        self.machine_type = machine_type

        print(
            f"DCASE2020Task2TestDataset: {machine_type} | {len(self.samples)} clips | "
            f"IDs: {self.machine_ids}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        wav_path, label, machine_id = self.samples[idx]
        wav, sr = torchaudio.load(wav_path)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        mel = self.mel_transform(wav)
        log_mel = self.to_db(mel)

        if self.mean is not None and self.std is not None:
            log_mel = (log_mel - self.mean) / (self.std + 1e-8)
            if log_mel.dim() == 4:
                log_mel = log_mel.squeeze(1)

        T = log_mel.shape[-1]
        if self.target_T is not None:
            if T < self.target_T:
                log_mel = F.pad(log_mel, (0, self.target_T - T))
            elif T > self.target_T:
                log_mel = log_mel[..., : self.target_T]
        else:
            # Pad to multiple of 16 for VQ-VAE 4x downsampling
            target = math.ceil(T / 16) * 16
            if T != target:
                log_mel = F.pad(log_mel, (0, target - T))

        return log_mel, label, machine_id


def make_dataloader(dataset: DCASE2020Task2LogMelDataset | DCASE2020Task2TestDataset, batch_size: int = 256) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = 0,       # no workers needed — data is already in RAM
        pin_memory  = True,    # speeds up CPU->GPU transfer
    )

if __name__ == "__main__":
    MACHINE_TYPES = ["fan", "pump", "slider", "valve", "ToyCar", "ToyConveyor"]
    # MACHINE_TYPES = ["fan"]

    for machine_type in MACHINE_TYPES:
        dataset = DCASE2020Task2LogMelDataset(
            root       = "/mnt/ssd/LaCie/dcase2020-task2-dev-dataset",
            machine_type = machine_type,
            normalize = True,
        )
        loader = make_dataloader(dataset, batch_size=256)

        # Sanity check
        specs, labels, machine_ids = next(iter(loader))
        print(f"Batch shape : {tuple(specs.shape)}")   # (256, 1, n_mels, T)
        print(f"Label shape : {tuple(labels.shape)}")
        print(f"Machine IDs (batch): {machine_ids[:3]}...")
        print(f"Value range : [{specs.min():.2f}, {specs.max():.2f}]")
        

        test_dataset = DCASE2020Task2TestDataset(
            root          = "/mnt/ssd/LaCie/dcase2020-task2-dev-dataset",
            machine_type  = machine_type,
            mean          = dataset.mean,
            std           = dataset.std,
            target_T      = dataset.target_T,
        )
        test_loader = make_dataloader(test_dataset, batch_size=256)

        # Sanity check
        specs, labels, machine_ids = next(iter(test_loader))
        print(f"Batch shape : {tuple(specs.shape)}")   # (256, 1, n_mels, T)
        print(f"Label shape : {tuple(labels.shape)}")  # (256,)
        print(f"Value range : [{specs.min():.2f}, {specs.max():.2f}]")
        print()