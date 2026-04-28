import logging
import random
import re
import math
from pathlib import Path
from typing import Any, Literal, cast

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader, Dataset

from ..utils.anomalies import AnomalyMapGenerator
from ..utils.audio import load_mel_for_dir

# Log-mel time crop before standardization (DCASE Task 2 shortest-clip alignment).
MEL_TIME_CROP = 313

# Globally unique machine key when multiple machine_types are loaded (train + eval calibration).
COMPOSITE_ID_SEP = "__"


def composite_machine_id(machine_type: str, machine_id: str) -> str:
    """Join DCASE machine_type and id (e.g. fan + id_00 -> fan__id_00)."""
    return f"{machine_type}{COMPOSITE_ID_SEP}{machine_id}"


class DCASE2020Task2LogMelDataset(Dataset):
    """
    Training dataset: pre-computes log-mel spectrograms into RAM.
    Single machine type or multiple (machine_types); machine_id parsed from filenames (normal_id_XX_....wav).
    All training samples are normal (no anomalies), so label is always 0..    
    
    For multiple types: (1) each type is truncated to 313 samples;
    (2) optional standardization uses only unpadded content (mean/std over T=313);
    (3) then zero-pad to the next multiple of 16 (320). Same order for a single type.

    __getitem__ returns (spectrogram, label, machine_id) with label 0 = normal.
    When ``machine_types`` has more than one entry, ``machine_id`` is composite:
    ``{machine_type}__{id_XX}`` (see :func:`composite_machine_id`) so IDs do not collide
    across types. ``self._machine_type_strs[i]`` is the DCASE machine type string for row ``i``.

    Stage1 training uses this dataset with machine_types (all types) as the general dataset.

    Stage2 uses a machine-type-specific dataset that includes anomalies (see AudDSRAnomTrainDataset).

    Expected layout:
        root/{machine_type}/train/  -> normal_id_01_00000000.wav, ...
        root/{machine_type}/test/  -> normal_id_01_*.wav, anomaly_id_01_*.wav (only used when include_test=True in _init_multi)
    """

    _FILENAME_RE = re.compile(r"^normal_(id_\d+)_\d+\.wav$", re.IGNORECASE)
    # test/ has both normal_* and anomaly_*; use this when loading from test/
    _FILENAME_RE_TEST = re.compile(r"^(?:normal|anomaly)_(id_\d+)_\d+\.wav$", re.IGNORECASE)

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
        machine_id: str | None = None,
        include_test: bool = True,
        # Normalization is intentionally disabled project-wide (work in raw log-mel dB).
        # These args are kept for backward compatibility but are ignored.
    ):
        # Hard-disable any standardization to keep values in proper dB scale end-to-end.
        if machine_types is not None:
            if machine_id is not None:
                raise ValueError("machine_id filter only applies to single machine_type")
            self._init_multi(
                root,
                machine_types,
                sample_rate,
                n_fft,
                hop_length,
                n_mels,
                f_min,
                f_max,
                top_db,
                include_test,
            )
        elif machine_type is not None:
            self._init_single(
                root,
                machine_type,
                sample_rate,
                n_fft,
                hop_length,
                n_mels,
                f_min,
                f_max,
                top_db,
                machine_id,
            )
        else:
            raise ValueError("Provide either machine_type or machine_types")

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
        machine_id: str | None = None,
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
        spectrograms, machine_id_strs = load_mel_for_dir(
            audio_dir,
            sample_rate,
            self.mel_transform,
            self.to_db,
            self._FILENAME_RE,
            map_to_01=True,
        )

        # Truncate to MEL_TIME_CROP (shortest spectrogram alignment)
        spectrograms = [s[..., :MEL_TIME_CROP] for s in spectrograms]
        stacked = torch.stack(spectrograms)

        target_T = ((MEL_TIME_CROP + 15) // 16) * 16  # 32

        pad = target_T - stacked.shape[-1]
        if pad > 0:
            stacked = F.pad(stacked, (0, pad), mode="constant", value=0.0)

        self.data = stacked
        self.target_T = target_T
        self.machine_ids = sorted(set(machine_id_strs))
        self._machine_id_strs = machine_id_strs
        self._machine_type_strs = [machine_type] * len(machine_id_strs)
        self._use_composite_ids = False

        if machine_id is not None:
            indices = [i for i, mid in enumerate(self._machine_id_strs) if mid == machine_id]
            if not indices:
                raise ValueError(f"No samples found for machine_id={machine_id} in {machine_type}")
            self.data = self.data[indices]
            self._machine_id_strs = [self._machine_id_strs[i] for i in indices]
            self._machine_type_strs = [self._machine_type_strs[i] for i in indices]
            self.machine_ids = [machine_id]
            print(
                f"DCASE2020Task2LogMelDataset: filtered to machine_id={machine_id} | "
                f"{len(self.data)} spectrograms"
            )

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
        include_test: bool = True,
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
        all_machine_type_strs: list[str] = []

        target_T = ((MEL_TIME_CROP + 15) // 16) * 16  # 320
        self._use_composite_ids = len(machine_types) > 1

        def _has_wavs(p: Path) -> bool:
            try:
                return p.is_dir() and any(p.glob("*.wav"))
            except OSError:
                return False

        # 1. Load each type; when include_test=True, also load test/ (normal + anomaly).
        for mt in sorted(machine_types):
            train_dir = root_path / mt / "train"
            spectrograms: list[torch.Tensor] = []
            machine_id_strs: list[str] = []

            # Prefer train/ if present; some roots (e.g. eval) only have test/.
            if _has_wavs(train_dir):
                spectrograms, machine_id_strs = load_mel_for_dir(
                    train_dir,
                    sample_rate,
                    self.mel_transform,
                    self.to_db,
                    self._FILENAME_RE,
                    map_to_01=True,
                )
            if include_test:
                test_dir = root_path / mt / "test"
                if _has_wavs(test_dir):
                    spec_test, mid_test = load_mel_for_dir(
                        test_dir,
                        sample_rate,
                        self.mel_transform,
                        self.to_db,
                        self._FILENAME_RE_TEST,
                        map_to_01=True,
                    )
                    spectrograms = spectrograms + spec_test
                    machine_id_strs = machine_id_strs + mid_test

            if not spectrograms:
                logging.warning(
                    f"DCASE2020Task2LogMelDataset: skip {mt} under root={root_path} "
                    f"(no wavs in train/ and {'test/' if include_test else 'test disabled'})"
                )
                continue

            spectrograms = [s[..., :MEL_TIME_CROP] for s in spectrograms]
            stacked_mt = torch.stack(spectrograms)
            all_spectrograms.append(stacked_mt)
            all_machine_id_strs.extend(machine_id_strs)
            all_machine_type_strs.extend([mt] * len(machine_id_strs))

        if not all_spectrograms:
            raise FileNotFoundError(
                f"No usable audio found under root={root_path}. "
                "Expected at least one of {machine}/train/*.wav, or {machine}/test/*.wav when include_test=True."
            )

        self.data = torch.cat(all_spectrograms, dim=0)

        pad = target_T - self.data.shape[-1]
        if pad > 0:
            self.data = F.pad(self.data, (0, pad), mode="constant", value=0.0)

        self.target_T = target_T
        self._machine_id_strs = all_machine_id_strs
        self._machine_type_strs = all_machine_type_strs
        if self._use_composite_ids:
            self.machine_ids = sorted(
                {
                    composite_machine_id(mt, mid)
                    for mt, mid in zip(self._machine_type_strs, self._machine_id_strs)
                }
            )
        else:
            self.machine_ids = sorted(set(self._machine_id_strs))

        print(f"DCASE2020Task2LogMelDataset: {self.machine_type} | {len(self.data)} spectrograms (T→{target_T}), shape {tuple(self.data.shape)} | IDs: {self.machine_ids} | {self.data.nbytes / 1e9:.2f} GB in RAM")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        # label 0 = normal (all training samples are normal)
        # self.data stores raw log-mel (N, 3, n_mels, T)
        x_logmel = self.data[idx]  # (1, 3, n_mels, T)
        mid = self._machine_id_strs[idx]
        if getattr(self, "_use_composite_ids", False):
            mid = composite_machine_id(self._machine_type_strs[idx], mid)
        return x_logmel, 0, mid


class ConcatLogMelDataset(Dataset):
    """
    Concatenate multiple precomputed :class:`DCASE2020Task2LogMelDataset` into one.

    Intended for Stage 1 only when you want to train representation learning on
    multiple DCASE roots (e.g. dev + eval + additional).
    """

    def __init__(self, datasets: list[DCASE2020Task2LogMelDataset]) -> None:
        if not datasets:
            raise ValueError("ConcatLogMelDataset: datasets must be non-empty")

        t0 = int(datasets[0].target_T)
        s0 = tuple(datasets[0].data.shape[1:])
        for ds in datasets[1:]:
            if int(ds.target_T) != t0:
                raise ValueError(
                    f"ConcatLogMelDataset: target_T mismatch {t0} vs {int(ds.target_T)}"
                )
            if tuple(ds.data.shape[1:]) != s0:
                raise ValueError(
                    f"ConcatLogMelDataset: data shape mismatch {s0} vs {tuple(ds.data.shape[1:])}"
                )

        self.datasets = datasets
        self.target_T = t0

        # Merge tensors + metadata lists so downstream code can read `.data` and `.machine_type`.
        self.data = torch.cat([ds.data for ds in datasets], dim=0)
        self._machine_id_strs = sum((list(ds._machine_id_strs) for ds in datasets), [])
        self._machine_type_strs = sum((list(ds._machine_type_strs) for ds in datasets), [])
        self._use_composite_ids = any(getattr(ds, "_use_composite_ids", False) for ds in datasets)

        # Informational fields
        self.machine_type = "+".join(sorted({str(ds.machine_type) for ds in datasets}))
        if self._use_composite_ids:
            self.machine_ids = sorted(
                {
                    composite_machine_id(mt, mid)
                    for mt, mid in zip(self._machine_type_strs, self._machine_id_strs)
                }
            )
        else:
            self.machine_ids = sorted(set(self._machine_id_strs))

        print(
            f"ConcatLogMelDataset: {len(datasets)} roots | {len(self.data)} spectrograms, "
            f"shape {tuple(self.data.shape)} | IDs: {len(self.machine_ids)} | "
            f"{self.data.nbytes / 1e9:.2f} GB in RAM"
        )

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        x_logmel = self.data[idx]
        mid = self._machine_id_strs[idx]
        if self._use_composite_ids:
            mid = composite_machine_id(self._machine_type_strs[idx], mid)
        return x_logmel, 0, mid


class AudDSRAnomTrainDataset(Dataset):
    """
    Stage-2 training dataset: wraps a normal spectrogram dataset and adds
    synthetic anomaly masks at dataset level (DSR-style).

    The base dataset is typically :class:`DCASE2020Task2LogMelDataset` with either
    a single ``machine_type`` or multiple ``machine_types`` (joint Stage-2).
    ``base.machine_type`` is the joined name when multi-type; ``machine_id`` in
    batches may be composite ``{type}__{id_XX}`` when multiple types are loaded.

    Each __getitem__ returns a dict: image (spectrogram), anomaly_mask,
    has_anomaly, label, machine_id. With probability zero_mask_prob the mask
    is zero (normal); otherwise a mask is generated. The model uses the mask for codebook
    replacement in feature space.

    When adversarial_dataset is provided (e.g. other machine_ids of the same
    single machine type), the anomaly half is split 50-50: 50% synthetic masks,
    50% adversarial samples with mask of all 1s. This path is intended for
    single-type bases only; joint multi-type training should use
    ``adversarial_dataset=None`` unless extended separately.

    Overall: zero_mask_prob normal,
    (1-zero_mask_prob)*0.5 synthetic, (1-zero_mask_prob)*0.5 adversarial.
    """

    def __init__(
        self,
        base_dataset: DCASE2020Task2LogMelDataset,
        zero_mask_prob: float = 0.5,
        adversarial_dataset: Dataset | None = None,
        machine_type: str | None = None,
    ) -> None:
        self.base = base_dataset
        self.machine_type = machine_type or getattr(base_dataset, "machine_type", None)
        self.zero_mask_prob = zero_mask_prob
        self.adversarial_dataset = (
            adversarial_dataset
            if (adversarial_dataset is not None and len(cast(Any, adversarial_dataset)) > 0)
            else None
        )
        # Spectrogram space: (n_mels, T); mask shape (1, 1, n_mels, T)
        _, _, n_mels, T = base_dataset.data.shape
        self.n_mels = n_mels
        self.T = T
        spectrogram_shape = (n_mels, T)
        self._mask_generator = AnomalyMapGenerator(
            spectrogram_shape=spectrogram_shape,
        )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int | str]:
        n_mels, T = self.n_mels, self.T
        if random.random() < self.zero_mask_prob:
            spectrogram, label, machine_id = self.base[idx]
            mask = torch.zeros(1, 1, n_mels, T, dtype=torch.float32)
            has_anomaly = 0.0
        else:
            # Anomaly branch: 50% synthetic mask, 50% adversarial (if available)
            adv_ds = self.adversarial_dataset
            use_adversarial = (
                adv_ds is not None
                and random.random() < 0.5
            )
            if use_adversarial and adv_ds is not None:
                j = random.randint(0, len(cast(Any, adv_ds)) - 1)
                spectrogram, label, machine_id = adv_ds[j]
                mask = torch.ones(1, 1, n_mels, T, dtype=torch.float32)
                has_anomaly = 1.0
            else:
                spectrogram, label, machine_id = self.base[idx]
                mask = self._mask_generator.generate_for_training_sample(
                    device="cpu",
                )
                has_anomaly = 1.0
        return {
            "image": spectrogram,
            "anomaly_mask": mask.squeeze(0),
            "has_anomaly": torch.tensor(has_anomaly, dtype=torch.float32),
            "label": 1 if has_anomaly else 0,
            "machine_id": machine_id,
        }


class DCASE2020Task2TestDataset(Dataset):
    """
    Test dataset: normal and anomalous samples per machine ID.
    Uses target_T from the training dataset for consistent normalization and length.

    __getitem__ returns (spectrogram, label, machine_id):
        label 0 = normal, 1 = anomalous (from filename).

    Provide exactly one of ``machine_type`` or ``machine_types``. When multiple types are
    loaded, ``machine_id`` in each tuple is composite ``{type}__{id_XX}`` (same as
    :class:`DCASE2020Task2LogMelDataset` with multiple types). ``machine_id`` filter
    applies only to single-type construction.

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
        machine_type: str | None = None,
        machine_types: list[str] | None = None,
        sample_rate: int = 16_000,
        n_fft: int = 1024,
        hop_length: int = 512,
        n_mels: int = 128,
        f_min: float = 0.0,
        f_max: float = 8_000.0,
        top_db: float = 80.0,
        target_T: int | None = None,
        machine_id: str | None = None,
        # Normalization is intentionally disabled project-wide (work in raw log-mel dB).
        # These args are kept for backward compatibility but are ignored.
        norm_mean: torch.Tensor | None = None,
        norm_std: torch.Tensor | None = None,
        standardize: bool = False,
    ):
        if (machine_type is None) == (machine_types is None):
            raise ValueError("Provide exactly one of machine_type or machine_types")
        if machine_types is not None:
            if machine_id is not None:
                raise ValueError("machine_id filter only applies to single machine_type")
            self._init_multi(
                root,
                machine_types,
                sample_rate,
                n_fft,
                hop_length,
                n_mels,
                f_min,
                f_max,
                top_db,
                target_T,
            )
        else:
            assert machine_type is not None
            self._init_single(
                root,
                machine_type,
                sample_rate,
                n_fft,
                hop_length,
                n_mels,
                f_min,
                f_max,
                top_db,
                target_T,
                machine_id,
            )
        self.norm_mean = None
        self.norm_std = None
        self.standardize = False

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
        target_T: int | None,
        machine_id: str | None,
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
        self.n_mels = n_mels
        self.sample_rate = sample_rate
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
                if machine_id is not None and mid != machine_id:
                    continue
                machine_ids.add(mid)
                self.samples.append((str(wav_path), label, mid))

        self.machine_ids = sorted(machine_ids)
        self.machine_type = machine_type

        print(
            f"DCASE2020Task2TestDataset: {machine_type} | {len(self.samples)} clips | "
            f"IDs: {self.machine_ids}"
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
        target_T: int | None,
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
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.target_T = target_T

        self.machine_type = "+".join(sorted(machine_types))
        self._use_composite_ids = len(machine_types) > 1
        root_path = Path(root)

        self.samples = []
        machine_ids: set[str] = set()

        for mt in sorted(machine_types):
            base = root_path / mt / "test"
            if not base.exists():
                raise FileNotFoundError(f"Test directory not found: {base}")
            for wav_path in sorted(base.glob("*.wav")):
                m = self._FILENAME_RE.match(wav_path.name)
                if m:
                    label = 0 if m.group(1).lower() == "normal" else 1
                    mid = m.group(2)
                    out_id = composite_machine_id(mt, mid) if self._use_composite_ids else mid
                    machine_ids.add(out_id)
                    self.samples.append((str(wav_path), label, out_id))

        self.machine_ids = sorted(machine_ids)

        print(
            f"DCASE2020Task2TestDataset: {self.machine_type} | {len(self.samples)} clips | "
            f"IDs: {len(self.machine_ids)} unique"
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
        log_mel = self.to_db(mel).float()  # (1, n_mels, T)
        # Match training: crop time, then standardize, then pad (in standardized space).
        log_mel = log_mel[..., :MEL_TIME_CROP]
        standardized_mel = log_mel

        T = standardized_mel.shape[-1]
        if self.target_T is not None and T < self.target_T:
            standardized_mel = F.pad(standardized_mel, (0, self.target_T - T), mode="constant", value=0.0)

        return standardized_mel, label, machine_id


def make_dataloader(dataset: DCASE2020Task2LogMelDataset | DCASE2020Task2TestDataset, batch_size: int = 256) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = 0,       # no workers needed — data is already in RAM
        pin_memory  = True,    # speeds up CPU->GPU transfer
    )

if __name__ == "__main__":
    import os
    # Smoke test: use one machine type to limit RAM (full multi-type can OOM)
    DATA_ROOT = Path(os.environ.get("DATA_PATH", "")) or (
        Path(__file__).resolve().parents[2] / "dataset" / "dcase2020_task2_dev_dataset"
    )
    if not DATA_ROOT.exists():
        DATA_ROOT = Path("/mnt/ssd/LaCie/dcase2020_task2/dcase2020_task2_dev_dataset")
    if not DATA_ROOT.exists():
        print("Dataset root not found. Set DATA_PATH or create dataset/dcase2020_task2_dev_dataset")
        raise SystemExit(0)

    DATA_ROOT = Path("/mnt/ssd/LaCie/dcase2020_task2/dcase2020_task2_dev_dataset")
    MACHINE_TYPES = ["fan", "valve"]  # single type to keep smoke test light
    MACHINE_TYPE_TEST = "valve"
    root_str = str(DATA_ROOT)


    print(f"Dataset root: {DATA_ROOT}")

    dataset = DCASE2020Task2LogMelDataset(root=root_str, machine_types=MACHINE_TYPES)
    if len(MACHINE_TYPES) > 1:
        _, _, mid0 = dataset[0]
        assert COMPOSITE_ID_SEP in mid0, f"expected composite machine_id, got {mid0}"
    loader = make_dataloader(dataset, batch_size=min(32, len(dataset)))

    specs, labels, machine_ids = next(iter(loader))
    print(f"Train batch shape: {tuple(specs.shape)}  (B, 1, n_mels, T)")
    print(f"Label shape: {tuple(labels.shape)}")
    print(f"Machine IDs (batch): {machine_ids[:3]}...")
    mn = specs.min()
    mx = specs.max()
    print(f"Value range: [{mn:.2f}, {mx:.2f}]")
    
    test_dataset = DCASE2020Task2TestDataset(
        root=root_str,
        machine_type=MACHINE_TYPE_TEST,
        target_T=dataset.target_T,
    )
    test_joint = DCASE2020Task2TestDataset(
        root=root_str,
        machine_types=MACHINE_TYPES,
        target_T=dataset.target_T,
    )
    if len(MACHINE_TYPES) > 1:
        assert len(test_joint) > 0
        _, _, tmid = test_joint[0]
        assert COMPOSITE_ID_SEP in tmid, f"expected composite test id, got {tmid}"

    test_loader = make_dataloader(test_dataset, batch_size=min(32, len(test_dataset)))
    specs, labels, machine_ids = next(iter(test_loader))
    print(f"Test batch shape: {tuple(specs.shape)}  (B, 1, n_mels, T)")
    print(f"Label shape: {tuple(labels.shape)}")
    mn = specs.min()
    mx = specs.max()
    print(f"Value range: [{mn:.2f}, {mx:.2f}]")
    print("Smoke test OK.")