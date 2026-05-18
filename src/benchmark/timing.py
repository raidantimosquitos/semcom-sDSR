"""Timing helpers for staged detection-latency benchmarks."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
import torch

STAGE_KEYS = (
    "t_load_wav",
    "t_mel",
    "t_tx_codec",
    "t_channel",
    "t_rx_codec",
    "t_detector",
    "t_score",
)


def sync_device(device: torch.device | str) -> None:
    dev = torch.device(device)
    if dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(dev)


@dataclass
class StageTimes:
    """Per-stage durations in seconds."""

    t_load_wav: float = 0.0
    t_mel: float = 0.0
    t_tx_codec: float = 0.0
    t_channel: float = 0.0
    t_rx_codec: float = 0.0
    t_detector: float = 0.0
    t_score: float = 0.0

    def stage_sum(self) -> float:
        return sum(getattr(self, k) for k in STAGE_KEYS)

    @property
    def t_e2e(self) -> float:
        return self.stage_sum()

    def to_ms_dict(self) -> dict[str, float]:
        out = {k: getattr(self, k) * 1000.0 for k in STAGE_KEYS}
        out["t_e2e"] = self.t_e2e * 1000.0
        return out


def empty_stage_times() -> StageTimes:
    return StageTimes()


class StageTimer:
    """Accumulate perf_counter durations into :class:`StageTimes`."""

    def __init__(self) -> None:
        self.times = empty_stage_times()
        self._stack: list[tuple[str, float]] = []

    def start(self, stage: str) -> None:
        if stage not in STAGE_KEYS:
            raise ValueError(f"Unknown stage: {stage}")
        self._stack.append((stage, time.perf_counter()))

    def stop(self, stage: str) -> None:
        if not self._stack or self._stack[-1][0] != stage:
            raise RuntimeError(f"stop({stage}) does not match start({self._stack[-1][0] if self._stack else None})")
        name, t0 = self._stack.pop()
        dt = time.perf_counter() - t0
        setattr(self.times, name, getattr(self.times, name) + dt)

    def add(self, stage: str, seconds: float) -> None:
        if stage not in STAGE_KEYS:
            raise ValueError(f"Unknown stage: {stage}")
        setattr(self.times, stage, getattr(self.times, stage) + seconds)


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    w = pos - lo
    return sorted_vals[lo] * (1.0 - w) + sorted_vals[hi] * w


def aggregate_times(samples: list[StageTimes]) -> dict[str, dict[str, float]]:
    """
    Aggregate a list of :class:`StageTimes` into mean / median / p95 per stage (seconds).

    Returns:
        {stage_name: {"mean": ..., "median": ..., "p95": ...}, "t_e2e": {...}}
    """
    if not samples:
        return {}

    keys = list(STAGE_KEYS) + ["t_e2e"]
    out: dict[str, dict[str, float]] = {}
    for key in keys:
        if key == "t_e2e":
            vals = [s.t_e2e for s in samples]
        else:
            vals = [getattr(s, key) for s in samples]
        sorted_vals = sorted(vals)
        out[key] = {
            "mean": float(sum(vals) / len(vals)),
            "median": float(_percentile(sorted_vals, 0.5)),
            "p95": float(_percentile(sorted_vals, 0.95)),
        }
    return out


@dataclass
class PipelineResult:
    times: StageTimes
    payload_bytes: int = 0
    decode_ok: bool = True
    anomaly_score: float = 0.0
    extra: dict[str, float] = field(default_factory=dict)
