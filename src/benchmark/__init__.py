"""Benchmark utilities for detection latency studies."""

from src.benchmark.timing import StageTimes, aggregate_times, empty_stage_times, sync_device

__all__ = [
    "StageTimes",
    "aggregate_times",
    "empty_stage_times",
    "sync_device",
]
