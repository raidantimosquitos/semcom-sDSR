"""
Anomaly simulation utilities for AudDSR training.

- AnomalyMapGenerator: spectromorphic masks.
- SpectromorphicMaskStrategy: rectangular band masks + optional Perlin
"""

from .anomaly_map import (
    AnomalyMapGenerator,
    SpectromorphicMaskStrategy,
)

__all__ = [
    "AnomalyMapGenerator",
    "SpectromorphicMaskStrategy",
]
