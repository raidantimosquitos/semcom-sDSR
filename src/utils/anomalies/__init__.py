"""
Anomaly simulation utilities for AudDSR training.

- AnomalyMapGenerator: spectromorphic masks.
- SpectromorphicMaskStrategy: valve, slider-style linear-Hz band × renewal, optional Perlin
"""

from .anomaly_map import (
    AnomalyMapGenerator,
    SpectromorphicMaskStrategy,
)

__all__ = [
    "AnomalyMapGenerator",
    "SpectromorphicMaskStrategy",
]
