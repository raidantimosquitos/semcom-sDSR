"""
Anomaly simulation utilities for AudDSR training.

- AnomalyMapGenerator: spectromorphic masks.
- SpectromorphicMaskStrategy: band / full-width multi-burst / Perlin masks
"""

from .anomaly_map import (
    AnomalyMapGenerator,
    SpectromorphicMaskStrategy,
)

__all__ = [
    "AnomalyMapGenerator",
    "SpectromorphicMaskStrategy",
]
