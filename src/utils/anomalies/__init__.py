"""
Anomaly simulation utilities for AudDSR training.

- AnomalyMapGenerator: spectromorphic masks.
- SpectromorphicMaskStrategy: valve, slider-style linear-Hz band × renewal, optional Perlin
- generate_fake_anomalies_distant: codebook-based feature replacement (distant +
  neighbor); distant mode skips a configurable nearest fraction (default 5 %)
- generate_fake_anomalies_uniform: uniform codebook sampling for masked positions
"""

from .anomaly_map import (
    AnomalyMapGenerator,
    SpectromorphicMaskStrategy,
)
from .anomaly_generation import (
    generate_fake_anomalies_distant,
    generate_fake_anomalies_uniform,
)

__all__ = [
    "AnomalyMapGenerator",
    "SpectromorphicMaskStrategy",
    "generate_fake_anomalies_distant",
    "generate_fake_anomalies_uniform",
]
