"""
Anomaly simulation utilities for AudDSR training.

- AnomalyMapGenerator: anomaly mask generation with strategy selection
- SpectromorphicMaskStrategy: stratified mel band × renewal, optional wideband
  periodic bursts, optional Perlin
- AudioSpecificStrategy / LatentAlignedBandStrategy: backward-compatible aliases
- generate_fake_anomalies_distant: codebook-based feature replacement (distant +
  neighbor); distant mode skips a configurable nearest fraction (default 5 %)
- generate_fake_anomalies_uniform: uniform codebook sampling for masked positions
"""

from .anomaly_map import (
    AnomalyMapGenerator,
    AudioSpecificStrategy,
    LatentAlignedBandStrategy,
    SpectromorphicMaskStrategy,
)
from .anomaly_generation import (
    generate_fake_anomalies_distant,
    generate_fake_anomalies_uniform,
)

__all__ = [
    "AnomalyMapGenerator",
    "AudioSpecificStrategy",
    "LatentAlignedBandStrategy",
    "SpectromorphicMaskStrategy",
    "generate_fake_anomalies_distant",
    "generate_fake_anomalies_uniform",
]
