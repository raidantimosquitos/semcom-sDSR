"""
Anomaly simulation utilities for AudDSR training.

- AnomalyMapGenerator: anomaly mask generation with strategy selection
- SpectromorphicMaskStrategy: four sub-strategies (full_band, multi_band,
  diffuse_rect, perlin) sampled per-item
- AudioSpecificStrategy / LatentAlignedBandStrategy: backward-compatible aliases
- generate_fake_anomalies_distant: codebook-based feature replacement (distant + neighbor)
- generate_fake_anomalies_uniform: uniform codebook sampling for masked positions
"""

from .anomaly_map import (
    AnomalyMapGenerator,
    AudioSpecificStrategy,
    LatentAlignedBandStrategy,
    MASK_PRESETS,
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
    "MASK_PRESETS",
    "SpectromorphicMaskStrategy",
    "generate_fake_anomalies_distant",
    "generate_fake_anomalies_uniform",
]
