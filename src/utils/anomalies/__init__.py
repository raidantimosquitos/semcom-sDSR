"""
Anomaly simulation utilities for AudDSR training.

- AnomalyMapGenerator: Perlin + audio-specific anomaly mask generation
- PerlinNoiseStrategy, AudioSpecificStrategy: individual strategies
- generate_fake_anomalies_distant: generic codebook-based feature replacement
- generate_fake_anomalies_uniform: uniform codebook sampling for masked positions
"""

from .anomaly_map import (
    AnomalyMapGenerator,
    PerlinNoiseStrategy,
    AudioSpecificStrategy,
)
from .anomaly_generation import (
    generate_fake_anomalies_distant,
    generate_fake_anomalies_uniform,
)

__all__ = [
    "AnomalyMapGenerator",
    "PerlinNoiseStrategy",
    "AudioSpecificStrategy",
    "generate_fake_anomalies_distant",
    "generate_fake_anomalies_uniform",
]
