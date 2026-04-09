"""
Anomaly simulation utilities for AudDSR training.

- AnomalyMapGenerator: anomaly mask generation with strategy selection
- LatentAlignedBandStrategy: full-width horizontal bands aligned to VQ-VAE latent grid
- AudioSpecificStrategy: backward-compatible alias for LatentAlignedBandStrategy
- generate_fake_anomalies_distant: codebook-based feature replacement (distant + neighbor)
- generate_fake_anomalies_uniform: uniform codebook sampling for masked positions
"""

from .anomaly_map import (
    AnomalyMapGenerator,
    AudioSpecificStrategy,
    LatentAlignedBandStrategy,
)
from .anomaly_generation import (
    generate_fake_anomalies_distant,
    generate_fake_anomalies_uniform,
)

__all__ = [
    "AnomalyMapGenerator",
    "AudioSpecificStrategy",
    "LatentAlignedBandStrategy",
    "generate_fake_anomalies_distant",
    "generate_fake_anomalies_uniform",
]
