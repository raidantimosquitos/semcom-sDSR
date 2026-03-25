"""
Anomaly simulation utilities for AudDSR training.

- AnomalyMapGenerator: audio-specific anomaly mask generation (paper-style)
- AudioSpecificStrategy / SliderSpecificStrategy: band + disjoint time intervals
- generate_fake_anomalies_distant: generic codebook-based feature replacement
- generate_fake_anomalies_uniform: uniform codebook sampling for masked positions
"""

from .anomaly_map import (
    AnomalyMapGenerator,
    AudioSpecificStrategy,
)
from .anomaly_generation import (
    generate_fake_anomalies_distant,
    generate_fake_anomalies_uniform,
)

__all__ = [
    "AnomalyMapGenerator",
    "AudioSpecificStrategy",
    "generate_fake_anomalies_distant",
    "generate_fake_anomalies_uniform",
]
