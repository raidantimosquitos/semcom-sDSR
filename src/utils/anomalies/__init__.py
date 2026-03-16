"""
Anomaly simulation utilities for AudDSR training.

- AnomalyMapGenerator: Perlin + audio-specific + machine_specific anomaly mask generation
- PerlinNoiseStrategy, AudioSpecificStrategy, MachineSpecificStrategy: individual strategies
- generate_fake_anomalies_distant: generic codebook-based feature replacement
- generate_fake_anomalies_uniform: uniform codebook sampling for masked positions
"""

from .anomaly_map import (
    AnomalyMapGenerator,
    AudioSpecificStrategy,
    MachineSpecificStrategy,
    PerlinNoiseStrategy,
)
from .anomaly_generation import (
    generate_fake_anomalies_distant,
    generate_fake_anomalies_uniform,
)

__all__ = [
    "AnomalyMapGenerator",
    "AudioSpecificStrategy",
    "MachineSpecificStrategy",
    "PerlinNoiseStrategy",
    "generate_fake_anomalies_distant",
    "generate_fake_anomalies_uniform",
]
