"""
Anomaly simulation utilities for AudDSR training.

- AnomalyMapGenerator: Perlin + machine-specific anomaly mask generation
- PerlinNoiseStrategy, SliderSpecificStrategy (alias AudioSpecificStrategy), ToyCar,
  MachineSpecificStrategy: strategies
- generate_fake_anomalies_distant: generic codebook-based feature replacement
- generate_fake_anomalies_uniform: uniform codebook sampling for masked positions
"""

from .anomaly_map import (
    AnomalyMapGenerator,
    AudioSpecificStrategy,
    MachineSpecificStrategy,
    PerlinNoiseStrategy,
    PlaceholderMachineSpecificStrategy,
    SliderSpecificStrategy,
    ToyCarSpecificStrategy,
    ToyConveyorSpecificStrategy,
    PumpSpecificStrategy,
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
    "PlaceholderMachineSpecificStrategy",
    "SliderSpecificStrategy",
    "ToyCarSpecificStrategy",
    "ToyConveyorSpecificStrategy",
    "PumpSpecificStrategy",
    "generate_fake_anomalies_distant",
    "generate_fake_anomalies_uniform",
]
