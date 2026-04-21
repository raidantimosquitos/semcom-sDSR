"""
Anomaly simulation utilities for AudDSR training.

- AnomalyMapGenerator: spectromorphic masks; per-call ``machine_type`` selects
  stationary vs non-stationary strategy.
- NonStationarySpectromorphicMaskStrategy / SpectromorphicMaskStrategy (alias):
  valve, slider-style stratified band × renewal, optional Perlin
- StationarySpectromorphicMaskStrategy: fan, pump, ToyConveyor, ToyCar
- AudioSpecificStrategy / LatentAlignedBandStrategy: aliases of the non-stationary class
- generate_fake_anomalies_distant: codebook-based feature replacement (distant +
  neighbor); distant mode skips a configurable nearest fraction (default 5 %)
- generate_fake_anomalies_uniform: uniform codebook sampling for masked positions
"""

from .anomaly_map import (
    AnomalyMapGenerator,
    AudioSpecificStrategy,
    LatentAlignedBandStrategy,
    NonStationarySpectromorphicMaskStrategy,
    STATIONARY_SPECTROMORPHIC_MACHINE_TYPES,
    SpectromorphicMaskStrategy,
    StationarySpectromorphicMaskStrategy,
    default_spectromorphic_perlin,
    uses_stationary_spectromorphic_mask,
)
from .anomaly_generation import (
    generate_fake_anomalies_distant,
    generate_fake_anomalies_uniform,
)

__all__ = [
    "AnomalyMapGenerator",
    "AudioSpecificStrategy",
    "LatentAlignedBandStrategy",
    "STATIONARY_SPECTROMORPHIC_MACHINE_TYPES",
    "NonStationarySpectromorphicMaskStrategy",
    "SpectromorphicMaskStrategy",
    "StationarySpectromorphicMaskStrategy",
    "default_spectromorphic_perlin",
    "generate_fake_anomalies_distant",
    "generate_fake_anomalies_uniform",
    "uses_stationary_spectromorphic_mask",
]
