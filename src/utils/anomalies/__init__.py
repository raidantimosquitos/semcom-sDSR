"""
Anomaly simulation utilities for AudDSR training.

- AnomalyMapGenerator: anomaly mask generation with strategy selection; for
  ``audio_specific``, masks are chosen from per-sample ``machine_type`` (stationary
  vs non-stationary spectromorphic families).
- SpectromorphicMaskStrategy: non-stationary (valve, slider): stratified mel band ×
  renewal, optional Perlin
- StationarySpectromorphicMaskStrategy: fan, pump, ToyConveyor, ToyCar harmonics + AM + Perlin
- AudioSpecificStrategy / LatentAlignedBandStrategy: backward-compatible aliases
  to SpectromorphicMaskStrategy only
- generate_fake_anomalies_distant: codebook-based feature replacement (distant +
  neighbor); distant mode skips a configurable nearest fraction (default 5 %)
- generate_fake_anomalies_uniform: uniform codebook sampling for masked positions
"""

from .anomaly_map import (
    AnomalyMapGenerator,
    AudioSpecificStrategy,
    LatentAlignedBandStrategy,
    SpectromorphicMaskStrategy,
    STATIONARY_SPECTROMORPHIC_MACHINE_TYPES,
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
    "SpectromorphicMaskStrategy",
    "StationarySpectromorphicMaskStrategy",
    "default_spectromorphic_perlin",
    "generate_fake_anomalies_distant",
    "generate_fake_anomalies_uniform",
    "uses_stationary_spectromorphic_mask",
]
