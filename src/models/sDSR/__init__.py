"""
sDSR: Spectrogram Dual Subspace Re-projection for anomaly detection.

Modules:
- AudDSR: top-level model
- ObjectSpecificDecoder: trainable decoder, reconstructs X_S
- AnomalyDetectionModule: UNet segmentation head
- AnomalyGeneration: DSR-specific VQ feature augmentation
- FocalLoss: segmentation loss
- sDSRConfig: configuration dataclass

Anomaly map utilities (AnomalyMapGenerator, etc.) live in src.utils.anomalies;
re-exported here for convenience.
"""

from .s_dsr import sDSR, sDSRConfig
from .object_specific_decoder import ObjectSpecificDecoder
from .anomaly_detection import AnomalyDetectionModule
from .anomaly_generation import AnomalyGeneration
from .subspace_restriction import SubspaceRestrictionModule, SubspaceRestrictionNetwork
from .loss import FocalLoss

# Re-export from utils for backward compatibility
from ...utils.anomalies import (
    AnomalyMapGenerator,
    PerlinNoiseStrategy,
    AudioSpecificStrategy,
    generate_fake_anomalies_distant,
)

__all__ = [
    "sDSR",
    "sDSRConfig",
    "ObjectSpecificDecoder",
    "SubspaceRestrictionModule",
    "SubspaceRestrictionNetwork",
    "AnomalyMapGenerator",
    "PerlinNoiseStrategy",
    "AudioSpecificStrategy",
    "AnomalyGeneration",
    "generate_fake_anomalies_distant",
    "AnomalyDetectionModule",
    "FocalLoss",
]
