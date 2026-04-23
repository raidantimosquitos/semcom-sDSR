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

from .s_dsr import sDSR, sDSRConfig, SAMPLING_PRESETS
from .object_specific_decoder import ObjectSpecificDecoder
from .anomaly_detection import AnomalyDetectionModule
from .anomaly_generation import (
    AnomalyGeneration,
    project_spec_mask_to_latent_binary,
    upsample_latent_mask_to_spec,
)
from .subspace_restriction import SubspaceRestrictionModule, SubspaceRestrictionNetwork
from .loss import FocalLoss

# Re-export from utils for backward compatibility
from ...utils.anomalies import (
    AnomalyMapGenerator,
    generate_fake_anomalies_distant,
)

__all__ = [
    "sDSR",
    "sDSRConfig",
    "SAMPLING_PRESETS",
    "ObjectSpecificDecoder",
    "SubspaceRestrictionModule",
    "SubspaceRestrictionNetwork",
    "AnomalyMapGenerator",
    "AnomalyGeneration",
    "project_spec_mask_to_latent_binary",
    "upsample_latent_mask_to_spec",
    "generate_fake_anomalies_distant",
    "AnomalyDetectionModule",
    "FocalLoss",
]
