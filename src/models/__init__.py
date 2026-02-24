"""
Model modules for sDSR.

- vq_vae: VQ-VAE-2 discrete autoencoder (stage 1)
- dsr: sDSR dual subspace re-projection (stage 2)
"""

from .sDSR.s_dsr import (
    sDSR,
    sDSRConfig,
)

__all__ = ["sDSR", "sDSRConfig"]
