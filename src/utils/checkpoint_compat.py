"""
Checkpoint compatibility: migrate state_dict keys from bot/top to fine/coarse.

When loading Stage 1 or Stage 2 checkpoints, old keys use _bot/_top;
new code uses _fine/_coarse. This module rewrites keys so old checkpoints still load.
"""

from __future__ import annotations


def migrate_vq_vae_state_dict(state_dict: dict) -> dict:
    """
    Rewrite VQ-VAE (and sDSR) state_dict keys from bot/top to fine/coarse.
    Modifies in place and returns the same dict.
    Use when loading Stage 1 or full model (Stage 2) checkpoints.
    """
    replacements = [
        ("_encoder_bot.", "_encoder_fine."),
        ("_encoder_top.", "_encoder_coarse."),
        ("_decoder_bot.", "_decoder_fine."),
        ("_decoder_top.", "_decoder_coarse."),
        ("_vq_bot.", "_vq_fine."),
        ("_vq_top.", "_vq_coarse."),
        ("_pre_vq_conv_bot.", "_pre_vq_conv_fine."),
        ("_pre_vq_conv_top.", "_pre_vq_conv_coarse."),
        ("_subspace_bot.", "_subspace_fine."),
        ("_subspace_top.", "_subspace_coarse."),
    ]
    keys = list(state_dict.keys())
    for old_key in keys:
        new_key = old_key
        for old, new in replacements:
            new_key = new_key.replace(old, new)
        if new_key != old_key:
            state_dict[new_key] = state_dict.pop(old_key)
    return state_dict
