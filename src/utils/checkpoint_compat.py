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

    # ------------------------------------------------------------------
    # Additional compatibility shims
    # ------------------------------------------------------------------
    # 1) Upscaler changed from a single Conv/ConvT module to a Sequential:
    #    old: "_upscale_coarse.weight"/".bias"
    #    new: "_upscale_coarse.1.weight"/".bias" (0 is Upsample, 1 is Conv2d)
    if "_upscale_coarse.weight" in state_dict and "_upscale_coarse.1.weight" not in state_dict:
        state_dict["_upscale_coarse.1.weight"] = state_dict.pop("_upscale_coarse.weight")
    if "_upscale_coarse.bias" in state_dict and "_upscale_coarse.1.bias" not in state_dict:
        state_dict["_upscale_coarse.1.bias"] = state_dict.pop("_upscale_coarse.bias")

    # 2) Residual blocks: older checkpoints were saved with conv bias=False, while the
    #    current implementation uses bias=True. Fill missing biases with zeros so
    #    strict load works.
    #
    #    Keys we expect in current model:
    #      "..._residual_stack._layers.<i>._block.1.bias" (3x3 conv)
    #      "..._residual_stack._layers.<i>._block.3.bias" (1x1 conv)
    for w_key in list(state_dict.keys()):
        if "._residual_stack._layers." not in w_key:
            continue
        if not (w_key.endswith("._block.1.weight") or w_key.endswith("._block.3.weight")):
            continue
        b_key = w_key[:-len("weight")] + "bias"
        if b_key in state_dict:
            continue
        w = state_dict[w_key]
        # Conv weights are (out_channels, in_channels, kH, kW)
        if hasattr(w, "shape") and len(getattr(w, "shape", ())) >= 1:
            state_dict[b_key] = w.new_zeros((int(w.shape[0]),))
    return state_dict
