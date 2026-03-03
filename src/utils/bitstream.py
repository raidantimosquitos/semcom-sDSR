"""
Pack/unpack VQ codebook indices to/from a binary bitstream for GNURadio.

Format: 4-byte num_clips (little-endian), then num_clips frames.
Each frame: top indices (10 bits each, row-major), then bottom indices (12 bits each, row-major).
Bits packed LSB-first within each index, then byte-packed (first byte = bits 0-7 of stream).
"""

from __future__ import annotations

import struct
from typing import Tuple

import torch


def frame_size_bytes(H_top: int, W_top: int, H_bot: int, W_bot: int, bits_top: int = 10, bits_bot: int = 12) -> int:
    """Number of bytes per clip frame."""
    n_top = H_top * W_top
    n_bot = H_bot * W_bot
    total_bits = n_top * bits_top + n_bot * bits_bot
    return (total_bits + 7) // 8


def pack_indices_to_frame(
    indices_top: torch.Tensor,
    indices_bot: torch.Tensor,
    bits_top: int = 10,
    bits_bot: int = 12,
) -> bytes:
    """
    Pack one clip's indices into a byte string (one frame).

    Args:
        indices_top: (H_top, W_top) or (1, H_top, W_top) long, values in [0, 2^bits_top)
        indices_bot: (H_bot, W_bot) or (1, H_bot, W_bot) long, values in [0, 2^bits_bot)
        bits_top: bits per top index (e.g. 10 for 1024 codebook)
        bits_bot: bits per bottom index (e.g. 12 for 4096 codebook)

    Returns:
        bytes: packed frame (length = frame_size_bytes(...))
    """
    if indices_top.dim() == 3:
        indices_top = indices_top.squeeze(0)
    if indices_bot.dim() == 3:
        indices_bot = indices_bot.squeeze(0)
    top_flat = indices_top.flatten().cpu().numpy()
    bot_flat = indices_bot.flatten().cpu().numpy()

    bits: list[int] = []
    for idx in top_flat:
        for b in range(bits_top):
            bits.append((int(idx) >> b) & 1)
    for idx in bot_flat:
        for b in range(bits_bot):
            bits.append((int(idx) >> b) & 1)

    n_bytes = (len(bits) + 7) // 8
    out = bytearray(n_bytes)
    for i, bit in enumerate(bits):
        if bit:
            out[i // 8] |= 1 << (i % 8)
    return bytes(out)


def unpack_frame_to_indices(
    frame: bytes,
    H_top: int,
    W_top: int,
    H_bot: int,
    W_bot: int,
    bits_top: int = 10,
    bits_bot: int = 12,
    device: torch.device | str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Unpack one frame (bytes) to index tensors.

    Args:
        frame: packed bytes (length must equal frame_size_bytes(...))
        H_top, W_top, H_bot, W_bot: spatial dimensions
        bits_top, bits_bot: bits per index
        device: torch device for output tensors

    Returns:
        indices_top: (1, H_top, W_top) long
        indices_bot: (1, H_bot, W_bot) long
    """
    expected = frame_size_bytes(H_top, W_top, H_bot, W_bot, bits_top, bits_bot)
    if len(frame) < expected:
        raise ValueError(f"Frame has {len(frame)} bytes, expected {expected}")

    bits: list[int] = []
    for byte in frame:
        for b in range(8):
            bits.append((byte >> b) & 1)

    n_top = H_top * W_top
    n_bot = H_bot * W_bot
    pos = 0

    top_list: list[int] = []
    for _ in range(n_top):
        v = 0
        for b in range(bits_top):
            if pos < len(bits):
                v |= bits[pos] << b
            pos += 1
        top_list.append(v)

    bot_list: list[int] = []
    for _ in range(n_bot):
        v = 0
        for b in range(bits_bot):
            if pos < len(bits):
                v |= bits[pos] << b
            pos += 1
        bot_list.append(v)

    indices_top = torch.tensor(top_list, dtype=torch.long, device=device).view(1, H_top, W_top)
    indices_bot = torch.tensor(bot_list, dtype=torch.long, device=device).view(1, H_bot, W_bot)
    return indices_top, indices_bot


def write_bitstream_file(
    path: str,
    frames: list[bytes],
) -> None:
    """
    Write bitstream file: 4-byte num_clips (LE) then concatenated frames.
    """
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(frames)))
        for frame in frames:
            f.write(frame)


def read_bitstream_file(path: str, frame_size: int) -> Tuple[int, list[bytes]]:
    """
    Read bitstream file. Returns (num_clips, list of frame bytes).
    frame_size: bytes per frame (e.g. from frame_size_bytes(...)).
    """
    with open(path, "rb") as f:
        num_clips = struct.unpack("<I", f.read(4))[0]
        payload = f.read()
    if len(payload) != num_clips * frame_size:
        raise ValueError(
            f"File has {len(payload)} bytes, expected num_clips={num_clips} * frame_size={frame_size} = {num_clips * frame_size}"
        )
    frames = [payload[i * frame_size : (i + 1) * frame_size] for i in range(num_clips)]
    return num_clips, frames
