"""
Pack/unpack VQ codebook indices to/from a binary bitstream for GNURadio.

Format: 4-byte num_clips (little-endian), then num_clips frames.
Each frame: coarse indices (row-major), then fine indices (row-major).
Bits packed LSB-first within each index, then byte-packed (first byte = bits 0-7 of stream).
"""

from __future__ import annotations

import struct
from typing import Tuple

import torch


def frame_size_bytes(H_coarse: int, W_coarse: int, H_fine: int, W_fine: int, bits_coarse: int = 10, bits_fine: int = 12) -> int:
    """Number of bytes per clip frame."""
    n_coarse = H_coarse * W_coarse
    n_fine = H_fine * W_fine
    total_bits = n_coarse * bits_coarse + n_fine * bits_fine
    return (total_bits + 7) // 8


def pack_indices_to_frame(
    indices_coarse: torch.Tensor,
    indices_fine: torch.Tensor,
    bits_coarse: int = 10,
    bits_fine: int = 12,
) -> bytes:
    """
    Pack one clip's indices into a byte string (one frame).

    Args:
        indices_coarse: (H_coarse, W_coarse) or (1, H_coarse, W_coarse) long
        indices_fine: (H_fine, W_fine) or (1, H_fine, W_fine) long
        bits_coarse: bits per coarse index (e.g. 10 for 1024 codebook)
        bits_fine: bits per fine index (e.g. 12 for 4096 codebook)

    Returns:
        bytes: packed frame (length = frame_size_bytes(...))
    """
    if indices_coarse.dim() == 3:
        indices_coarse = indices_coarse.squeeze(0)
    if indices_fine.dim() == 3:
        indices_fine = indices_fine.squeeze(0)
    coarse_flat = indices_coarse.flatten().cpu().numpy()
    fine_flat = indices_fine.flatten().cpu().numpy()

    bits: list[int] = []
    for idx in coarse_flat:
        for b in range(bits_coarse):
            bits.append((int(idx) >> b) & 1)
    for idx in fine_flat:
        for b in range(bits_fine):
            bits.append((int(idx) >> b) & 1)

    n_bytes = (len(bits) + 7) // 8
    out = bytearray(n_bytes)
    for i, bit in enumerate(bits):
        if bit:
            out[i // 8] |= 1 << (i % 8)
    return bytes(out)


def unpack_frame_to_indices(
    frame: bytes,
    H_coarse: int,
    W_coarse: int,
    H_fine: int,
    W_fine: int,
    bits_coarse: int = 10,
    bits_fine: int = 12,
    device: torch.device | str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Unpack one frame (bytes) to index tensors.

    Args:
        frame: packed bytes (length must equal frame_size_bytes(...))
        H_coarse, W_coarse, H_fine, W_fine: spatial dimensions
        bits_coarse, bits_fine: bits per index
        device: torch device for output tensors

    Returns:
        indices_coarse: (1, H_coarse, W_coarse) long
        indices_fine: (1, H_fine, W_fine) long
    """
    expected = frame_size_bytes(H_coarse, W_coarse, H_fine, W_fine, bits_coarse, bits_fine)
    if len(frame) < expected:
        raise ValueError(f"Frame has {len(frame)} bytes, expected {expected}")

    bits: list[int] = []
    for byte in frame:
        for b in range(8):
            bits.append((byte >> b) & 1)

    n_coarse = H_coarse * W_coarse
    n_fine = H_fine * W_fine
    pos = 0

    coarse_list: list[int] = []
    for _ in range(n_coarse):
        v = 0
        for b in range(bits_coarse):
            if pos < len(bits):
                v |= bits[pos] << b
            pos += 1
        coarse_list.append(v)

    fine_list: list[int] = []
    for _ in range(n_fine):
        v = 0
        for b in range(bits_fine):
            if pos < len(bits):
                v |= bits[pos] << b
            pos += 1
        fine_list.append(v)

    indices_coarse = torch.tensor(coarse_list, dtype=torch.long, device=device).view(1, H_coarse, W_coarse)
    indices_fine = torch.tensor(fine_list, dtype=torch.long, device=device).view(1, H_fine, W_fine)
    return indices_coarse, indices_fine


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
