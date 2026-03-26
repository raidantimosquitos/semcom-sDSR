"""
Pack/unpack VQ codebook indices to/from a binary bitstream for GNURadio.

File format:
  - Raw frame bytes only (no headers, no CRC).
  - One `.bin` file corresponds to exactly one clip (one frame).

Each frame: coarse indices (row-major), then fine indices (row-major).
Bits packed LSB-first within each index, then byte-packed (first byte = bits 0-7 of stream).
"""

from __future__ import annotations

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


def write_frame_file(path: str, frame: bytes) -> None:
    """Write a single clip frame as raw bytes (no headers)."""
    with open(path, "wb") as f:
        f.write(frame)


def read_frame_file(path: str, expected_frame_size: int | None = None) -> bytes:
    """
    Read a single clip frame from a raw `.bin`.

    If expected_frame_size is provided, raises ValueError when the file length differs.
    """
    with open(path, "rb") as f:
        b = f.read()
    if expected_frame_size is not None and len(b) != expected_frame_size:
        raise ValueError(f"{path}: expected {expected_frame_size} bytes, got {len(b)}")
    return b


def read_frames_from_raw_stream(path: str, frame_size: int) -> list[bytes]:
    """
    Read a raw bitstream containing concatenated frames (no headers).
    This is only for convenience; the recommended workflow is one `.bin` per clip.
    """
    with open(path, "rb") as f:
        blob = f.read()
    if frame_size <= 0:
        raise ValueError(f"frame_size must be > 0, got {frame_size}")
    if len(blob) % frame_size != 0:
        raise ValueError(f"{path}: byte length {len(blob)} is not a multiple of frame_size={frame_size}")
    n = len(blob) // frame_size
    return [blob[i * frame_size : (i + 1) * frame_size] for i in range(n)]
