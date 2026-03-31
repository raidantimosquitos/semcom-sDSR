"""
Packetization (framing) + CRC32 for byte payloads.

Purpose:
- Make byte-stream payloads (Huffman bitstreams, JPEG/Opus bytes, etc.) robust under channel errors.
- Residual bit errors become *frame erasures* (CRC fail) rather than desynchronizing decoders.

Frame format (fixed size):
  [payload_len: uint16 little-endian] [payload bytes] [pad zeros] [crc32: uint32 little-endian]

CRC32 is computed over: payload_len || payload || pad  (i.e., all bytes except the CRC field).
"""

from __future__ import annotations

from dataclasses import dataclass
import struct
import zlib


_LEN_FMT = "<H"  # uint16
_CRC_FMT = "<I"  # uint32
_LEN_SIZE = struct.calcsize(_LEN_FMT)
_CRC_SIZE = struct.calcsize(_CRC_FMT)


@dataclass(frozen=True)
class FrameStats:
    n_frames: int
    n_failed: int

    @property
    def fer(self) -> float:
        return (self.n_failed / self.n_frames) if self.n_frames else 0.0


def frame_size(frame_payload_bytes: int) -> int:
    fpb = int(frame_payload_bytes)
    if fpb <= 0:
        raise ValueError(f"frame_payload_bytes must be > 0, got {fpb}")
    return _LEN_SIZE + fpb + _CRC_SIZE


def packetize(payload: bytes, frame_payload_bytes: int) -> list[bytes]:
    fpb = int(frame_payload_bytes)
    if fpb <= 0:
        raise ValueError(f"frame_payload_bytes must be > 0, got {fpb}")
    out: list[bytes] = []
    pos = 0
    n = len(payload)
    while pos < n:
        chunk = payload[pos : pos + fpb]
        out.append(_build_frame(chunk, fpb))
        pos += fpb
    if n == 0:
        # represent empty payload as one empty frame (payload_len=0)
        out.append(_build_frame(b"", fpb))
    return out


def depacketize(frames_payloads: list[bytes], orig_len: int) -> bytes:
    blob = b"".join(frames_payloads)
    if orig_len < 0:
        raise ValueError(f"orig_len must be >= 0, got {orig_len}")
    return blob[:orig_len]


def transmit_frames_over_channel(
    frames: list[bytes],
    *,
    frame_payload_bytes: int,
    channel_txrx: callable,
    concealment: str = "zeros",
) -> tuple[list[bytes], FrameStats]:
    """
    Apply a byte-level channel to each frame (already CRC-protected).

    Args:
      frames: list of full encoded frames
      channel_txrx: function(frame_bytes)->frame_bytes (same length)
      concealment: currently only 'zeros' supported for failed frames
    Returns:
      rx_payloads: list of payload bytes (already length-correct per frame, with pad stripped)
      stats: FrameStats(n_frames, n_failed)
    """
    fpb = int(frame_payload_bytes)
    if concealment != "zeros":
        raise ValueError(f"Unsupported concealment '{concealment}' (only 'zeros' supported).")
    rx_payloads: list[bytes] = []
    failed = 0
    for fr in frames:
        rx_fr = channel_txrx(fr)
        ok, payload = check_and_strip_crc(rx_fr, frame_payload_bytes=fpb)
        if not ok:
            failed += 1
            payload = b"\x00" * len(payload)
        rx_payloads.append(payload)
    return rx_payloads, FrameStats(n_frames=len(frames), n_failed=failed)


def _build_frame(payload: bytes, frame_payload_bytes: int) -> bytes:
    fpb = int(frame_payload_bytes)
    if len(payload) > fpb:
        raise ValueError(f"payload too large: {len(payload)} > frame_payload_bytes={fpb}")
    pad_len = fpb - len(payload)
    header = struct.pack(_LEN_FMT, int(len(payload)))
    body = payload + (b"\x00" * pad_len)
    crc_val = zlib.crc32(header + body) & 0xFFFFFFFF
    crc = struct.pack(_CRC_FMT, int(crc_val))
    return header + body + crc


def check_and_strip_crc(frame: bytes, *, frame_payload_bytes: int) -> tuple[bool, bytes]:
    fpb = int(frame_payload_bytes)
    fs = frame_size(fpb)
    if len(frame) != fs:
        raise ValueError(f"frame has {len(frame)} bytes, expected {fs}")
    header = frame[:_LEN_SIZE]
    body = frame[_LEN_SIZE : _LEN_SIZE + fpb]
    crc_bytes = frame[_LEN_SIZE + fpb :]
    (payload_len,) = struct.unpack(_LEN_FMT, header)
    payload_len = int(payload_len)
    if payload_len < 0 or payload_len > fpb:
        return (False, body[: min(max(payload_len, 0), fpb)])
    (crc_rx,) = struct.unpack(_CRC_FMT, crc_bytes)
    crc_calc = zlib.crc32(header + body) & 0xFFFFFFFF
    ok = int(crc_rx) == int(crc_calc)
    return ok, body[:payload_len]

