"""
Ogg page parsing + payload-only BER/BSC corruption with CRC recomputation.

This module supports Ogg container streams such as Ogg Opus produced by ffmpeg.
By default, callers can protect initial pages (e.g. OpusHead/OpusTags pages) and
flip only later page bodies, then recompute Ogg page CRCs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


OGG_CAPTURE = b"OggS"
OGG_FIXED_HEADER = 27


@dataclass(frozen=True)
class OggPage:
    index: int
    start: int
    end: int
    body_start: int
    body_end: int
    page_segments: int


def parse_ogg_pages(blob: bytes) -> List[OggPage]:
    pages: List[OggPage] = []
    p = 0
    i = 0
    n = len(blob)
    while p < n:
        if p + OGG_FIXED_HEADER > n:
            raise ValueError("Truncated Ogg page header.")
        if blob[p : p + 4] != OGG_CAPTURE:
            raise ValueError(f"Invalid Ogg capture pattern at offset {p}.")
        page_segments = int(blob[p + 26])
        seg_start = p + OGG_FIXED_HEADER
        seg_end = seg_start + page_segments
        if seg_end > n:
            raise ValueError("Truncated Ogg segment table.")
        body_len = int(sum(blob[seg_start:seg_end]))
        body_start = seg_end
        body_end = body_start + body_len
        if body_end > n:
            raise ValueError("Truncated Ogg page body.")
        pages.append(
            OggPage(
                index=i,
                start=p,
                end=body_end,
                body_start=body_start,
                body_end=body_end,
                page_segments=page_segments,
            )
        )
        p = body_end
        i += 1
    return pages


def _ogg_crc_table() -> np.ndarray:
    table = np.zeros(256, dtype=np.uint32)
    poly = np.uint32(0x04C11DB7)
    for i in range(256):
        r = np.uint32(i << 24)
        for _ in range(8):
            if (r & np.uint32(0x80000000)) != 0:
                r = np.uint32((r << 1) ^ poly)
            else:
                r = np.uint32(r << 1)
        table[i] = np.uint32(r & np.uint32(0xFFFFFFFF))
    return table


_CRC_TABLE = _ogg_crc_table()


def ogg_crc32(page_bytes: bytes) -> int:
    crc = np.uint32(0)
    arr = np.frombuffer(page_bytes, dtype=np.uint8)
    for b in arr:
        idx = int(((crc >> np.uint32(24)) ^ np.uint32(int(b))) & np.uint32(0xFF))
        crc = np.uint32(((crc << np.uint32(8)) ^ _CRC_TABLE[idx]) & np.uint32(0xFFFFFFFF))
    return int(crc)


def rewrite_page_crc(blob: bytearray, page: OggPage) -> None:
    page_bytes = bytearray(blob[page.start : page.end])
    # CRC field at offset 22..25 in page header
    page_bytes[22:26] = b"\x00\x00\x00\x00"
    c = ogg_crc32(bytes(page_bytes))
    page_bytes[22:26] = int(c).to_bytes(4, byteorder="little", signed=False)
    blob[page.start : page.end] = page_bytes


def verify_ogg_page_crc(blob: bytes, page: OggPage) -> bool:
    page_bytes = bytearray(blob[page.start : page.end])
    if len(page_bytes) < OGG_FIXED_HEADER:
        return False
    crc_rx = int.from_bytes(page_bytes[22:26], byteorder="little", signed=False)
    page_bytes[22:26] = b"\x00\x00\x00\x00"
    crc_calc = ogg_crc32(bytes(page_bytes))
    return int(crc_rx) == int(crc_calc)


def bitflip_ogg_payload_pages(
    blob: bytes,
    *,
    ber: float,
    protect_first_pages: int = 2,
    seed: int | None = None,
) -> bytes:
    """
    Flip bits in Ogg page bodies only (after protected leading pages), then
    recompute CRC on every modified page.
    """
    p = float(ber)
    if p <= 0.0:
        return blob
    p = min(p, 1.0)
    out = bytearray(blob)
    pages = parse_ogg_pages(blob)
    rng = np.random.default_rng(seed)

    n_protect = max(0, int(protect_first_pages))
    for pg in pages:
        if pg.index < n_protect:
            continue
        body = np.frombuffer(bytes(out[pg.body_start : pg.body_end]), dtype=np.uint8).copy()
        if body.size:
            bits = np.unpackbits(body, bitorder="little")
            flips = rng.random(bits.size) < p
            bits ^= flips.astype(np.uint8)
            body_rx = np.packbits(bits, bitorder="little")
            out[pg.body_start : pg.body_end] = body_rx.astype(np.uint8).tobytes()
        rewrite_page_crc(out, pg)
    return bytes(out)

