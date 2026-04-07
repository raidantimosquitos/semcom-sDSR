"""
JPEG payload masking and BER/BSC corruption helpers.

The mask is conservative by default:
- protected: all bytes outside entropy-coded scan data
- protected: JPEG restart markers in scan data (FFD0..FFD7)
- flip-eligible: entropy-coded bytes inside scans

Use `prefix_protect_bytes` to keep any custom outer header untouched (e.g. the
8-byte min/max header used by the JPEG baseline script).
"""

from __future__ import annotations

import numpy as np


_NO_LEN_MARKERS = set([0x01, 0xD8, 0xD9] + list(range(0xD0, 0xD8)))


def _read_marker_code(data: bytes, p: int) -> tuple[int | None, int]:
    n = len(data)
    if p >= n or data[p] != 0xFF:
        return None, p
    while p < n and data[p] == 0xFF:
        p += 1
    if p >= n:
        return None, p
    return int(data[p]), p + 1


def jpeg_protect_mask(blob: bytes, *, prefix_protect_bytes: int = 0) -> np.ndarray:
    """
    Returns a boolean protect mask for `blob`.
    True = protected, False = eligible for bit flips.
    """
    n_total = len(blob)
    mask = np.ones(n_total, dtype=bool)
    pfx = max(0, int(prefix_protect_bytes))
    if pfx >= n_total:
        return mask

    data = blob[pfx:]
    n = len(data)
    local = np.ones(n, dtype=bool)
    if n < 2:
        return mask
    if not (data[0] == 0xFF and data[1] == 0xD8):
        raise ValueError("Not a JPEG stream (missing SOI marker).")

    p = 2  # after SOI
    while p < n:
        # seek next marker prefix
        while p < n and data[p] != 0xFF:
            p += 1
        if p >= n:
            break

        m, p_after_marker = _read_marker_code(data, p)
        if m is None:
            break
        p = p_after_marker

        if m in _NO_LEN_MARKERS:
            if m == 0xD9:  # EOI
                break
            continue

        if p + 2 > n:
            break
        seg_len = (int(data[p]) << 8) | int(data[p + 1])
        if seg_len < 2:
            break
        seg_end = p + seg_len
        if seg_end > n:
            break
        p = seg_end

        # Non-SOS segments are fully protected.
        if m != 0xDA:
            continue

        # SOS found: parse entropy-coded segment until a true marker.
        while p < n:
            b = int(data[p])
            if b != 0xFF:
                local[p] = False
                p += 1
                continue

            if p + 1 >= n:
                break
            b2 = int(data[p + 1])
            if b2 == 0x00:
                # byte-stuffed 0xFF data byte => both bytes are in entropy stream
                local[p] = False
                local[p + 1] = False
                p += 2
                continue
            if b2 == 0xFF:
                # fill byte between markers
                p += 1
                continue
            if 0xD0 <= b2 <= 0xD7:
                # restart marker: keep protected
                p += 2
                continue
            # Reached a real marker (EOI, next SOS, DNL, APP, etc.); hand control
            # back to outer marker loop.
            break

    mask[pfx:] = local
    return mask


def bitflip_with_mask(blob: bytes, *, ber: float, protect_mask: np.ndarray, seed: int | None = None) -> bytes:
    """
    Apply i.i.d. BSC flips with probability `ber` on bytes where protect_mask=False.
    """
    p = float(ber)
    if p <= 0.0:
        return blob
    p = min(p, 1.0)

    m = np.asarray(protect_mask, dtype=bool).reshape(-1)
    if m.size != len(blob):
        raise ValueError(f"protect_mask size mismatch: {m.size} vs blob {len(blob)}")
    flip_idx = np.flatnonzero(~m)
    if flip_idx.size == 0:
        return blob

    rng = np.random.default_rng(seed)
    out = np.frombuffer(blob, dtype=np.uint8).copy()
    sel = out[flip_idx]
    bits = np.unpackbits(sel, bitorder="little")
    flips = rng.random(bits.size) < p
    bits ^= flips.astype(np.uint8)
    out[flip_idx] = np.packbits(bits, bitorder="little")
    return out.tobytes()


def bitflip_jpeg_entropy_payload(
    blob: bytes,
    *,
    ber: float,
    prefix_protect_bytes: int = 0,
    seed: int | None = None,
) -> bytes:
    """
    JPEG-aware BSC corruption helper for entropy payload only.
    """
    mask = jpeg_protect_mask(blob, prefix_protect_bytes=prefix_protect_bytes)
    return bitflip_with_mask(blob, ber=ber, protect_mask=mask, seed=seed)

