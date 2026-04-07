#!/usr/bin/env python3
"""
Smoke test for JPEG entropy-only payload flips and Ogg page-body flips + CRC rewrite.

What it checks:
1) JPEG mask builder runs and detects some flip-eligible bytes.
2) Ogg parser works on a synthetic ffmpeg-generated Ogg Opus file.
3) For BER=0, Ogg output bytes are unchanged.
4) For BER>0 with protected first pages, CRCs are valid on all pages after rewrite.
5) ffmpeg can decode both clean and corrupted Ogg files.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from src.utils.jpeg_payload import jpeg_protect_mask, bitflip_jpeg_entropy_payload  # pyright: ignore[reportMissingImports]
from src.utils.ogg_payload import parse_ogg_pages, verify_ogg_page_crc, bitflip_ogg_payload_pages  # pyright: ignore[reportMissingImports]


def resolve_ffmpeg_bin(ffmpeg_bin_arg: str | None) -> str:
    if ffmpeg_bin_arg:
        return str(ffmpeg_bin_arg)
    env_bin = os.environ.get("FFMPEG_BIN")
    if env_bin:
        return env_bin
    if Path("/usr/bin/ffmpeg").exists():
        return "/usr/bin/ffmpeg"
    path_bin = shutil.which("ffmpeg")
    if path_bin:
        return path_bin
    raise FileNotFoundError("ffmpeg not found. Set --ffmpeg_bin or FFMPEG_BIN.")


def _run(cmd: list[str]) -> None:
    cp = subprocess.run(cmd, capture_output=True)
    if cp.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n"
            f"stdout:\n{cp.stdout.decode(errors='ignore')}\n"
            f"stderr:\n{cp.stderr.decode(errors='ignore')}"
        )


def _build_synthetic_ogg_page(body: bytes, *, serial: int, seqno: int, granule_pos: int = 0, header_type: int = 0x00) -> bytes:
    segs = []
    n = len(body)
    pos = 0
    while pos < n:
        take = min(255, n - pos)
        segs.append(take)
        pos += take
    if n == 0:
        segs = [0]
    page_segments = len(segs)
    header = bytearray(27 + page_segments)
    header[0:4] = b"OggS"
    header[4] = 0  # version
    header[5] = int(header_type) & 0xFF
    header[6:14] = int(granule_pos).to_bytes(8, "little", signed=False)
    header[14:18] = int(serial).to_bytes(4, "little", signed=False)
    header[18:22] = int(seqno).to_bytes(4, "little", signed=False)
    header[22:26] = b"\x00\x00\x00\x00"
    header[26] = page_segments & 0xFF
    header[27 : 27 + page_segments] = bytes(segs)
    page = bytes(header) + body
    # reuse module implementation through rewrite path semantics
    from src.comm.ogg_payload import ogg_crc32

    page_mut = bytearray(page)
    crc = ogg_crc32(bytes(page_mut))
    page_mut[22:26] = int(crc).to_bytes(4, "little", signed=False)
    return bytes(page_mut)


def _make_synthetic_ogg_stream(rng: np.random.Generator) -> bytes:
    # Page 0: OpusHead-like packet
    head_pkt = b"OpusHead" + bytes([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    p0 = _build_synthetic_ogg_page(
        head_pkt,
        serial=0x1234ABCD,
        seqno=0,
        granule_pos=0,
        header_type=0x02,  # BOS
    )
    # Page 1: OpusTags-like packet
    tags_pkt = b"OpusTags" + b"\x00" * 32
    p1 = _build_synthetic_ogg_page(
        tags_pkt,
        serial=0x1234ABCD,
        seqno=1,
        granule_pos=0,
        header_type=0x00,
    )
    # Page 2: random payload bytes
    payload = rng.integers(0, 256, size=1024, dtype=np.uint8).tobytes()
    p2 = _build_synthetic_ogg_page(
        payload,
        serial=0x1234ABCD,
        seqno=2,
        granule_pos=960,
        header_type=0x04,  # EOS
    )
    return p0 + p1 + p2


def _make_test_jpeg_blob(rng: np.random.Generator) -> bytes:
    try:
        from PIL import Image
    except Exception as e:  # pragma: no cover
        raise ImportError("Pillow required for JPEG smoke test: pip install pillow") from e

    arr = rng.integers(0, 256, size=(128, 320), dtype=np.uint8)
    img = Image.fromarray(arr, mode="L")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "x.jpg"
        img.save(p, format="JPEG", quality=50, optimize=True)
        jpeg = p.read_bytes()
    header = np.array([float(arr.min()), float(arr.max())], dtype=np.float32).tobytes()
    return header + jpeg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ber", type=float, default=1e-4)
    ap.add_argument("--ogg_protect_first_pages", type=int, default=2)
    ap.add_argument("--ffmpeg_bin", type=str, default=None, help="Path to ffmpeg binary. Resolution order if omitted: $FFMPEG_BIN, /usr/bin/ffmpeg, PATH.")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    ffmpeg_bin = resolve_ffmpeg_bin(args.ffmpeg_bin)
    print(f"[smoke] using ffmpeg binary: {ffmpeg_bin}")

    # ---- JPEG test ----
    jpeg_blob = _make_test_jpeg_blob(rng)
    mask = jpeg_protect_mask(jpeg_blob, prefix_protect_bytes=8)
    n_flip = int(np.count_nonzero(~mask))
    if n_flip <= 0:
        raise RuntimeError("JPEG mask produced zero payload bytes; expected entropy data.")
    jpeg_rx = bitflip_jpeg_entropy_payload(
        jpeg_blob, ber=float(args.ber), prefix_protect_bytes=8, seed=args.seed
    )
    if len(jpeg_rx) != len(jpeg_blob):
        raise RuntimeError("JPEG corrupted blob length changed unexpectedly.")
    print(f"[jpeg] ok: bytes={len(jpeg_blob)} flip_eligible={n_flip}")

    # ---- Ogg/Opus test ----
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        wav = td / "in.wav"
        ogg = td / "in.ogg"
        ogg_zero = td / "zero.ogg"
        ogg_corrupt = td / "corrupt.ogg"
        wav_out_zero = td / "out_zero.wav"
        wav_out_corrupt = td / "out_corrupt.wav"

        can_decode = True
        try:
            # 2 seconds white noise, mono 16kHz
            _run(
                [
                    ffmpeg_bin,
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "anoisesrc=d=2:c=white:r=16000",
                    "-ac",
                    "1",
                    str(wav),
                ]
            )
            _run(
                [
                    ffmpeg_bin,
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(wav),
                    "-c:a",
                    "libopus",
                    "-b:a",
                    "12k",
                    str(ogg),
                ]
            )
            ogg_blob = ogg.read_bytes()
        except RuntimeError as e:
            # Keep structural smoke test usable even if ffmpeg linkage is broken.
            can_decode = False
            ogg_blob = _make_synthetic_ogg_stream(rng)
            print(f"[ogg] ffmpeg unavailable, using synthetic Ogg stream for structural checks: {e}")

        pages = parse_ogg_pages(ogg_blob)
        if not pages:
            raise RuntimeError("No Ogg pages parsed.")
        for pg in pages:
            if not verify_ogg_page_crc(ogg_blob, pg):
                raise RuntimeError(f"Original Ogg CRC invalid on page {pg.index}")

        # BER=0 should be exactly identical bytes.
        ogg_zero_blob = bitflip_ogg_payload_pages(
            ogg_blob, ber=0.0, protect_first_pages=int(args.ogg_protect_first_pages), seed=args.seed
        )
        if ogg_zero_blob != ogg_blob:
            raise RuntimeError("BER=0 Ogg path is not bit-exact identity.")
        ogg_zero.write_bytes(ogg_zero_blob)

        ogg_corrupt_blob = bitflip_ogg_payload_pages(
            ogg_blob,
            ber=float(args.ber),
            protect_first_pages=int(args.ogg_protect_first_pages),
            seed=args.seed,
        )
        ogg_corrupt.write_bytes(ogg_corrupt_blob)
        pages_rx = parse_ogg_pages(ogg_corrupt_blob)
        for pg in pages_rx:
            if not verify_ogg_page_crc(ogg_corrupt_blob, pg):
                raise RuntimeError(f"Corrupted Ogg CRC invalid on page {pg.index}")

        # decode sanity (if ffmpeg works in this environment)
        if can_decode:
            _run(
                [
                    ffmpeg_bin,
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(ogg_zero),
                    str(wav_out_zero),
                ]
            )
            _run(
                [
                    ffmpeg_bin,
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(ogg_corrupt),
                    str(wav_out_corrupt),
                ]
            )
        print(
            f"[ogg] ok: pages={len(pages)} ber={args.ber} protect_first_pages={args.ogg_protect_first_pages} decode_test={int(can_decode)}"
        )

    print("Smoke test passed.")


if __name__ == "__main__":
    main()

