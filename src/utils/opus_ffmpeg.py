"""
Ogg Opus encode/decode via ffmpeg + libopus (CBR by default: -vbr off).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import torch
import torchaudio


def resolve_ffmpeg_bin(ffmpeg_bin_arg: str | None = None) -> str:
    """Resolve ffmpeg binary: explicit arg, FFMPEG_BIN env, /usr/bin/ffmpeg, then PATH."""
    if ffmpeg_bin_arg:
        return str(ffmpeg_bin_arg)
    import os

    env_bin = os.environ.get("FFMPEG_BIN")
    if env_bin:
        return env_bin
    if Path("/usr/bin/ffmpeg").exists():
        return "/usr/bin/ffmpeg"
    path_bin = shutil.which("ffmpeg")
    if path_bin:
        return path_bin
    raise FileNotFoundError("ffmpeg not found. Set --ffmpeg_bin or FFMPEG_BIN.")


def _opus_encode_cmd(
    ffmpeg: str,
    in_path: str | Path,
    out_path: str | Path,
    bitrate_kbps: int,
    *,
    cbr: bool = True,
) -> list[str]:
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(in_path),
        "-c:a",
        "libopus",
        "-b:a",
        f"{int(bitrate_kbps)}k",
    ]
    if cbr:
        cmd.extend(["-vbr", "off", "-application", "audio"])
    cmd.append(str(out_path))
    return cmd


def ffmpeg_opus_encode_file(
    in_wav: str | Path,
    out_ogg: str | Path,
    bitrate_kbps: int,
    *,
    ffmpeg: str | None = None,
    cbr: bool = True,
) -> None:
    """
    Encode PCM wav to Ogg Opus. Uses CBR (-vbr off) when supported by the ffmpeg build.
    """
    ff = ffmpeg or resolve_ffmpeg_bin()
    enc_cmd = _opus_encode_cmd(ff, in_wav, out_ogg, bitrate_kbps, cbr=cbr)
    try:
        subprocess.run(enc_cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").strip()
        if cbr and ("Unrecognized option 'vbr'" in stderr or "Option not found" in stderr):
            enc_cmd_fallback = [x for x in enc_cmd if x not in ("-vbr", "off")]
            subprocess.run(enc_cmd_fallback, check=True, capture_output=True, text=True)
        else:
            raise


def opus_encode_wav_path_to_bytes(
    wav_path: str | Path,
    bitrate_kbps: int,
    *,
    ffmpeg: str | None = None,
) -> bytes:
    """Encode a wav file to Ogg Opus bytes (CBR when ffmpeg supports -vbr off)."""
    ff = ffmpeg or resolve_ffmpeg_bin()
    with tempfile.TemporaryDirectory() as td:
        out_ogg = Path(td) / "clip.ogg"
        ffmpeg_opus_encode_file(wav_path, out_ogg, bitrate_kbps, ffmpeg=ff)
        return out_ogg.read_bytes()


def opus_encode_tensor_to_bytes(
    wav: torch.Tensor,
    sr: int,
    bitrate_kbps: int,
    *,
    ffmpeg: str | None = None,
) -> bytes:
    """Encode a waveform tensor (C, N) to Ogg Opus bytes."""
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    ff = ffmpeg or resolve_ffmpeg_bin()
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        in_wav = td_path / "in.wav"
        out_ogg = td_path / "out.ogg"
        torchaudio.save(str(in_wav), wav.cpu(), sample_rate=sr)
        ffmpeg_opus_encode_file(in_wav, out_ogg, bitrate_kbps, ffmpeg=ff)
        return out_ogg.read_bytes()


def opus_decode_bytes_to_wav(
    opus_bytes: bytes,
    *,
    ffmpeg: str | None = None,
) -> tuple[torch.Tensor, int]:
    """Decode Ogg Opus bytes to PCM wav (s16le)."""
    ff = ffmpeg or resolve_ffmpeg_bin()
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        in_ogg = td_path / "in.ogg"
        out_wav = td_path / "out.wav"
        in_ogg.write_bytes(opus_bytes)
        dec_cmd = [
            ff,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(in_ogg),
            "-acodec",
            "pcm_s16le",
            str(out_wav),
        ]
        subprocess.run(dec_cmd, check=True, capture_output=True, text=True)
        wav, out_sr = torchaudio.load(str(out_wav))
        return wav, int(out_sr)
