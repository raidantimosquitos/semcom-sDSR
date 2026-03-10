# Bitstream format for GNURadio SDR link

This document describes the binary format of the codebook-index bitstream produced by `scripts/transmit_indices.py` and consumed by `scripts/receive_and_evaluate.py` after FEC decoding on the receiver side.

## File layout

- **Header**: 4 bytes, little-endian unsigned 32-bit integer = number of clips (frames) `N`.
- **Payload**: `N` consecutive frames. Each frame has the same fixed size in bytes (see below).

## Frame layout (one clip)

Each frame encodes the VQ-VAE codebook indices for one spectrogram (one test clip).

- **Coarse indices**: grid `H_coarse × W_coarse` (e.g. 8×20 for 128×320 spectrograms). Each index is **10 bits** (codebook size 1024). Indices are in **row-major** order.
- **Fine indices**: grid `H_fine × W_fine` (e.g. 32×80). Each index is **12 bits** (codebook size 4096). Row-major order.

**Bit packing**: Within each index, bits are **LSB first** (bit 0 = LSB). Indices are packed in order: all coarse indices first, then all fine indices. The resulting bit stream is then packed into bytes (byte 0 = bits 0–7 of the stream, byte 1 = bits 8–15, etc.).

## Frame size (default 128×320, 1024/4096 codebooks)

- `H_coarse = 8`, `W_coarse = 20` → 160 coarse indices × 10 bits = 1,600 bits.
- `H_fine = 32`, `W_fine = 80` → 2,560 fine indices × 12 bits = 30,720 bits.
- Total per frame: **32,320 bits = 4,040 bytes**.

## Usage with GNURadio

1. **Transmitter**: Run `transmit_indices.py` to produce a single binary file (header + N frames). Feed this file (or the raw payload after skipping the 4-byte header if your flow expects only frames) into your FEC encoder and then modulator. The FEC encoder typically expects a stream of bits; you can unpack the bytes to bits in the same order (LSB of first byte = first bit).
2. **Receiver**: After demodulation and FEC decoding, write the decoded bitstream to a file in the **same format** (4-byte `N` + N × 4,040 bytes). Run `receive_and_evaluate.py --input_bitstream <file>` to decode indices and evaluate AUC/pAUC.

## Changing spectrogram or codebook size

If you change `n_mels`, `T`, or codebook sizes, the frame size changes. Use `src.utils.bitstream.frame_size_bytes(H_coarse, W_coarse, H_fine, W_fine, bits_coarse, bits_fine)` to compute the correct frame length. Transmitter and receiver must use the same dimensions and bit widths.
