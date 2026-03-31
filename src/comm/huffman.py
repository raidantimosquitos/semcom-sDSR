"""
Canonical Huffman coder for non-negative integer symbols.

This is used as a lightweight, dependency-free entropy coder for index maps.
It is not as efficient as arithmetic coding, but is simple and gives a strong
baseline using the measured p(k).
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Dict, Iterable, List, Tuple

import numpy as np


@dataclass(frozen=True)
class HuffmanCode:
    # symbol -> (code_bits_as_int, code_len)
    enc: Dict[int, Tuple[int, int]]
    # canonical decode: map (len -> (first_code, symbols_sorted))
    dec_by_len: Dict[int, Tuple[int, List[int]]]
    max_len: int


def _code_lengths_from_counts(counts: np.ndarray) -> Dict[int, int]:
    # Build Huffman tree, return symbol->code_length.
    # Ensure at least 2 symbols with nonzero weight to avoid degenerate tree.
    items = [(int(c), int(i)) for i, c in enumerate(counts) if c > 0]
    if not items:
        # fallback: uniform length 1 for symbol 0 only
        return {0: 1}
    if len(items) == 1:
        # add a dummy symbol with tiny weight
        (c0, s0) = items[0]
        s1 = 0 if s0 != 0 else 1
        items.append((1, s1))

    # heap nodes: (weight, tie_id, node)
    # node is (sym) for leaf or (left,right) for internal.
    heap = []
    tie = 0
    for w, sym in items:
        heapq.heappush(heap, (w, tie, sym))
        tie += 1
    while len(heap) > 1:
        w1, _, n1 = heapq.heappop(heap)
        w2, _, n2 = heapq.heappop(heap)
        heapq.heappush(heap, (w1 + w2, tie, (n1, n2)))
        tie += 1
    _w, _t, root = heap[0]

    lengths: Dict[int, int] = {}

    def dfs(node, depth: int) -> None:
        if isinstance(node, int):
            lengths[node] = max(1, depth)
            return
        left, right = node
        dfs(left, depth + 1)
        dfs(right, depth + 1)

    dfs(root, 0)
    return lengths


def build_huffman_from_counts(counts: np.ndarray) -> HuffmanCode:
    """
    Args:
        counts: (K,) nonnegative ints
    Returns:
        canonical Huffman code for symbols 0..K-1 (only nonzero-count symbols get codes;
        unseen symbols fall back to a uniform-length escape not implemented here).
    """
    counts = np.asarray(counts, dtype=np.int64).reshape(-1)
    lengths = _code_lengths_from_counts(counts)

    # Canonical assignment: sort by (length, symbol)
    syms = sorted(lengths.items(), key=lambda kv: (kv[1], kv[0]))
    enc: Dict[int, Tuple[int, int]] = {}
    dec_by_len: Dict[int, Tuple[int, List[int]]] = {}

    code = 0
    prev_len = syms[0][1]
    for sym, L in syms:
        if L > prev_len:
            code <<= (L - prev_len)
            prev_len = L
        enc[int(sym)] = (int(code), int(L))
        code += 1

    # Build decode tables per length
    by_len: Dict[int, List[Tuple[int, int]]] = {}
    for sym, (c, L) in enc.items():
        by_len.setdefault(L, []).append((c, sym))
    max_len = max(by_len.keys())
    for L, pairs in by_len.items():
        pairs_sorted = sorted(pairs, key=lambda x: x[0])
        first_code = pairs_sorted[0][0]
        symbols_sorted = [s for _c, s in pairs_sorted]
        dec_by_len[int(L)] = (int(first_code), symbols_sorted)

    return HuffmanCode(enc=enc, dec_by_len=dec_by_len, max_len=int(max_len))


def encode_symbols(code: HuffmanCode, symbols: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Encode symbols to a packed LSB-first bitstream.

    Returns:
        payload_bytes: uint8 array of bytes
        n_bits: number of valid bits in the stream
    """
    symbols = np.asarray(symbols, dtype=np.int64).reshape(-1)
    # bit buffer as list of 0/1 (LSB-first in bytes, same convention as src/utils/bitstream.py)
    bits: List[int] = []
    enc = code.enc
    for s in symbols:
        c, L = enc[int(s)]
        for b in range(L):
            bits.append((c >> b) & 1)
    n_bits = len(bits)
    n_bytes = (n_bits + 7) // 8
    out = np.zeros(n_bytes, dtype=np.uint8)
    for i, bit in enumerate(bits):
        if bit:
            out[i // 8] |= np.uint8(1 << (i % 8))
    return out, int(n_bits)


def decode_symbols(code: HuffmanCode, payload_bytes: np.ndarray, n_bits: int, n_symbols: int) -> np.ndarray:
    """
    Decode a packed LSB-first bitstream into exactly n_symbols symbols.
    """
    payload_bytes = np.asarray(payload_bytes, dtype=np.uint8).reshape(-1)
    # expand bits (LSB-first per byte)
    bits = np.empty(n_bits, dtype=np.uint8)
    for i in range(n_bits):
        bits[i] = (payload_bytes[i // 8] >> (i % 8)) & 1

    out = np.empty(n_symbols, dtype=np.int64)
    pos = 0
    cur = 0
    cur_len = 0
    dec_by_len = code.dec_by_len

    j = 0
    while j < n_symbols:
        if pos >= n_bits:
            raise ValueError("Ran out of bits while decoding.")
        cur |= int(bits[pos]) << cur_len
        cur_len += 1
        pos += 1
        if cur_len in dec_by_len:
            first_code, symbols_sorted = dec_by_len[cur_len]
            idx = cur - first_code
            if 0 <= idx < len(symbols_sorted):
                out[j] = symbols_sorted[idx]
                j += 1
                cur = 0
                cur_len = 0
        if cur_len > code.max_len:
            raise ValueError("Invalid Huffman codeword (exceeded max length).")
    return out

