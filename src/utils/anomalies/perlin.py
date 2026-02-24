"""
Perlin noise generation for anomaly map simulation.

Ported from DSR_anomaly_detection:
https://github.com/VitjanZ/DSR_anomaly_detection/blob/main/perlin.py

Used by PerlinNoiseStrategy in anomaly_map.py to generate blob-like
anomaly regions during training.
"""

from __future__ import annotations

import math

import numpy as np
import torch


def _quintic_fade(t: np.ndarray) -> np.ndarray:
    """Quintic smoothstep: 6t^5 - 15t^4 + 10t^3."""
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def rand_perlin_2d_np(
    shape: tuple[int, int],
    res: tuple[int, int],
    fade: callable = _quintic_fade,
) -> np.ndarray:
    """
    Generate 2D Perlin noise (NumPy, CPU).

    Args:
        shape: (H, W) output shape
        res: (res_y, res_x) grid resolution for noise tiles
        fade: smoothing function for interpolation

    Returns:
        Noise array of shape (H, W), values roughly in [-1, 1]
    """
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = (
        np.mgrid[0 : res[0] : delta[0], 0 : res[1] : delta[1]].transpose(1, 2, 0) % 1
    )

    angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)

    def tile_grads(s0: tuple[int, int | None], s1: tuple[int, int | None]):
        sl0 = slice(s0[0], s0[1])
        sl1 = slice(s1[0], s1[1])
        g = gradients[sl0, sl1]
        return np.repeat(np.repeat(g, d[0], axis=0), d[1], axis=1)

    def dot(grad, shift):
        gr = np.stack(
            (
                grid[: shape[0], : shape[1], 0] + shift[0],
                grid[: shape[0], : shape[1], 1] + shift[1],
            ),
            axis=-1,
        )
        return (gr * grad[: shape[0], : shape[1]]).sum(axis=-1)

    n00 = dot(tile_grads((0, -1), (0, -1)), [0, 0])
    n10 = dot(tile_grads((1, None), (0, -1)), [-1, 0])
    n01 = dot(tile_grads((0, -1), (1, None)), [0, -1])
    n11 = dot(tile_grads((1, None), (1, None)), [-1, -1])
    t = fade(grid[: shape[0], : shape[1]])
    n0 = (1 - t[..., 0]) * n00 + t[..., 0] * n10
    n1 = (1 - t[..., 0]) * n01 + t[..., 0] * n11
    return math.sqrt(2) * ((1 - t[..., 1]) * n0 + t[..., 1] * n1)


def rand_perlin_2d(
    shape: tuple[int, int],
    res: tuple[int, int],
    fade: callable | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    Generate 2D Perlin noise (PyTorch, GPU-compatible).

    Args:
        shape: (H, W) output shape
        res: (res_y, res_x) grid resolution
        fade: optional smoothing function; default quintic
        device: torch device

    Returns:
        Noise tensor of shape (H, W)
    """
    if fade is None:
        fade = lambda t: 6 * t**5 - 15 * t**4 + 10 * t**3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = (
        torch.stack(
            torch.meshgrid(
                torch.arange(0, res[0], delta[0], device=device),
                torch.arange(0, res[1], delta[1], device=device),
                indexing="ij",
            ),
            dim=-1,
        )
        % 1
    )
    angles = 2 * math.pi * torch.rand(res[0] + 1, res[1] + 1, device=device)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    def tile_grads(s0: tuple[int, int | None], s1: tuple[int, int | None]):
        sl0 = slice(s0[0], s0[1])
        sl1 = slice(s1[0], s1[1])
        g = gradients[sl0, sl1]
        return g.repeat_interleave(d[0], 0).repeat_interleave(d[1], 1)

    def dot(grad, shift):
        gr = torch.stack(
            (
                grid[: shape[0], : shape[1], 0] + shift[0],
                grid[: shape[0], : shape[1], 1] + shift[1],
            ),
            dim=-1,
        )
        return (gr * grad[: shape[0], : shape[1]]).sum(dim=-1)

    n00 = dot(tile_grads((0, -1), (0, -1)), [0, 0])
    n10 = dot(tile_grads((1, None), (0, -1)), [-1, 0])
    n01 = dot(tile_grads((0, -1), (1, None)), [0, -1])
    n11 = dot(tile_grads((1, None), (1, None)), [-1, -1])
    t = fade(grid[: shape[0], : shape[1]])
    n0 = torch.lerp(n00, n10, t[..., 0])
    n1 = torch.lerp(n01, n11, t[..., 0])
    return math.sqrt(2) * torch.lerp(n0, n1, t[..., 1])
