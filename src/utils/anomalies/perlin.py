"""
Perlin noise generation for anomaly map simulation.

Ported from DSR_anomaly_detection:
https://github.com/VitjanZ/DSR_anomaly_detection/blob/main/perlin.py

Design matches the original: lerp_np, same delta/d/grid/tile_grads/dot formulas.
When shape is not divisible by res, noise is computed at effective shape then
resized to the requested shape (for arbitrary spectrogram sizes).
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import torch
from scipy.ndimage import zoom


def lerp_np(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Linear interpolation: (y - x) * w + x (original DSR)."""
    return (y - x) * w + x


def _quintic_fade(t: np.ndarray) -> np.ndarray:
    """Quintic smoothstep: 6t^5 - 15t^4 + 10t^3 (original DSR)."""
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def generate_perlin_noise_2d(shape: tuple[int, int], res: tuple[int, int]) -> np.ndarray:
    """
    Generate 2D Perlin noise (original DSR generate_perlin_noise_2d).
    Uses same formulas as the paper repo: grid, g00/g10/g01/g11, ramps, interpolation.
    """
    def f(t: np.ndarray) -> np.ndarray:
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0 : res[0] : delta[0], 0 : res[1] : delta[1]].transpose(1, 2, 0) % 1

    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))

    g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)

    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)

    t = f(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)


def generate_fractal_noise_2d(
    shape: tuple[int, int],
    res: tuple[int, int],
    octaves: int = 1,
    persistence: float = 0.5,
) -> np.ndarray:
    """Fractal Perlin noise (original DSR generate_fractal_noise_2d)."""
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(
            shape, (frequency * res[0], frequency * res[1])
        )
        frequency *= 2
        amplitude *= persistence
    return noise


def rand_perlin_2d_np(
    shape: tuple[int, int],
    res: tuple[int, int],
    fade: Callable[..., np.ndarray] = _quintic_fade,
) -> np.ndarray:
    """
    Generate 2D Perlin noise (NumPy). Matches original DSR rand_perlin_2d_np:
    delta = res/shape, d = shape//res, tile_grads with repeat, dot with grid/grad slices,
    lerp_np for interpolation.

    When shape is not divisible by res, the original assumes divisible sizes; we compute
    at effective shape (res[0]*d[0], res[1]*d[1]) then zoom to shape for arbitrary
    spectrogram dimensions.

    Returns:
        Noise array of shape (H, W), values roughly in [-1, 1]
    """
    d = (max(1, shape[0] // res[0]), max(1, shape[1] // res[1]))
    effective_shape = (res[0] * d[0], res[1] * d[1])

    delta = (res[0] / effective_shape[0], res[1] / effective_shape[1])
    grid = (
        np.mgrid[
            0 : res[0] : delta[0],
            0 : res[1] : delta[1],
        ].transpose(1, 2, 0)
        % 1
    )
    grid = grid[: effective_shape[0], : effective_shape[1]]

    angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)

    tile_grads = lambda slice1, slice2: np.repeat(
        np.repeat(
            gradients[slice1[0] : slice1[1], slice2[0] : slice2[1]],
            d[0],
            axis=0,
        ),
        d[1],
        axis=1,
    )
    dot = lambda grad, shift: (
        np.stack(
            (
                grid[:, :, 0] + shift[0],
                grid[:, :, 1] + shift[1],
            ),
            axis=-1,
        )
        * grad
    ).sum(axis=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid)
    noise = math.sqrt(2) * lerp_np(
        lerp_np(n00, n10, t[..., 0]),
        lerp_np(n01, n11, t[..., 0]),
        t[..., 1],
    )

    if effective_shape != shape:
        zoom_factors = (shape[0] / effective_shape[0], shape[1] / effective_shape[1])
        noise = np.asarray(zoom(noise, zoom_factors, order=1, mode="nearest"))
    return noise


def _default_fade_torch(t: torch.Tensor) -> torch.Tensor:
    """Quintic smoothstep for torch (original DSR)."""
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def rand_perlin_2d(
    shape: tuple[int, int],
    res: tuple[int, int],
    fade: Callable[..., torch.Tensor] | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    Generate 2D Perlin noise (PyTorch). Matches original DSR rand_perlin_2d:
    same delta, d, grid, tile_grads, dot, torch.lerp.
    """
    fade_fn = fade if fade is not None else _default_fade_torch

    d = (max(1, shape[0] // res[0]), max(1, shape[1] // res[1]))
    effective_shape = (res[0] * d[0], res[1] * d[1])

    delta = (res[0] / effective_shape[0], res[1] / effective_shape[1])
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
    grid = grid[: effective_shape[0], : effective_shape[1]]

    angles = 2 * math.pi * torch.rand(res[0] + 1, res[1] + 1, device=device)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    tile_grads = lambda slice1, slice2: gradients[
        slice1[0] : slice1[1], slice2[0] : slice2[1]
    ].repeat_interleave(d[0], 0).repeat_interleave(d[1], 1)
    dot = lambda grad, shift: (
        torch.stack(
            (
                grid[:, :, 0] + shift[0],
                grid[:, :, 1] + shift[1],
            ),
            dim=-1,
        )
        * grad
    ).sum(dim=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade_fn(grid)
    noise = math.sqrt(2) * torch.lerp(
        torch.lerp(n00, n10, t[..., 0]),
        torch.lerp(n01, n11, t[..., 0]),
        t[..., 1],
    )

    if effective_shape != shape:
        noise = noise.unsqueeze(0).unsqueeze(0)
        noise = torch.nn.functional.interpolate(
            noise, size=shape, mode="bilinear", align_corners=False
        ).squeeze(0).squeeze(0)
    return noise


def rand_perlin_2d_octaves(
    shape: tuple[int, int],
    res: tuple[int, int],
    octaves: int = 1,
    persistence: float = 0.5,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Fractal Perlin noise in PyTorch (original DSR rand_perlin_2d_octaves)."""
    noise = torch.zeros(shape, device=device)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise = noise + amplitude * rand_perlin_2d(
            shape, (frequency * res[0], frequency * res[1]), device=device
        )
        frequency *= 2
        amplitude *= persistence
    return noise
