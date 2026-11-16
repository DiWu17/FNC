"""Label-free summaries of a query token's attention to image patches.

The inputs are *raw* softmax probabilities over the complete sequence, sliced
to image positions. Normalizing within the image first would destroy the image
attention mass, which is explicitly retained here. No target box is accepted by
this API so ground-truth localization cannot accidentally enter a probe.
"""

from __future__ import annotations

import numpy as np


PER_HEAD_FEATURES = (
    "image_mass", "entropy", "peak", "peak_margin", "top5_mass",
    "centroid_x", "centroid_y", "std_x", "std_y", "concentration",
)


def _distribution_features(weights: np.ndarray, grid_shape: tuple[int, int]) -> np.ndarray:
    """Return features [heads, features], accepting an empty image slice."""
    heads, patches = weights.shape
    mass = weights.sum(axis=1)
    result = np.zeros((heads, len(PER_HEAD_FEATURES)), dtype=np.float64)
    result[:, 0] = mass
    if patches == 0:
        return result
    height, width = grid_shape
    normalized = np.divide(weights, mass[:, None], out=np.zeros_like(weights), where=mass[:, None] > 0)
    positive = normalized > 0
    logs = np.zeros_like(normalized)
    np.log(normalized, out=logs, where=positive)
    if patches > 1:
        result[:, 1] = -(normalized * logs).sum(axis=1) / np.log(patches)
    ordered = np.sort(normalized, axis=1)
    result[:, 2] = ordered[:, -1]
    result[:, 3] = ordered[:, -1] - (ordered[:, -2] if patches > 1 else 0)
    result[:, 4] = ordered[:, -min(5, patches):].sum(axis=1)
    x = np.tile((np.arange(width) + 0.5) / width, height)
    y = np.repeat((np.arange(height) + 0.5) / height, width)
    cx, cy = normalized @ x, normalized @ y
    result[:, 5], result[:, 6] = cx, cy
    result[:, 7] = np.sqrt(np.maximum(0, (normalized * (x[None, :] - cx[:, None]) ** 2).sum(axis=1)))
    result[:, 8] = np.sqrt(np.maximum(0, (normalized * (y[None, :] - cy[:, None]) ** 2).sum(axis=1)))
    result[:, 9] = (normalized ** 2).sum(axis=1)
    return result


def attention_feature_names(num_heads: int, layer: int | str) -> list[str]:
    """Stable names in exactly the order returned by ``summarize_image_attention``."""
    prefix = f"layer_{layer}"
    names = [f"{prefix}.head_{head}.{name}" for head in range(num_heads) for name in PER_HEAD_FEATURES]
    names.extend(f"{prefix}.heads_{stat}.{name}" for stat in ("mean", "std") for name in PER_HEAD_FEATURES)
    names.extend(f"{prefix}.mean_map.{name}" for name in PER_HEAD_FEATURES)
    names.extend((f"{prefix}.agreement_mean", f"{prefix}.agreement_std"))
    return names


def summarize_image_attention(
    weights: np.ndarray,
    grid_shape: tuple[int, int],
    *,
    layer: int | str = 0,
) -> tuple[np.ndarray, list[str]]:
    """Summarize raw image attention [heads, patches] without label information.

    Coordinates use patch centers normalized to [0, 1]. The grid is the
    *post-merger* raster grid, not the original vision encoder's patch order.
    A no-image ablation supplies ``(heads, 0)`` and ``grid_shape=(0, 0)``.
    The returned vector includes each head, across-head mean/std, the mean
    attention map, and cosine agreement between heads and their mean map.
    """
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 2 or weights.shape[0] < 1:
        raise ValueError("weights must have shape [positive heads, patches]")
    if len(grid_shape) != 2 or any(int(value) != value or value < 0 for value in grid_shape):
        raise ValueError("grid_shape must contain two nonnegative integers")
    grid_shape = (int(grid_shape[0]), int(grid_shape[1]))
    if int(np.prod(grid_shape)) != weights.shape[1]:
        raise ValueError("post-merger grid size does not match image attention length")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("attention probabilities must be finite and nonnegative")
    if (weights.sum(axis=1) > 1.001).any():
        raise ValueError("image attention mass exceeds one; provide raw softmax probabilities")
    per_head = _distribution_features(weights, grid_shape)
    mean_map = weights.mean(axis=0, keepdims=True)
    mean_features = _distribution_features(mean_map, grid_shape).ravel()
    norms = np.linalg.norm(weights, axis=1) * np.linalg.norm(mean_map)
    agreement = np.divide(weights @ mean_map.ravel(), norms, out=np.zeros(weights.shape[0]), where=norms > 0)
    vector = np.concatenate((per_head.ravel(), per_head.mean(axis=0), per_head.std(axis=0),
                             mean_features, [agreement.mean(), agreement.std()])).astype(np.float32)
    return vector, attention_feature_names(weights.shape[0], layer)
