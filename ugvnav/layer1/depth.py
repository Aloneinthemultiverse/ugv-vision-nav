"""Node A1 - monocular depth.

Wraps Depth Anything V2 (Apache-2.0) behind a small interface so the rest of
the stack never imports torch. Tests inject ``StubDepth`` instead of the real
network, which keeps the geometric code fully testable on CPU with no download.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

__all__ = ["DepthEstimator", "MonocularDepth", "StubDepth", "to_metric"]

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


class DepthEstimator(Protocol):
    """Anything that turns an RGB image into a relative inverse-depth map."""

    def infer(self, rgb: np.ndarray) -> np.ndarray:  # pragma: no cover - protocol
        ...


class MonocularDepth:
    """Depth Anything V2 wrapper. Returns *relative inverse depth* (near = large)."""

    def __init__(self, model_id: str = MODEL_ID, device: int = -1) -> None:
        from transformers import pipeline  # imported lazily: heavy dependency

        self._pipe = pipeline("depth-estimation", model=model_id, device=device)

    def infer(self, rgb: np.ndarray) -> np.ndarray:
        from PIL import Image

        img = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
        out = self._pipe(img)
        d = np.asarray(out["depth"], dtype=np.float32)
        if d.shape != rgb.shape[:2]:
            import cv2

            d = cv2.resize(d, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
        return d


class StubDepth:
    """Deterministic stand-in used by the test suite."""

    def __init__(self, field: np.ndarray) -> None:
        self._field = np.asarray(field, dtype=np.float32)

    def infer(self, rgb: np.ndarray) -> np.ndarray:
        return self._field.copy()


def to_metric(inverse_depth: np.ndarray, near_m: float = 1.0,
              far_m: float = 30.0, eps: float = 1e-6) -> np.ndarray:
    """Convert relative inverse depth to a metric-*like* depth map.

    Monocular depth is scale-ambiguous. We map the observed inverse-depth range
    onto a plausible working range so that downstream geometry has consistent
    units. Absolute scale is recovered later by fitting the ground plane against
    the known camera mounting height.

    Args:
        inverse_depth: (H, W) relative inverse depth, larger = nearer.
        near_m: metric depth assigned to the nearest observed pixel.
        far_m: metric depth assigned to the furthest observed pixel.

    Returns:
        (H, W) float32 depth in metres, increasing away from the camera.
    """
    d = np.asarray(inverse_depth, dtype=np.float32)
    lo, hi = float(d.min()), float(d.max())
    if hi - lo < eps:
        return np.full_like(d, far_m, dtype=np.float32)
    norm = (d - lo) / (hi - lo)          # 0 = furthest, 1 = nearest
    inv_near, inv_far = 1.0 / near_m, 1.0 / far_m
    inv = inv_far + norm * (inv_near - inv_far)
    return (1.0 / np.maximum(inv, eps)).astype(np.float32)
