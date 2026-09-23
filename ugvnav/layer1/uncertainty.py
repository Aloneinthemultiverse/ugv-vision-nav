"""Node C - uncertainty estimation.

Runs the depth estimator several times under mild photometric and geometric
perturbations and measures how much the answers disagree. Where they scatter,
the reading is untrustworthy - typically glare, deep shadow, or texture-poor
ground - and the fusion layer downgrades that region to CAUTION rather than
guessing that it is safe.

This is test-time augmentation consistency, which needs no dropout layers and
works with any depth backend.
"""
from __future__ import annotations

import numpy as np

__all__ = ["UncertaintyEstimator", "normalise"]


def normalise(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Scale an array to [0, 1] using its own min/max."""
    x = np.asarray(x, dtype=np.float32)
    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    if hi - lo < eps:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


class UncertaintyEstimator:
    """Augmentation-consistency uncertainty over a depth estimator.

    Args:
        estimator: object exposing ``infer(rgb) -> (H, W)``.
        samples: number of perturbed passes.
        gains: multiplicative brightness factors cycled over the passes.
        flip: also evaluate a horizontally mirrored pass.
    """

    def __init__(self, estimator, samples: int = 4,
                 gains: tuple[float, ...] = (0.85, 1.0, 1.15, 1.3),
                 flip: bool = True) -> None:
        self.estimator = estimator
        self.samples = int(samples)
        self.gains = gains
        self.flip = bool(flip)

    def process(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Args:
            rgb: (H, W, 3) uint8 image.

        Returns:
            (mean_depth, uncertainty) where ``uncertainty`` is in [0, 1];
            higher means the passes disagreed more.
        """
        rgb = np.asarray(rgb, dtype=np.uint8)
        preds: list[np.ndarray] = []

        for i in range(self.samples):
            gain = self.gains[i % len(self.gains)]
            aug = np.clip(rgb.astype(np.float32) * gain, 0, 255).astype(np.uint8)
            mirrored = self.flip and (i % 2 == 1)
            if mirrored:
                aug = aug[:, ::-1]
            d = np.asarray(self.estimator.infer(aug), dtype=np.float32)
            if mirrored:
                d = d[:, ::-1]
            preds.append(normalise(d))

        stack = np.stack(preds, axis=0)
        mean = stack.mean(axis=0)
        # Spread across passes, rescaled so the map is comparable frame to frame.
        return mean, normalise(stack.std(axis=0))
