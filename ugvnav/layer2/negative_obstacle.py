"""Node E - negative obstacle detection (ditches, drop-offs, potholes).

A hole is invisible to appearance-based perception: in a photograph a ditch and
a shadow look alike. It is not invisible to geometry. Scanning down each image
column, the ground should recede smoothly; where depth jumps or the surface
falls below the fitted ground plane, the ground has dropped away.

Two independent cues are combined:
  * depth discontinuity - the vertical derivative of the depth field;
  * plane deviation     - points sitting measurably *below* the ground plane.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

__all__ = ["NegativeObstacleResult", "NegativeObstacleDetector"]

#: Depth-derivative magnitude below which a pixel is treated as flat ground.
_MIN_GRAD = 1e-5


@dataclass
class NegativeObstacleResult:
    mask: np.ndarray        # (H, W) bool, pixels belonging to a drop-off
    edge: np.ndarray        # (H, W) float32, normalised discontinuity strength
    gradient: np.ndarray    # (H, W) float32, raw vertical depth derivative
    ranges: np.ndarray      # (W,) distance to the nearest drop-off per column,
                            #      +inf where the column is clear


class NegativeObstacleDetector:
    """Detects ground that falls away from under the vehicle.

    Args:
        drop_m: how far below the ground plane counts as a negative obstacle.
        grad_percentile: percentile of the depth-derivative magnitude treated
            as a discontinuity.
        blur: Gaussian sigma applied to depth before differentiating.
        min_area: connected components smaller than this are discarded.
    """

    def __init__(self, drop_m: float = 0.25, grad_percentile: float = 99.0,
                 blur: float = 2.0, min_area: int = 40) -> None:
        self.drop_m = float(drop_m)
        self.grad_percentile = float(grad_percentile)
        self.blur = float(blur)
        self.min_area = int(min_area)

    def process(self, depth: np.ndarray, height: np.ndarray | None = None,
                valid: np.ndarray | None = None) -> NegativeObstacleResult:
        """Args:
            depth: (H, W) metric depth.
            height: optional (H, W) signed height above the ground plane
                (from Node A). Negative values are below the plane.
            valid: optional (H, W) bool mask of analysable ground pixels;
                sky and out-of-range pixels should be excluded.

        Returns:
            NegativeObstacleResult.
        """
        depth = np.asarray(depth, dtype=np.float32)
        h, w = depth.shape
        valid = np.ones((h, w), bool) if valid is None else np.asarray(valid, bool)

        smooth = cv2.GaussianBlur(depth, (0, 0), self.blur) if self.blur > 0 else depth
        grad = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=5)
        grad = grad * valid.astype(np.float32)

        mag = np.abs(grad)
        pool = mag[valid]
        thr = float(np.percentile(pool, self.grad_percentile)) if pool.size else np.inf
        # A flat depth map has zero gradient everywhere. Without this guard the
        # threshold collapses to 0 and `mag >= 0` marks the whole image as a
        # drop-off edge.
        if np.isfinite(thr) and thr > _MIN_GRAD:
            edge = (mag >= thr) & (mag > _MIN_GRAD)
        else:
            edge = np.zeros_like(mag, dtype=bool)

        below = np.zeros((h, w), bool)
        if height is not None:
            below = (np.asarray(height, dtype=np.float32) < -self.drop_m) & valid

        mask = (edge | below) & valid
        mask = self._despeckle(mask)

        norm = mag / (thr if np.isfinite(thr) and thr > _MIN_GRAD else 1.0)
        return NegativeObstacleResult(
            mask=mask,
            edge=np.clip(norm, 0.0, 1.0).astype(np.float32),
            gradient=grad.astype(np.float32),
            ranges=self._column_ranges(mask, depth),
        )

    def _despeckle(self, mask: np.ndarray) -> np.ndarray:
        if self.min_area <= 1 or not mask.any():
            return mask
        n, lab, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), 8)
        keep = np.zeros_like(mask)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= self.min_area:
                keep |= lab == i
        return keep

    @staticmethod
    def _column_ranges(mask: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Nearest drop-off distance per image column - a virtual laser scan.

        This is the form Nav2 consumes: each column becomes one ray, so the
        costmap can treat an invisible hole exactly like a solid obstacle.
        """
        h, w = mask.shape
        out = np.full(w, np.inf, dtype=np.float32)
        cols = np.flatnonzero(mask.any(axis=0))
        for c in cols:
            rows = np.flatnonzero(mask[:, c])
            out[c] = float(depth[rows, c].min())
        return out
