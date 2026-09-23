"""Node A - elevation / 2.5D ground model.

Turns a metric depth image into a height-above-ground field plus a coarse
top-down elevation map. This is what Layer 2 and the geometry override reason
about: a semantic label can lie about texture, but height does not.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..camera import Camera, fit_ground_plane, plane_heights

__all__ = ["ElevationResult", "ElevationNetwork"]


@dataclass
class ElevationResult:
    points: np.ndarray        # (H, W, 3) camera-frame XYZ, metres
    height: np.ndarray        # (H, W) signed height above the ground plane
    ground_mask: np.ndarray   # (H, W) bool, plane inliers
    normal: np.ndarray        # (3,) unit plane normal
    offset: float             # plane offset d in n.x + d = 0
    scale: float              # factor applied to reach metric scale


class ElevationNetwork:
    """Depth image -> ground plane -> height field.

    The ground plane is fitted with RANSAC over the lower part of the image,
    where drivable surface is most likely. Because monocular depth has unknown
    scale, the fitted camera-to-ground distance is rescaled to the known
    mounting height, which recovers absolute scale for the whole scene.
    """

    def __init__(self, camera: Camera, roi_top: float = 0.45,
                 inlier_thresh: float = 0.08, iters: int = 200) -> None:
        self.camera = camera
        self.roi_top = float(roi_top)
        self.inlier_thresh = float(inlier_thresh)
        self.iters = int(iters)

    def _roi(self, shape: tuple[int, int]) -> np.ndarray:
        h, w = shape
        m = np.zeros((h, w), dtype=bool)
        m[int(h * self.roi_top):, :] = True
        return m

    def process(self, depth: np.ndarray, seed: int = 0) -> ElevationResult:
        """Args:
            depth: (H, W) metric depth in metres.

        Returns:
            ElevationResult with a metrically-scaled height field.
        """
        depth = np.asarray(depth, dtype=np.float64)
        pts = self.camera.unproject(depth)
        n, d, inl = fit_ground_plane(pts, self._roi(depth.shape),
                                     iters=self.iters,
                                     thresh=self.inlier_thresh, seed=seed)

        # |d| is the camera's distance to the fitted plane in the depth map's
        # arbitrary units; rescale so it equals the real mounting height.
        scale = 1.0
        if abs(d) > 1e-6 and self.camera.height_m > 0:
            scale = float(self.camera.height_m / abs(d))
        pts = pts * scale
        d = d * scale

        height = plane_heights(pts, n, d)
        return ElevationResult(points=pts, height=height, ground_mask=inl,
                               normal=n, offset=float(d), scale=scale)

    @staticmethod
    def top_down(points: np.ndarray, height: np.ndarray, cells: int = 32,
                 x_range: tuple[float, float] = (-4.0, 4.0),
                 z_range: tuple[float, float] = (0.5, 12.0)) -> np.ndarray:
        """Rasterise the height field into a top-down elevation grid.

        Returns:
            (cells, cells) array of max height per cell; NaN where unobserved.
        """
        p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        h = np.asarray(height, dtype=np.float64).reshape(-1)
        out = np.full((cells, cells), np.nan)

        ok = np.isfinite(p).all(axis=1) & np.isfinite(h)
        x, z = p[ok, 0], p[ok, 2]
        h = h[ok]
        cx = ((x - x_range[0]) / (x_range[1] - x_range[0]) * cells).astype(int)
        cz = ((z - z_range[0]) / (z_range[1] - z_range[0]) * cells).astype(int)
        keep = (cx >= 0) & (cx < cells) & (cz >= 0) & (cz < cells)
        cx, cz, h = cx[keep], cz[keep], h[keep]

        for xi, zi, hv in zip(cx, cz, h):
            cur = out[cells - 1 - zi, xi]
            if not np.isfinite(cur) or hv > cur:
                out[cells - 1 - zi, xi] = hv
        return out
