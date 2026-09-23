"""Pinhole camera model and depth back-projection.

Everything downstream reasons in 3D, so this module is the single place where
image coordinates become metric coordinates. Kept dependency-free (numpy only)
so it can be unit-tested against exact ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Camera", "fit_ground_plane", "plane_heights"]


@dataclass(frozen=True)
class Camera:
    """Simple pinhole intrinsics.

    Args:
        width, height: image size in pixels.
        fx, fy: focal lengths in pixels.
        cx, cy: principal point in pixels.
        height_m: mounting height of the camera above the ground, in metres.
            Used to resolve the unknown scale of monocular depth.
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    height_m: float = 0.8

    @classmethod
    def from_fov(cls, width: int, height: int, hfov_deg: float = 70.0,
                 height_m: float = 0.8) -> "Camera":
        """Build intrinsics from a horizontal field of view."""
        f = (width / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
        return cls(width, height, f, f, width / 2.0, height / 2.0, height_m)

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    def unproject(self, depth: np.ndarray) -> np.ndarray:
        """Back-project a metric depth image to a camera-frame point cloud.

        Args:
            depth: (H, W) array of metric depth along the optical axis (Z).

        Returns:
            (H, W, 3) array of XYZ points. X right, Y down, Z forward.
        """
        h, w = depth.shape
        us, vs = np.meshgrid(np.arange(w, dtype=np.float64),
                             np.arange(h, dtype=np.float64))
        z = depth.astype(np.float64)
        x = (us - self.cx) * z / self.fx
        y = (vs - self.cy) * z / self.fy
        return np.stack([x, y, z], axis=-1)

    def project(self, points: np.ndarray) -> np.ndarray:
        """Project camera-frame XYZ points back to pixel coordinates.

        Points with non-positive Z are returned as NaN.
        """
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        z = pts[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.fx * pts[:, 0] / z + self.cx
            v = self.fy * pts[:, 1] / z + self.cy
        uv = np.stack([u, v], axis=-1)
        uv[z <= 0] = np.nan
        return uv.reshape(np.asarray(points).shape[:-1] + (2,))


def fit_ground_plane(points: np.ndarray, mask: np.ndarray | None = None,
                     iters: int = 200, thresh: float = 0.08,
                     seed: int = 0) -> tuple[np.ndarray, float, np.ndarray]:
    """RANSAC plane fit, used to find the drivable ground surface.

    Args:
        points: (H, W, 3) or (N, 3) camera-frame points.
        mask: optional boolean array selecting candidate ground pixels.
        iters: RANSAC iterations.
        thresh: inlier distance in metres.

    Returns:
        (normal, offset, inlier_mask) where the plane is ``n . x + d = 0``
        and ``normal`` is unit length and points away from the ground.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    valid = np.isfinite(pts).all(axis=1)
    if mask is not None:
        valid &= np.asarray(mask).reshape(-1).astype(bool)
    idx = np.flatnonzero(valid)
    if idx.size < 3:
        raise ValueError("need at least 3 valid points to fit a plane")

    rng = np.random.default_rng(seed)
    best_n = np.array([0.0, -1.0, 0.0])
    best_d = 0.0
    best_inl = np.zeros(pts.shape[0], dtype=bool)

    for _ in range(iters):
        s = pts[rng.choice(idx, 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        d = -float(n @ s[0])
        dist = np.abs(pts @ n + d)
        inl = valid & (dist < thresh)
        if inl.sum() > best_inl.sum():
            best_n, best_d, best_inl = n, d, inl

    if best_inl.sum() >= 3:  # least-squares refit on the inlier set
        P = pts[best_inl]
        c = P.mean(axis=0)
        _, _, vt = np.linalg.svd(P - c, full_matrices=False)
        best_n = vt[-1] / np.linalg.norm(vt[-1])
        best_d = -float(best_n @ c)

    if best_n[1] > 0:  # keep the normal pointing "up" (-Y in camera frame)
        best_n, best_d = -best_n, -best_d
    return best_n, best_d, best_inl.reshape(np.asarray(points).shape[:-1])


def plane_heights(points: np.ndarray, normal: np.ndarray, offset: float) -> np.ndarray:
    """Signed height of each point above the given plane, in metres."""
    pts = np.asarray(points, dtype=np.float64)
    return pts @ np.asarray(normal, dtype=np.float64) + float(offset)
