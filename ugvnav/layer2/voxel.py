"""Node G - sparse voxel mapping and overhead clearance.

A 2.5D height map only describes the ground, so a branch hanging at chest
height sits entirely outside it and the corridor underneath is reported clear.
We accumulate depth points into a sparse 3D voxel grid over successive frames
and ask a different question: for this ground cell, is anything solid sitting
between the wheels and the top of the vehicle?

Sparse dict-of-voxels rather than a dense array: outdoor volumes are mostly
empty and the occupied set stays small.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

__all__ = ["VoxelGrid"]

_EPS = 1e-9


class VoxelGrid:
    """Sparse occupancy grid in the world frame.

    Axes follow the navigation convention: X right, Y forward, Z up.

    Args:
        resolution: voxel edge length in metres.
        hit_threshold: observations before a voxel is considered occupied.
    """

    def __init__(self, resolution: float = 0.15, hit_threshold: int = 2) -> None:
        if resolution <= 0:
            raise ValueError("resolution must be positive")
        self.resolution = float(resolution)
        self.hit_threshold = int(hit_threshold)
        self._hits: dict[tuple[int, int, int], int] = defaultdict(int)

    # ---------------------------------------------------------------- helpers
    def key(self, point) -> tuple[int, int, int]:
        """Voxel index containing a world point."""
        p = np.asarray(point, dtype=np.float64).reshape(3)
        # EPS guards against binary float error: 1.2 / 0.2 evaluates to
        # 5.999... , which would otherwise place the point one voxel too low.
        return tuple(np.floor(p / self.resolution + _EPS).astype(int).tolist())

    def centre(self, key) -> np.ndarray:
        """World-frame centre of a voxel."""
        return (np.asarray(key, dtype=np.float64) + 0.5) * self.resolution

    def __len__(self) -> int:
        return sum(1 for v in self._hits.values() if v >= self.hit_threshold)

    def occupied(self, point) -> bool:
        return self._hits.get(self.key(point), 0) >= self.hit_threshold

    def occupied_keys(self):
        return [k for k, v in self._hits.items() if v >= self.hit_threshold]

    # ------------------------------------------------------------- integration
    def integrate(self, points: np.ndarray, pose_R: np.ndarray | None = None,
                  pose_t: np.ndarray | None = None,
                  max_range: float = 15.0) -> int:
        """Fold a point cloud into the grid.

        Args:
            points: (..., 3) points in the sensor frame.
            pose_R: (3, 3) sensor-to-world rotation. Identity if omitted.
            pose_t: (3,) sensor-to-world translation. Zero if omitted.
            max_range: points beyond this distance are ignored, since
                monocular depth is unreliable far away.

        Returns:
            Number of points integrated.
        """
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        pts = pts[np.isfinite(pts).all(axis=1)]
        if pts.size == 0:
            return 0
        pts = pts[np.linalg.norm(pts, axis=1) <= max_range]
        if pts.size == 0:
            return 0
        if pose_R is not None:
            pts = pts @ np.asarray(pose_R, dtype=np.float64).T
        if pose_t is not None:
            pts = pts + np.asarray(pose_t, dtype=np.float64).reshape(1, 3)

        idx = np.floor(pts / self.resolution + _EPS).astype(int)
        for k in map(tuple, idx.tolist()):
            self._hits[k] += 1
        return int(pts.shape[0])

    # -------------------------------------------------------------- clearance
    def overhead_blocked(self, x: float, y: float, robot_height: float,
                         ground_z: float = 0.0,
                         floor_clearance: float = 0.05) -> bool:
        """Is the column above this ground cell obstructed?

        Args:
            x, y: ground position in world metres.
            robot_height: height of the vehicle including cargo, in metres.
            ground_z: world height of the ground at this cell.
            floor_clearance: ignore voxels this close to the ground, so the
                ground surface itself does not count as an overhead obstacle.

        Returns:
            True if any occupied voxel lies between the ground and the top of
            the vehicle, meaning the vehicle cannot pass.
        """
        if robot_height <= floor_clearance:
            return False
        kx = int(np.floor(x / self.resolution + _EPS))
        ky = int(np.floor(y / self.resolution + _EPS))
        z0 = int(np.floor((ground_z + floor_clearance) / self.resolution + _EPS))
        z1 = int(np.floor((ground_z + robot_height) / self.resolution + _EPS))
        for kz in range(min(z0, z1), max(z0, z1) + 1):
            if self._hits.get((kx, ky, kz), 0) >= self.hit_threshold:
                return True
        return False

    def clearance_map(self, x_range=(-4.0, 4.0), y_range=(0.5, 12.0),
                      cells: int = 32, robot_height: float = 1.5,
                      ground_z: float = 0.0) -> np.ndarray:
        """Top-down boolean map of cells blocked by overhead structure."""
        out = np.zeros((cells, cells), dtype=bool)
        dx = (x_range[1] - x_range[0]) / cells
        dy = (y_range[1] - y_range[0]) / cells
        # A map cell may span several voxel columns, so test every column that
        # overlaps the cell rather than probing a single sample point.
        step = min(self.resolution, dx, dy) * 0.5
        for j in range(cells):
            y0 = y_range[0] + j * dy
            ys = np.arange(y0, y0 + dy + step * 0.5, step)
            for i in range(cells):
                x0 = x_range[0] + i * dx
                xs = np.arange(x0, x0 + dx + step * 0.5, step)
                hit = any(self.overhead_blocked(float(x), float(y),
                                                robot_height, ground_z)
                          for y in ys for x in xs)
                out[cells - 1 - j, i] = hit
        return out
