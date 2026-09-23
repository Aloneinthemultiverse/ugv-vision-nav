"""Layer 3 - the multi-layer costmap.

Every hazard claim from Layer 2 is written into one top-down grid in the
vehicle's frame, then inflated by the vehicle's radius so the planner can treat
the robot as a point. This mirrors the Nav2 ``costmap_2d`` plugin chain, but in
plain NumPy so it runs and is testable without ROS.

Cost convention matches Nav2 exactly:
    0    free
    128  caution - traversable, but speed-limited
    253  inscribed - a point robot centre here would collide
    254  lethal
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["FREE", "CAUTION", "INSCRIBED", "LETHAL", "GridSpec", "Costmap"]

FREE, CAUTION, INSCRIBED, LETHAL = 0, 128, 253, 254


@dataclass(frozen=True)
class GridSpec:
    """Metric definition of a top-down grid.

    The vehicle sits at the bottom-centre looking along +Y.

    Args:
        width, height: grid size in cells.
        resolution: metres per cell.
        origin_x: world X of the grid's left edge.
        origin_y: world Y of the grid's bottom edge.
    """

    width: int = 80
    height: int = 80
    resolution: float = 0.1
    origin_x: float = -4.0
    origin_y: float = 0.0

    def world_to_cell(self, x, y):
        """Metres -> (col, row). Row 0 is the far edge, as rendered."""
        cx = np.floor((np.asarray(x) - self.origin_x) / self.resolution).astype(int)
        ry = np.floor((np.asarray(y) - self.origin_y) / self.resolution).astype(int)
        return cx, self.height - 1 - ry

    def cell_to_world(self, col, row):
        """(col, row) -> metres at the cell centre."""
        x = self.origin_x + (np.asarray(col) + 0.5) * self.resolution
        y = self.origin_y + (self.height - 1 - np.asarray(row) + 0.5) * self.resolution
        return x, y

    def inside(self, col, row) -> bool:
        return 0 <= int(col) < self.width and 0 <= int(row) < self.height


@dataclass
class Costmap:
    """Layered occupancy costmap.

    Args:
        spec: grid geometry.
        robot_radius: footprint radius in metres, used for inflation.
        inflation_radius: distance over which cost decays away from obstacles.
    """

    spec: GridSpec = field(default_factory=GridSpec)
    robot_radius: float = 0.35
    inflation_radius: float = 0.9
    grid: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.grid = np.zeros((self.spec.height, self.spec.width), dtype=np.uint8)

    # ----------------------------------------------------------------- basics
    def reset(self) -> None:
        self.grid[:] = FREE

    def _raise_to(self, mask: np.ndarray, cost: int) -> int:
        """Write ``cost`` only where it exceeds what is already there."""
        m = np.asarray(mask, dtype=bool) & (self.grid < cost)
        self.grid[m] = cost
        return int(m.sum())

    def mark(self, x, y, cost: int = LETHAL) -> int:
        """Stamp individual world points into the grid."""
        col, row = self.spec.world_to_cell(x, y)
        col = np.atleast_1d(col); row = np.atleast_1d(row)
        ok = (col >= 0) & (col < self.spec.width) & (row >= 0) & (row < self.spec.height)
        mask = np.zeros_like(self.grid, dtype=bool)
        mask[row[ok], col[ok]] = True
        return self._raise_to(mask, cost)

    # -------------------------------------------------------- hazard layers
    def add_virtual_scan(self, ranges: np.ndarray, angles: np.ndarray,
                         cost: int = LETHAL) -> int:
        """Inject Node E's per-column drop-off scan as if it were laser returns.

        Args:
            ranges: distance per ray, metres. Non-finite rays are skipped.
            angles: bearing per ray, radians, 0 = straight ahead (+Y).
        """
        r = np.asarray(ranges, dtype=float)
        a = np.asarray(angles, dtype=float)
        ok = np.isfinite(r) & (r > 0)
        if not ok.any():
            return 0
        return self.mark(r[ok] * np.sin(a[ok]), r[ok] * np.cos(a[ok]), cost)

    def add_dynamic(self, x: float, y: float, vx: float, vy: float,
                    horizon_s: float = 2.0, steps: int = 8,
                    radius: float = 0.4, cost: int = LETHAL) -> int:
        """Sweep a moving obstacle forward in time and block where it *will* be.

        This is the point of tracking velocity: avoiding where a hazard was is
        too late.
        """
        total = 0
        for k in range(steps + 1):
            t = horizon_s * k / max(steps, 1)
            total += self.add_disc(x + vx * t, y + vy * t, radius, cost)
        return total

    def add_disc(self, x: float, y: float, radius: float,
                 cost: int = LETHAL) -> int:
        """Stamp a filled metric circle."""
        cols = np.arange(self.spec.width)
        rows = np.arange(self.spec.height)
        wx, wy = self.spec.cell_to_world(cols[None, :], rows[:, None])
        return self._raise_to((wx - x) ** 2 + (wy - y) ** 2 <= radius ** 2, cost)

    def add_mask(self, mask: np.ndarray, cost: int) -> int:
        """Merge a same-shaped boolean layer, e.g. overhead clearance."""
        return self._raise_to(mask, cost)

    # ------------------------------------------------------------- inflation
    def inflate(self) -> np.ndarray:
        """Grow obstacles by the robot radius and decay cost outward.

        After inflation the planner may treat the vehicle as a single point.
        """
        import cv2

        lethal = (self.grid >= LETHAL).astype(np.uint8)
        if lethal.any():
            dist = cv2.distanceTransform(1 - lethal, cv2.DIST_L2, 5) * self.spec.resolution
        else:
            dist = np.full(self.grid.shape, np.inf, dtype=np.float32)

        self._raise_to(dist <= self.robot_radius, INSCRIBED)

        band = (dist > self.robot_radius) & (dist <= self.inflation_radius)
        if band.any():
            span = max(self.inflation_radius - self.robot_radius, 1e-6)
            decay = 1.0 - (dist - self.robot_radius) / span
            vals = (CAUTION * decay).astype(np.uint8)
            upd = band & (vals > self.grid)
            self.grid[upd] = vals[upd]
        return self.grid

    # ------------------------------------------------------------- accessors
    def cost_at(self, x: float, y: float) -> int:
        col, row = self.spec.world_to_cell(x, y)
        if not self.spec.inside(col, row):
            return LETHAL
        return int(self.grid[int(row), int(col)])

    def is_free(self, x: float, y: float, limit: int = INSCRIBED) -> bool:
        return self.cost_at(x, y) < limit
