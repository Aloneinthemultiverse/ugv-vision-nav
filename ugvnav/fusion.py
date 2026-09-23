"""Layer 3 preview - fusing every hazard claim into one safety grid.

The full system implements this as Nav2 ``costmap_2d`` plugins. This module is
the same decision logic in plain NumPy, so it can run and be tested without a
ROS installation.

Rule: the most pessimistic claim wins. A cell is only SAFE when nothing
objected to it. The geometry override lives here - it is the reason a rock
wearing a coat of dirt does not get driven over.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .layer1.semantics import traversable_mask

__all__ = ["SAFE", "CAUTION", "LETHAL", "UNKNOWN", "COST", "FusionResult",
           "geometry_override", "fuse"]

SAFE, CAUTION, LETHAL, UNKNOWN = 0, 1, 2, 3

#: Nav2 costmap values these states map onto.
COST = {SAFE: 0, CAUTION: 128, LETHAL: 254, UNKNOWN: 255}


@dataclass
class FusionResult:
    grid: np.ndarray          # (H, W) uint8 of SAFE / CAUTION / LETHAL / UNKNOWN
    override: np.ndarray      # (H, W) bool, cells rescued by geometry override
    reasons: dict[str, int]   # pixel counts per contributing rule

    @property
    def costmap(self) -> np.ndarray:
        """The grid expressed in Nav2 cost values."""
        out = np.zeros(self.grid.shape, dtype=np.uint8)
        for state, cost in COST.items():
            out[self.grid == state] = cost
        return out


def geometry_override(labels: np.ndarray, height: np.ndarray,
                      spike_m: float = 0.22) -> np.ndarray:
    """Cells semantics calls drivable but geometry says are not.

    A half-buried rock is labelled "grass" or "trail" because that is what its
    surface looks like. Its *shape* gives it away. Where appearance and height
    disagree, height wins.

    Args:
        labels: (H, W) semantic class ids.
        height: (H, W) signed height above the ground plane, metres.
        spike_m: height above the plane that counts as an obstacle.

    Returns:
        (H, W) bool mask of overridden pixels.
    """
    looks_drivable = traversable_mask(labels)
    is_bumpy = np.asarray(height, dtype=np.float32) > float(spike_m)
    return looks_drivable & is_bumpy


def fuse(labels: np.ndarray,
         height: np.ndarray,
         negative: np.ndarray | None = None,
         dynamic: np.ndarray | None = None,
         overhead: np.ndarray | None = None,
         uncertainty: np.ndarray | None = None,
         valid: np.ndarray | None = None,
         spike_m: float = 0.22,
         uncertainty_thresh: float = 0.6) -> FusionResult:
    """Combine every layer into one safety grid.

    Args:
        labels: (H, W) semantic class ids (Node B).
        height: (H, W) height above ground plane (Node A).
        negative: (H, W) bool drop-off mask (Node E).
        dynamic: (H, W) bool moving-object mask (Node F).
        overhead: (H, W) bool overhead-blocked mask (Node G).
        uncertainty: (H, W) uncertainty in [0, 1] (Node C).
        valid: (H, W) bool analysable region; elsewhere the grid is UNKNOWN.
        spike_m: geometry override threshold.
        uncertainty_thresh: above this, ground is downgraded to CAUTION.

    Returns:
        FusionResult.
    """
    labels = np.asarray(labels)
    shape = labels.shape
    grid = np.full(shape, SAFE, dtype=np.uint8)
    reasons: dict[str, int] = {}

    def apply(mask: np.ndarray | None, state: int, name: str) -> None:
        if mask is None:
            return
        m = np.asarray(mask, dtype=bool)
        # Never downgrade a stronger claim: LETHAL outranks CAUTION.
        m = m & (grid < state)
        grid[m] = state
        reasons[name] = int(m.sum())

    # --- non-traversable by appearance -------------------------------------
    apply(~traversable_mask(labels), LETHAL, "semantic_obstacle")

    # --- the geometry override ---------------------------------------------
    override = geometry_override(labels, height, spike_m)
    apply(override, LETHAL, "geometry_override")

    # --- explicit hazard layers --------------------------------------------
    apply(negative, LETHAL, "negative_obstacle")
    apply(dynamic, LETHAL, "dynamic_obstacle")
    apply(overhead, LETHAL, "overhead_clearance")

    # --- uncertainty only ever downgrades to CAUTION ------------------------
    if uncertainty is not None:
        apply(np.asarray(uncertainty, dtype=np.float32) > uncertainty_thresh,
              CAUTION, "uncertainty")

    if valid is not None:
        grid[~np.asarray(valid, dtype=bool)] = UNKNOWN

    return FusionResult(grid=grid, override=override, reasons=reasons)
