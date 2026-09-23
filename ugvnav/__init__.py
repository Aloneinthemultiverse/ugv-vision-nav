"""Vision-based autonomous navigation for UGVs in outdoor environments.

Layer 1 (perception) and Layer 2 (hazard resolution), plus a NumPy preview of
the Layer 3 costmap fusion. Reference implementation for SIH26126.
"""
__version__ = "0.1.0"

from .camera import Camera, fit_ground_plane, plane_heights
from .fusion import SAFE, CAUTION, LETHAL, UNKNOWN, COST, fuse, geometry_override

__all__ = [
    "Camera", "fit_ground_plane", "plane_heights",
    "SAFE", "CAUTION", "LETHAL", "UNKNOWN", "COST", "fuse", "geometry_override",
]
