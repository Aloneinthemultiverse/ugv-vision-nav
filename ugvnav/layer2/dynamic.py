"""Node F - ego-motion compensated dynamic obstacle detection.

When the vehicle moves, everything in the image moves. Naive frame differencing
therefore flags the whole world. We first predict the optical flow the camera's
own motion would produce, then subtract it. What is left moving is genuinely
moving, and gets a velocity vector so the planner can avoid where the hazard
*will be* rather than where it was.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

__all__ = ["Track", "DynamicResult", "DynamicTracker"]


@dataclass
class Track:
    cx: float          # centroid x, pixels
    cy: float          # centroid y, pixels
    vx: float          # residual velocity x, pixels/frame
    vy: float          # residual velocity y, pixels/frame
    area: int          # blob area, pixels
    speed: float       # residual speed magnitude, pixels/frame

    def predict(self, frames: float) -> tuple[float, float]:
        """Where this object will be after ``frames`` more frames."""
        return self.cx + self.vx * frames, self.cy + self.vy * frames


@dataclass
class DynamicResult:
    mask: np.ndarray        # (H, W) bool, independently-moving pixels
    residual: np.ndarray    # (H, W) float32, residual flow magnitude
    flow: np.ndarray        # (H, W, 2) measured optical flow
    tracks: list[Track]


class DynamicTracker:
    """Detects independently-moving objects from two consecutive frames.

    Args:
        thresh_px: residual flow, in pixels/frame, above which a pixel counts
            as independently moving.
        min_area: minimum blob area to report as a track.
    """

    def __init__(self, thresh_px: float = 1.5, min_area: int = 60) -> None:
        self.thresh_px = float(thresh_px)
        self.min_area = int(min_area)

    @staticmethod
    def _gray(img: np.ndarray) -> np.ndarray:
        img = np.asarray(img)
        if img.ndim == 3:
            return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        return img.astype(np.uint8)

    @staticmethod
    def ego_flow(shape: tuple[int, int], H: np.ndarray) -> np.ndarray:
        """Flow field implied by a background homography.

        Args:
            shape: (H, W) image shape.
            H: 3x3 homography mapping previous-frame pixels to current-frame.

        Returns:
            (H, W, 2) predicted displacement for every pixel.
        """
        h, w = shape
        us, vs = np.meshgrid(np.arange(w, dtype=np.float64),
                             np.arange(h, dtype=np.float64))
        ones = np.ones_like(us)
        pts = np.stack([us, vs, ones], axis=-1).reshape(-1, 3).T
        warped = np.asarray(H, dtype=np.float64) @ pts
        warped = warped[:2] / np.where(np.abs(warped[2]) < 1e-9, 1e-9, warped[2])
        pred = warped.T.reshape(h, w, 2)
        return (pred - np.stack([us, vs], axis=-1)).astype(np.float32)

    def background_homography(self, prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        """Estimate the dominant (ego-motion) image transform between frames.

        Falls back to identity when too few features are available.
        """
        g0, g1 = self._gray(prev), self._gray(curr)
        orb = cv2.ORB_create(1200)
        k0, d0 = orb.detectAndCompute(g0, None)
        k1, d1 = orb.detectAndCompute(g1, None)
        if d0 is None or d1 is None or len(k0) < 8 or len(k1) < 8:
            return np.eye(3)
        pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d0, d1, k=2)
        good = [m for m, n in (p for p in pairs if len(p) == 2)
                if m.distance < 0.75 * n.distance]
        if len(good) < 8:
            return np.eye(3)
        p0 = np.float64([k0[m.queryIdx].pt for m in good])
        p1 = np.float64([k1[m.trainIdx].pt for m in good])
        H, _ = cv2.findHomography(p0, p1, cv2.RANSAC, 3.0)
        return np.eye(3) if H is None else H

    def process(self, prev: np.ndarray, curr: np.ndarray,
                H: np.ndarray | None = None) -> DynamicResult:
        """Args:
            prev, curr: consecutive frames (RGB or grayscale).
            H: optional known background homography. Estimated if omitted.

        Returns:
            DynamicResult with one Track per independently-moving blob.
        """
        g0, g1 = self._gray(prev), self._gray(curr)
        if H is None:
            H = self.background_homography(g0, g1)

        flow = cv2.calcOpticalFlowFarneback(
            g0, g1, None, 0.5, 3, 21, 3, 5, 1.2, 0)
        residual = flow - self.ego_flow(g0.shape, H)
        mag = np.linalg.norm(residual, axis=-1).astype(np.float32)

        mask = mag > self.thresh_px
        mask = cv2.morphologyEx(mask.astype(np.uint8),
                                cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                np.ones((9, 9), np.uint8)).astype(bool)

        tracks: list[Track] = []
        n, lab, stats, cent = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), 8)
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < self.min_area:
                continue
            sel = lab == i
            vx = float(residual[..., 0][sel].mean())
            vy = float(residual[..., 1][sel].mean())
            tracks.append(Track(float(cent[i][0]), float(cent[i][1]),
                                vx, vy, area, float(np.hypot(vx, vy))))
        tracks.sort(key=lambda t: t.area, reverse=True)
        return DynamicResult(mask, mag, flow.astype(np.float32), tracks)
