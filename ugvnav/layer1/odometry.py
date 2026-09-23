"""Node D - visual odometry / localization without GPS.

Sparse feature-based monocular VO: ORB features, ratio-test matching, essential
matrix, pose recovery. Monocular translation is recovered only up to scale, so
the caller supplies a scale hint (wheel odometry, or the ground-plane fit from
Node A). That fusion is what lets the stack keep a pose estimate when GPS is
denied and the wheels are slipping.

``estimate_relative_pose`` is deliberately separate from image handling so it
can be tested against exactly-known synthetic camera motion.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

__all__ = ["PoseDelta", "estimate_relative_pose", "VisualOdometry"]


@dataclass
class PoseDelta:
    R: np.ndarray                 # (3, 3) rotation, previous -> current
    t: np.ndarray                 # (3,) unit translation direction
    inliers: int                  # number of inlier correspondences
    valid: bool                   # False when the estimate is degenerate


def estimate_relative_pose(pts_prev: np.ndarray, pts_curr: np.ndarray,
                           K: np.ndarray, thresh: float = 1.0) -> PoseDelta:
    """Recover relative camera pose from matched image points.

    Args:
        pts_prev: (N, 2) pixel coordinates in the previous frame.
        pts_curr: (N, 2) corresponding coordinates in the current frame.
        K: (3, 3) camera intrinsics.
        thresh: RANSAC reprojection threshold in pixels.

    Returns:
        PoseDelta. ``t`` is a unit vector: monocular VO cannot observe scale.
    """
    p0 = np.asarray(pts_prev, dtype=np.float64).reshape(-1, 2)
    p1 = np.asarray(pts_curr, dtype=np.float64).reshape(-1, 2)
    if p0.shape[0] < 6 or p0.shape != p1.shape:
        return PoseDelta(np.eye(3), np.zeros(3), 0, False)

    E, mask = cv2.findEssentialMat(p0, p1, K, method=cv2.RANSAC,
                                   prob=0.999, threshold=thresh)
    if E is None or E.shape != (3, 3):
        return PoseDelta(np.eye(3), np.zeros(3), 0, False)

    n, R, t, _ = cv2.recoverPose(E, p0, p1, K, mask=mask)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    norm = np.linalg.norm(t)
    if norm > 1e-9:
        t = t / norm
    return PoseDelta(np.asarray(R, dtype=np.float64), t, int(n), int(n) >= 6)


@dataclass
class VisualOdometry:
    """Frame-to-frame VO with an accumulated pose.

    Args:
        K: camera intrinsics.
        max_features: ORB feature budget per frame.
        ratio: Lowe ratio-test threshold.
    """

    K: np.ndarray
    max_features: int = 1500
    ratio: float = 0.75
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    p: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self) -> None:
        self._orb = cv2.ORB_create(self.max_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self._prev: tuple | None = None

    @staticmethod
    def _gray(image: np.ndarray) -> np.ndarray:
        image = np.asarray(image)
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image.astype(np.uint8)

    def match(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """Detect and match against the stored previous frame.

        Returns None on the first frame or when matching fails.
        """
        kp, des = self._orb.detectAndCompute(self._gray(image), None)
        prev, self._prev = self._prev, (kp, des)
        if prev is None or des is None or prev[1] is None:
            return None
        kp0, des0 = prev
        if len(kp0) < 8 or len(kp) < 8:
            return None

        pairs = self._matcher.knnMatch(des0, des, k=2)
        good = [m for m, n in (p for p in pairs if len(p) == 2)
                if m.distance < self.ratio * n.distance]
        if len(good) < 8:
            return None
        p0 = np.float64([kp0[m.queryIdx].pt for m in good])
        p1 = np.float64([kp[m.trainIdx].pt for m in good])
        return p0, p1

    def process(self, image: np.ndarray, scale: float = 1.0) -> PoseDelta:
        """Ingest a frame and update the accumulated pose.

        Args:
            image: RGB or grayscale frame.
            scale: metres travelled since the previous frame, from wheel
                odometry or the ground-plane fit. Monocular VO cannot supply it.
        """
        matched = self.match(image)
        if matched is None:
            return PoseDelta(np.eye(3), np.zeros(3), 0, False)

        delta = estimate_relative_pose(matched[0], matched[1], self.K)
        if delta.valid:
            self.p = self.p + self.R @ (delta.t * float(scale))
            self.R = self.R @ delta.R
        return delta
