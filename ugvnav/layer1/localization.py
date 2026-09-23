"""Node D+ - GPS-denied localization: sensor fusion, loop closure, pose graph.

``odometry.VisualOdometry`` produces frame-to-frame motion and nothing else, so
its error grows without bound and monocular translation has no scale. That is
not localization. This module adds the three pieces that make it one:

  * ``PoseFilter``           an EKF on SE(2) fusing visual odometry, IMU yaw
                             rate and wheel odometry. Wheel odometry supplies
                             the metric scale monocular vision cannot observe;
                             the IMU carries heading through wheel slip.
  * ``LoopClosureDetector``  recognises somewhere the vehicle has already been,
                             from compact ORB descriptor signatures.
  * ``PoseGraph``            redistributes accumulated drift around a closed
                             loop by Gauss-Newton on SE(2).

All three are pure NumPy/OpenCV and are tested against trajectories with exactly
known ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["wrap_angle", "PoseFilter", "LoopClosureDetector", "PoseGraph"]


def wrap_angle(a):
    """Wrap angle(s) to the half-open interval [-pi, pi).

    Note the interval is closed at -pi and open at +pi, so ``wrap_angle(pi)``
    returns ``-pi``. Both represent the same rotation; the convention only
    matters when comparing wrapped values for equality.
    """
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


# =============================================================== EKF fusion
class PoseFilter:
    """Extended Kalman filter over ``[x, y, theta]`` in the world frame.

    Prediction uses a unicycle model driven by wheel speed and yaw rate.
    Corrections come from whichever sensors report: visual odometry gives a
    relative motion, the IMU gives heading rate, the wheels give speed.

    Args:
        q: process noise (x, y, theta) per step.
        r_vo: visual-odometry measurement noise.
        r_yaw: IMU heading measurement noise.
    """

    def __init__(self, q=(0.02, 0.02, 0.01), r_vo=(0.08, 0.08, 0.05),
                 r_yaw: float = 0.02) -> None:
        self.x = np.zeros(3)                      # [x, y, theta]
        self.P = np.eye(3) * 0.01
        self.Q = np.diag(np.asarray(q, float) ** 2)
        self.R_vo = np.diag(np.asarray(r_vo, float) ** 2)
        self.r_yaw = float(r_yaw) ** 2

    @property
    def pose(self) -> np.ndarray:
        return self.x.copy()

    def predict(self, v: float, omega: float, dt: float) -> np.ndarray:
        """Propagate with wheel speed ``v`` and yaw rate ``omega``."""
        th = self.x[2]
        # +Y is forward at theta = 0, matching the costmap convention
        self.x = self.x + np.array([-v * np.sin(th) * dt,
                                    v * np.cos(th) * dt,
                                    omega * dt])
        self.x[2] = float(wrap_angle(self.x[2]))

        F = np.eye(3)
        F[0, 2] = -v * np.cos(th) * dt
        F[1, 2] = -v * np.sin(th) * dt
        self.P = F @ self.P @ F.T + self.Q * dt
        return self.pose

    def update_visual_odometry(self, dx: float, dy: float, dtheta: float) -> np.ndarray:
        """Correct with a body-frame relative motion from visual odometry.

        Args:
            dx, dy: translation in the *previous body frame*, metres. The scale
                must already have been resolved (wheel odometry or ground plane).
            dtheta: heading change, radians.
        """
        th = self.x[2] - dtheta                   # heading when the motion began
        z = np.array([self.x[0] + (dx * np.cos(th) - dy * np.sin(th)),
                      self.x[1] + (dx * np.sin(th) + dy * np.cos(th)),
                      wrap_angle(self.x[2])])
        return self._correct(z, np.eye(3), self.R_vo)

    def update_heading(self, theta: float) -> np.ndarray:
        """Correct heading alone, from the IMU."""
        H = np.array([[0.0, 0.0, 1.0]])
        y = np.array([wrap_angle(theta - self.x[2])])
        S = H @ self.P @ H.T + self.r_yaw
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y).ravel()
        self.x[2] = float(wrap_angle(self.x[2]))
        self.P = (np.eye(3) - K @ H) @ self.P
        return self.pose

    def _correct(self, z, H, R):
        y = z - H @ self.x
        y[2] = float(wrap_angle(y[2]))
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[2] = float(wrap_angle(self.x[2]))
        self.P = (np.eye(3) - K @ H) @ self.P
        return self.pose


# ============================================================ loop closure
@dataclass
class LoopClosureDetector:
    """Recognises previously-visited places from ORB descriptor signatures.

    A full system would use a bag-of-words vocabulary. For a bounded outdoor
    run, a compact per-keyframe signature - the mean of binary descriptors,
    which is cheap and rotation-tolerant enough at the place level - separates
    revisits from new ground well enough to trigger a pose-graph correction.

    Args:
        min_gap: keyframes that must pass before a match counts as a loop,
            so consecutive frames never self-trigger.
        threshold: cosine similarity required to accept a candidate.
    """

    min_gap: int = 12
    threshold: float = 0.88
    signatures: list = field(default_factory=list)

    @staticmethod
    def signature(image: np.ndarray, n: int = 600) -> np.ndarray:
        """Compact appearance descriptor for one frame."""
        import cv2

        img = np.asarray(image)
        if img.ndim == 3:
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        _, des = cv2.ORB_create(n).detectAndCompute(img.astype(np.uint8), None)
        if des is None or len(des) == 0:
            return np.zeros(256, np.float32)
        bits = np.unpackbits(des, axis=1).astype(np.float32)   # (N, 256)
        sig = bits.mean(axis=0)
        norm = np.linalg.norm(sig)
        return sig / norm if norm > 1e-9 else sig

    def add(self, image_or_signature) -> int:
        """Register a keyframe. Returns its index."""
        sig = (np.asarray(image_or_signature, np.float32)
               if np.ndim(image_or_signature) == 1
               else self.signature(image_or_signature))
        self.signatures.append(sig)
        return len(self.signatures) - 1

    def detect(self, image_or_signature) -> tuple[int, float] | None:
        """Best loop candidate for this frame, or None.

        Returns:
            (index_of_matched_keyframe, similarity) if a revisit is found.
        """
        sig = (np.asarray(image_or_signature, np.float32)
               if np.ndim(image_or_signature) == 1
               else self.signature(image_or_signature))
        limit = len(self.signatures) - self.min_gap
        if limit <= 0:
            return None
        past = np.stack(self.signatures[:limit])
        scores = past @ sig
        i = int(np.argmax(scores))
        return (i, float(scores[i])) if scores[i] >= self.threshold else None


# ============================================================== pose graph
class PoseGraph:
    """SE(2) pose graph with Gauss-Newton optimisation.

    Odometry edges chain the trajectory; a loop edge says "node i and node j are
    the same place". Optimising redistributes the accumulated drift around the
    loop instead of letting it sit at the end.
    """

    def __init__(self) -> None:
        self.nodes: list[np.ndarray] = []
        self.edges: list[tuple[int, int, np.ndarray, float]] = []

    def add_node(self, pose) -> int:
        self.nodes.append(np.asarray(pose, dtype=float).copy())
        return len(self.nodes) - 1

    def add_edge(self, i: int, j: int, measurement, weight: float = 1.0) -> None:
        """Constrain node j relative to node i by a body-frame measurement."""
        self.edges.append((int(i), int(j), np.asarray(measurement, float), float(weight)))

    @staticmethod
    def relative(a, b) -> np.ndarray:
        """Pose of b expressed in a's frame."""
        a = np.asarray(a, float); b = np.asarray(b, float)
        c, s = np.cos(-a[2]), np.sin(-a[2])
        d = b[:2] - a[:2]
        return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1],
                         float(wrap_angle(b[2] - a[2]))])

    def residual(self) -> float:
        """Total squared error over all edges."""
        total = 0.0
        for i, j, m, w in self.edges:
            e = self.relative(self.nodes[i], self.nodes[j]) - m
            e[2] = float(wrap_angle(e[2]))
            total += w * float(e @ e)
        return total

    def optimise(self, iterations: int = 30, damping: float = 1e-6,
                 fix_first: bool = True) -> float:
        """Gauss-Newton with numeric Jacobians. Returns the final residual."""
        n = len(self.nodes)
        if n < 2 or not self.edges:
            return self.residual()

        for _ in range(iterations):
            H = np.zeros((3 * n, 3 * n))
            g = np.zeros(3 * n)
            for i, j, m, w in self.edges:
                e = self.relative(self.nodes[i], self.nodes[j]) - m
                e[2] = float(wrap_angle(e[2]))
                Ji, Jj = self._jacobians(i, j, m)
                for a, Ja in ((i, Ji), (j, Jj)):
                    g[3 * a:3 * a + 3] += w * (Ja.T @ e)
                    for b, Jb in ((i, Ji), (j, Jj)):
                        H[3 * a:3 * a + 3, 3 * b:3 * b + 3] += w * (Ja.T @ Jb)

            if fix_first:                      # anchor node 0, else gauge freedom
                H[:3, :] = 0.0; H[:, :3] = 0.0
                H[:3, :3] = np.eye(3); g[:3] = 0.0

            try:
                dx = np.linalg.solve(H + damping * np.eye(3 * n), -g)
            except np.linalg.LinAlgError:
                break
            for k in range(n):
                self.nodes[k] = self.nodes[k] + dx[3 * k:3 * k + 3]
                self.nodes[k][2] = float(wrap_angle(self.nodes[k][2]))
            if np.linalg.norm(dx) < 1e-9:
                break
        return self.residual()

    def _jacobians(self, i, j, m, eps=1e-6):
        Ji = np.zeros((3, 3)); Jj = np.zeros((3, 3))
        base = self.relative(self.nodes[i], self.nodes[j]) - m
        base[2] = float(wrap_angle(base[2]))
        for k in range(3):
            for node, J in ((i, Ji), (j, Jj)):
                saved = self.nodes[node][k]
                self.nodes[node][k] = saved + eps
                pert = self.relative(self.nodes[i], self.nodes[j]) - m
                pert[2] = float(wrap_angle(pert[2]))
                J[:, k] = (pert - base) / eps
                self.nodes[node][k] = saved
        return Ji, Jj
