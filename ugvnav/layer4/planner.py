"""Layer 4 - planning and control.

Three stages, mirroring the Nav2 chain the full system targets:

  * ``AStarPlanner``    global search over the inflated costmap (Smac's role)
  * ``MPPIPlanner``     sampling-based local planner that rolls out candidate
                        control sequences and takes a cost-weighted average
                        (Nav2's MPPI controller)
  * ``PurePursuit``     turns the chosen path into wheel commands

The planners read *cost*, not occupancy, so a CAUTION cell is expensive but
still passable - which is exactly how the uncertainty layer slows the vehicle
down instead of trapping it.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

from ..layer3.costmap import Costmap, INSCRIBED, LETHAL

__all__ = ["AStarPlanner", "MPPIPlanner", "PurePursuit", "Twist"]

_NEIGHBOURS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
               (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
               (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]


@dataclass
class Twist:
    """Velocity command, the ROS ``geometry_msgs/Twist`` subset we need."""
    linear: float    # m/s
    angular: float   # rad/s


class AStarPlanner:
    """Grid A* that prefers cheap cells rather than merely legal ones.

    Args:
        costmap: the inflated costmap to search.
        lethal_cost: cells at or above this are impassable.
        cost_weight: how strongly to avoid expensive-but-passable cells.
    """

    def __init__(self, costmap: Costmap, lethal_cost: int = INSCRIBED,
                 cost_weight: float = 0.03) -> None:
        self.cm = costmap
        self.lethal_cost = int(lethal_cost)
        self.cost_weight = float(cost_weight)

    def _passable(self, row: int, col: int) -> bool:
        return int(self.cm.grid[row, col]) < self.lethal_cost

    def plan_cells(self, start_rc: tuple[int, int],
                   goal_rc: tuple[int, int]) -> list[tuple[int, int]]:
        """A* in cell space. Returns [] when no route exists."""
        g = self.cm.grid
        h, w = g.shape
        sr, sc = start_rc
        gr, gc = goal_rc
        for r, c in ((sr, sc), (gr, gc)):
            if not (0 <= r < h and 0 <= c < w):
                return []
        if not self._passable(sr, sc) or not self._passable(gr, gc):
            return []

        def heur(r, c):
            return math.hypot(r - gr, c - gc)

        open_q = [(heur(sr, sc), 0.0, (sr, sc))]
        came: dict[tuple[int, int], tuple[int, int]] = {}
        best = {(sr, sc): 0.0}
        closed: set[tuple[int, int]] = set()

        while open_q:
            _, cost, node = heapq.heappop(open_q)
            if node in closed:
                continue
            closed.add(node)
            if node == (gr, gc):
                path = [node]
                while path[-1] in came:
                    path.append(came[path[-1]])
                return path[::-1]

            r, c = node
            for dr, dc, step in _NEIGHBOURS:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w) or (nr, nc) in closed:
                    continue
                if not self._passable(nr, nc):
                    continue
                nxt = cost + step + self.cost_weight * float(g[nr, nc])
                if nxt < best.get((nr, nc), math.inf):
                    best[(nr, nc)] = nxt
                    came[(nr, nc)] = node
                    heapq.heappush(open_q, (nxt + heur(nr, nc), nxt, (nr, nc)))
        return []

    def plan(self, start_xy: tuple[float, float],
             goal_xy: tuple[float, float]) -> np.ndarray:
        """Plan in world metres. Returns an (N, 2) path, empty if unreachable."""
        sc, sr = self.cm.spec.world_to_cell(*start_xy)
        gc, gr = self.cm.spec.world_to_cell(*goal_xy)
        cells = self.plan_cells((int(sr), int(sc)), (int(gr), int(gc)))
        if not cells:
            return np.empty((0, 2))
        pts = [self.cm.spec.cell_to_world(c, r) for r, c in cells]
        return np.array([(float(x), float(y)) for x, y in pts])


class MPPIPlanner:
    """Model Predictive Path Integral local planner.

    Samples many control sequences, rolls each through a unicycle model, scores
    the resulting trajectories against the costmap and the reference path, then
    returns the cost-weighted average control. Smooth, and naturally reactive to
    obstacles that appear between global replans.

    Args:
        costmap: costmap to evaluate trajectories against.
        horizon: rollout steps.
        dt: seconds per step.
        samples: number of candidate sequences.
        temperature: MPPI lambda. Lower = greedier.
    """

    def __init__(self, costmap: Costmap, horizon: int = 20, dt: float = 0.1,
                 samples: int = 400, temperature: float = 0.25,
                 v_max: float = 1.2, w_max: float = 1.5,
                 v_std: float = 0.35, w_std: float = 0.6,
                 seed: int = 0) -> None:
        self.cm = costmap
        self.horizon = int(horizon)
        self.dt = float(dt)
        self.samples = int(samples)
        self.temperature = float(temperature)
        self.v_max, self.w_max = float(v_max), float(w_max)
        self.v_std, self.w_std = float(v_std), float(w_std)
        self.rng = np.random.default_rng(seed)

    def rollout(self, state: tuple[float, float, float],
                v: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Unicycle rollout. Returns (S, H, 3) of (x, y, theta)."""
        x = np.full(v.shape[0], state[0], dtype=float)
        y = np.full(v.shape[0], state[1], dtype=float)
        th = np.full(v.shape[0], state[2], dtype=float)
        out = np.empty((v.shape[0], self.horizon, 3))
        for k in range(self.horizon):
            th = th + w[:, k] * self.dt
            # Convention, shared with Vehicle and PoseFilter: +Y forward,
            # +X right, positive theta counter-clockwise (toward -X).
            x = x - v[:, k] * np.sin(th) * self.dt
            y = y + v[:, k] * np.cos(th) * self.dt
            out[:, k] = np.stack([x, y, th], axis=-1)
        return out

    def _traj_cost(self, traj: np.ndarray, reference: np.ndarray) -> np.ndarray:
        cols, rows = self.cm.spec.world_to_cell(traj[..., 0], traj[..., 1])
        h, w = self.cm.grid.shape
        off = (cols < 0) | (cols >= w) | (rows < 0) | (rows >= h)
        cc = np.clip(cols, 0, w - 1)
        rr = np.clip(rows, 0, h - 1)
        cell = self.cm.grid[rr, cc].astype(float)
        cell[off] = float(LETHAL)

        obstacle = cell.sum(axis=1)
        obstacle += (cell >= INSCRIBED).sum(axis=1) * 5000.0   # hard penalty

        goal = reference[-1]
        to_goal = np.linalg.norm(traj[:, -1, :2] - goal, axis=1) * 120.0
        return obstacle + to_goal

    def plan(self, state: tuple[float, float, float],
             reference: np.ndarray) -> tuple[Twist, np.ndarray]:
        """Compute the next command.

        Args:
            state: (x, y, theta) of the vehicle, theta measured from +Y.
            reference: (N, 2) path from the global planner.

        Returns:
            (command, best_trajectory).
        """
        if reference.size == 0:
            return Twist(0.0, 0.0), np.empty((0, 3))

        v = np.clip(self.rng.normal(self.v_max * 0.6, self.v_std,
                                    (self.samples, self.horizon)), 0.0, self.v_max)
        w = np.clip(self.rng.normal(0.0, self.w_std,
                                    (self.samples, self.horizon)), -self.w_max, self.w_max)

        traj = self.rollout(state, v, w)
        cost = self._traj_cost(traj, reference)

        weights = np.exp(-(cost - cost.min()) / max(self.temperature * cost.std(), 1e-6))
        weights /= weights.sum()
        vb = float((weights[:, None] * v).sum(axis=0)[0])
        wb = float((weights[:, None] * w).sum(axis=0)[0])
        return Twist(vb, wb), traj[int(np.argmin(cost))]


class PurePursuit:
    """Regulated pure-pursuit controller.

    Chases a lookahead point on the path, and slows down both for curvature and
    for CAUTION cost - which is how the uncertainty layer actually changes the
    vehicle's behaviour rather than just colouring a map.
    """

    def __init__(self, lookahead: float = 1.2, v_nominal: float = 1.0,
                 v_min: float = 0.2, w_max: float = 1.5,
                 clutter_radius: float = 1.6, clutter_gain: float = 0.0) -> None:
        self.lookahead = float(lookahead)
        self.v_nominal = float(v_nominal)
        self.v_min = float(v_min)
        self.w_max = float(w_max)
        self.clutter_radius = float(clutter_radius)
        # Measured: scaling speed by clutter REGRESSED closed-loop success
        # (easy 60% -> 52%, hard 4% -> 0%). Because omega = curvature * v in
        # pure pursuit, slowing down also slows the turn, so the vehicle drives
        # slowly into the obstacle instead of turning out of the way. Kept as an
        # opt-in knob, default off, with the measurement recorded rather than
        # the idea silently retained.
        self.clutter_gain = float(clutter_gain)

    def clutter(self, costmap: "Costmap", x: float, y: float) -> float:
        """Fraction of the neighbourhood around (x, y) that is impassable.

        Available for callers that want it, but disabled by default: see the
        note on ``clutter_gain`` for the measurement that rejected it.
        """
        spec = costmap.spec
        n = max(int(self.clutter_radius / spec.resolution), 1)
        col, row = spec.world_to_cell(x, y)
        c0 = max(int(col) - n, 0); c1 = min(int(col) + n + 1, spec.width)
        r0 = max(int(row) - n, 0); r1 = min(int(row) + n + 1, spec.height)
        if c1 <= c0 or r1 <= r0:
            return 0.0
        return float((costmap.grid[r0:r1, c0:c1] >= INSCRIBED).mean())

    def target(self, state: tuple[float, float, float],
               path: np.ndarray) -> np.ndarray | None:
        """First path point at least ``lookahead`` metres away."""
        if path.size == 0:
            return None
        d = np.linalg.norm(path - np.asarray(state[:2]), axis=1)
        ahead = np.flatnonzero(d >= self.lookahead)
        return path[ahead[0]] if ahead.size else path[-1]

    def step(self, state: tuple[float, float, float], path: np.ndarray,
             costmap: Costmap | None = None) -> Twist:
        """Produce the wheel command for this instant."""
        tgt = self.target(state, path)
        if tgt is None:
            return Twist(0.0, 0.0)

        x, y, th = state
        dx, dy = tgt[0] - x, tgt[1] - y
        # rotate the target into the body frame (+Y forward)
        fx = dx * math.cos(-th) - dy * math.sin(-th)
        fy = dx * math.sin(-th) + dy * math.cos(-th)

        dist = math.hypot(fx, fy)
        if dist < 1e-6:
            return Twist(0.0, 0.0)

        # Pure-pursuit curvature. fx is the lateral offset of the target in the
        # body frame, positive to the right. Because positive yaw rate turns the
        # vehicle counter-clockwise (toward -X), steering toward a target on the
        # right requires a NEGATIVE omega - hence the sign here. Getting this
        # wrong makes the controller steer away from its target; see
        # test_controller_and_vehicle_model_agree_on_turn_direction.
        curvature = -2.0 * fx / (dist ** 2)
        v = self.v_nominal / (1.0 + 1.5 * abs(curvature))

        if costmap is not None:
            c = costmap.cost_at(x, y)
            if c >= INSCRIBED:
                v = self.v_min
            elif c > 0:
                v *= max(0.25, 1.0 - c / 255.0)
            if self.clutter_gain > 0.0:
                v *= max(0.3, 1.0 - self.clutter_gain * self.clutter(costmap, x, y))

        v = max(self.v_min, min(v, self.v_nominal))
        return Twist(v, float(np.clip(curvature * v, -self.w_max, self.w_max)))
