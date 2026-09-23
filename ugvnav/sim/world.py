"""Closed-loop evaluation without a vehicle.

Still-frame IoU says whether perception labels pixels correctly. It does not say
whether the robot *arrives*. This module closes the loop: a synthetic world, a
vehicle with unicycle kinematics, a sensor that only reveals what is in range
and in front, and the real Layer 3 + Layer 4 code driving.

The sensor model is deliberately imperfect - limited range, limited field of
view, occlusion behind obstacles, and noise - so the planner must cope with a
partial, changing map rather than being handed the answer.

Metrics follow the off-road navigation literature:

    SR   success rate - reached the goal without collision
    SPL  success weighted by inverse path length, the standard efficiency
         measure: ``SR * shortest / max(actual, shortest)``
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..layer3.costmap import Costmap, GridSpec, PersistentMap, LETHAL
from ..layer4.planner import AStarPlanner, MPPIPlanner, PurePursuit

__all__ = ["Obstacle", "World", "Vehicle", "EpisodeResult", "run_episode",
           "random_world"]


@dataclass
class Obstacle:
    """A circular hazard. ``negative=True`` marks a ditch rather than a rock."""
    x: float
    y: float
    radius: float
    negative: bool = False


@dataclass
class World:
    """Ground truth the vehicle cannot see directly."""
    width: float = 20.0
    depth: float = 20.0
    obstacles: list[Obstacle] = field(default_factory=list)
    goal: tuple[float, float] = (0.0, 16.0)

    def blocked(self, x: float, y: float, margin: float = 0.0) -> bool:
        """Is this point inside an obstacle (or outside the world)?"""
        if not (-self.width / 2 <= x <= self.width / 2) or not (0.0 <= y <= self.depth):
            return True
        return any((x - o.x) ** 2 + (y - o.y) ** 2 <= (o.radius + margin) ** 2
                   for o in self.obstacles)

    def clearance(self, x: float, y: float) -> float:
        """Distance to the nearest obstacle surface, metres."""
        if not self.obstacles:
            return float("inf")
        return min(float(np.hypot(x - o.x, y - o.y) - o.radius) for o in self.obstacles)


@dataclass
class Vehicle:
    """Unicycle with actuation noise."""
    x: float = 0.0
    y: float = 0.5
    theta: float = 0.0
    radius: float = 0.35
    noise: float = 0.02

    def step(self, v: float, omega: float, dt: float, rng) -> None:
        v = v * (1.0 + rng.normal(0.0, self.noise))
        omega = omega + rng.normal(0.0, self.noise)
        self.theta += omega * dt
        self.x += -np.sin(self.theta) * v * dt
        self.y += np.cos(self.theta) * v * dt

    @property
    def pose(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.theta)


def sense(world: World, veh: Vehicle, spec: GridSpec, max_range: float = 8.0,
          fov_deg: float = 90.0, rng=None, noise_m: float = 0.05):
    """Ray-cast what the vehicle can actually see right now.

    Each ray stops at the first obstacle it meets, so ground behind an obstacle
    stays unobserved rather than free.

    Returns:
        (xs, ys) hit coordinates in the vehicle frame (+Y forward, +X right).
    """
    half = np.deg2rad(fov_deg) / 2.0
    xs, ys = [], []
    for bearing in np.linspace(-half, half, 121):
        ang = veh.theta + bearing
        for r in np.arange(0.3, max_range, 0.08):
            wx = veh.x - np.sin(ang) * r
            wy = veh.y + np.cos(ang) * r
            if world.blocked(wx, wy):
                if rng is not None and noise_m:
                    r += rng.normal(0.0, noise_m)
                xs.append(r * np.sin(bearing))      # vehicle frame
                ys.append(r * np.cos(bearing))
                break
    return np.asarray(xs, float), np.asarray(ys, float)


@dataclass
class EpisodeResult:
    success: bool
    collided: bool
    timeout: bool
    path_length: float
    shortest: float
    steps: int
    trajectory: np.ndarray

    @property
    def spl(self) -> float:
        """Success weighted by path length."""
        if not self.success:
            return 0.0
        return self.shortest / max(self.path_length, self.shortest, 1e-6)


def run_episode(world: World, seed: int = 0, dt: float = 0.15,
                max_steps: int = 600, replan_every: int = 5,
                goal_tol: float = 0.9, controller: str = "pursuit",
                remember: bool = True) -> EpisodeResult:
    """Drive the real stack through one world until it arrives, hits, or times out.

    Args:
        controller: ``"mppi"`` uses the sampling local planner, which re-scores
            trajectories against the live costmap every step; ``"pursuit"`` uses
            pure pursuit alone, which follows the global path and cannot react
            between replans.
        remember: keep a world-frame ``PersistentMap`` of everything seen so
            far. With this off, each replan sees only the current field of view
            and the vehicle forgets obstacles it has already passed. Comparing
            both settings is a genuine ablation.
    """
    rng = np.random.default_rng(seed)
    veh = Vehicle()
    spec = GridSpec(width=90, height=90, resolution=0.12, origin_x=-5.4, origin_y=0.0)
    control = PurePursuit(lookahead=1.1, v_nominal=1.1, v_min=0.2)
    memory = PersistentMap(resolution=0.15, hits_to_occupy=3.0, decay=0.93)
    stuck = 0

    start = np.array([veh.x, veh.y])
    shortest = float(np.linalg.norm(np.array(world.goal) - start))
    traj = [start.copy()]
    length = 0.0
    path = np.empty((0, 2))

    for step in range(max_steps):
        if np.hypot(world.goal[0] - veh.x, world.goal[1] - veh.y) <= goal_tol:
            return EpisodeResult(True, False, False, length, shortest, step,
                                 np.array(traj))
        if world.blocked(veh.x, veh.y, margin=veh.radius * 0.6):
            return EpisodeResult(False, True, False, length, shortest, step,
                                 np.array(traj))

        if step % replan_every == 0 or path.size == 0:
            hx, hy = sense(world, veh, spec, rng=rng)
            if remember:
                memory.integrate(hx, hy, veh.pose)
                cm = memory.to_costmap(veh.pose, spec, veh.radius, veh.radius + 0.5)
            else:
                cm = Costmap(spec, robot_radius=veh.radius,
                             inflation_radius=veh.radius + 0.5)
                if hx.size:
                    cm.mark(hx, hy, LETHAL)
                cm.inflate()
            # goal expressed in the vehicle frame
            dx, dy = world.goal[0] - veh.x, world.goal[1] - veh.y
            c, s = np.cos(-veh.theta), np.sin(-veh.theta)
            gx, gy = c * dx - s * dy, s * dx + c * dy
            gy = min(gy, spec.origin_y + spec.height * spec.resolution - 0.4)
            gx = float(np.clip(gx, spec.origin_x + 0.4,
                               spec.origin_x + spec.width * spec.resolution - 0.4))
            local = AStarPlanner(cm).plan((0.0, 0.3), (gx, max(gy, 0.5)))
            if local.size == 0:
                # Recovery, following the Nav2 pattern: a plan failing usually
                # means the goal is outside a narrow field of view, not that the
                # world is blocked. Rotating widens what we can see; reversing
                # blindly drives into ground we have never observed.
                stuck += 1
                veh.step(0.0, 0.9 if (stuck // 4) % 2 == 0 else -0.9, dt, rng)
                if stuck > 40:
                    return EpisodeResult(False, False, True, length, shortest,
                                         step, np.array(traj))
                continue
            stuck = 0
            # local (vehicle) frame -> world.  Vehicle frame: +Y forward,
            # +X right, rotated by theta about Z.
            c_t, s_t = np.cos(veh.theta), np.sin(veh.theta)
            lx, ly = local[:, 0], local[:, 1]
            path = np.stack([veh.x + lx * c_t - ly * s_t,
                             veh.y + lx * s_t + ly * c_t], axis=-1)

        if controller == "mppi":
            # MPPI works in the vehicle frame against the live costmap.
            c_i, s_i = np.cos(-veh.theta), np.sin(-veh.theta)
            d = path - np.array([veh.x, veh.y])
            ref = np.stack([d[:, 0] * c_i - d[:, 1] * s_i,
                            d[:, 0] * s_i + d[:, 1] * c_i], axis=-1)
            cmd, _ = MPPIPlanner(cm, horizon=16, dt=dt, samples=260,
                                 v_max=1.1, seed=seed + step).plan((0.0, 0.3, 0.0), ref)
        else:
                cmd = control.step(veh.pose, path, costmap=cm)
        if cmd.linear <= 0.0 and abs(cmd.angular) <= 0.0:
            path = np.empty((0, 2))
            continue
        before = np.array([veh.x, veh.y])
        veh.step(cmd.linear, cmd.angular, dt, rng)
        length += float(np.linalg.norm(np.array([veh.x, veh.y]) - before))
        traj.append(np.array([veh.x, veh.y]))

    return EpisodeResult(False, False, True, length, shortest, max_steps,
                         np.array(traj))


def random_world(seed: int = 0, n_obstacles: int = 9,
                 difficulty: str = "medium") -> World:
    """Generate a world with a guaranteed-clear start and goal.

    ``difficulty`` scales obstacle count and size: easy / medium / hard.
    """
    scale = {"easy": (0.6, 0.8), "medium": (1.0, 1.0), "hard": (1.5, 1.25)}[difficulty]
    rng = np.random.default_rng(seed)
    goal = (float(rng.uniform(-3.0, 3.0)), 16.0)
    obs: list[Obstacle] = []
    tries = 0
    while len(obs) < int(n_obstacles * scale[0]) and tries < 500:
        tries += 1
        x = float(rng.uniform(-6.0, 6.0))
        y = float(rng.uniform(2.5, 14.5))
        r = float(rng.uniform(0.4, 1.1) * scale[1])
        if np.hypot(x, y - 0.5) < r + 1.5:              # keep the start clear
            continue
        if np.hypot(x - goal[0], y - goal[1]) < r + 1.5:  # keep the goal clear
            continue
        obs.append(Obstacle(x, y, r, negative=bool(rng.random() < 0.25)))
    return World(obstacles=obs, goal=goal)
