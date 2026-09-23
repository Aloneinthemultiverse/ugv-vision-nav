import numpy as np
import pytest

from ugvnav.layer3.costmap import GridSpec, INSCRIBED, LETHAL, PersistentMap
from ugvnav.sim.world import Obstacle, Vehicle, World, random_world, run_episode, sense


# --------------------------------------------------------------- persistent map
def test_persistent_map_rejects_bad_resolution():
    with pytest.raises(ValueError):
        PersistentMap(resolution=0.0)


def test_single_observation_is_not_enough_evidence():
    """One noisy hit must not permanently occupy a cell - that silts up the map."""
    m = PersistentMap(resolution=0.2, hits_to_occupy=3.0, decay=1.0)
    m.integrate([0.0], [2.0], (0.0, 0.0, 0.0))
    assert len(m) == 0
    for _ in range(2):
        m.integrate([0.0], [2.0], (0.0, 0.0, 0.0))
    assert len(m) == 1


def test_evidence_decays_so_stale_hits_fade():
    # Evidence asymptotes at 1/(1 - decay), so decay must be high enough for
    # hits_to_occupy to be reachable at all: 0.8 caps at 5.0, above the 3.0
    # threshold. See test_evidence_saturates_below_an_unreachable_threshold.
    m = PersistentMap(resolution=0.2, hits_to_occupy=3.0, decay=0.8)
    for _ in range(6):
        m.integrate([0.0], [2.0], (0.0, 0.0, 0.0))
    assert len(m) == 1
    for _ in range(25):                       # stop seeing it
        m.integrate([], [], (0.0, 0.0, 0.0))
    assert len(m) == 0


def test_evidence_saturates_below_an_unreachable_threshold():
    """decay = 0.5 caps evidence at 1/(1-0.5) = 2.0, so a threshold of 3 can
    never be met no matter how often the cell is observed. Documented so the
    two parameters are not tuned independently."""
    m = PersistentMap(resolution=0.2, hits_to_occupy=3.0, decay=0.5)
    for _ in range(200):
        m.integrate([0.0], [2.0], (0.0, 0.0, 0.0))
    assert len(m) == 0


def test_observations_are_stored_in_the_world_frame():
    """A hit 2 m ahead while facing -X must land to the left of the origin."""
    m = PersistentMap(resolution=0.1, hits_to_occupy=1.0, decay=1.0)
    m.integrate([0.0], [2.0], (0.0, 0.0, np.pi / 2))
    pts = m.occupied_points()
    assert pts.shape == (1, 2)
    assert pts[0, 0] == pytest.approx(-2.0, abs=0.15)
    assert abs(pts[0, 1]) < 0.15


def test_same_obstacle_seen_from_two_poses_lands_in_one_place():
    """The whole point of a world frame: observations must agree across motion."""
    m = PersistentMap(resolution=0.2, hits_to_occupy=1.0, decay=1.0)
    m.integrate([0.0], [4.0], (0.0, 0.0, 0.0))      # obstacle at world (0, 4)
    m.integrate([0.0], [3.0], (0.0, 1.0, 0.0))      # moved 1 m closer
    pts = m.occupied_points()
    assert pts.shape[0] == 1                         # one obstacle, not two
    assert np.allclose(pts[0], [0.0, 4.0], atol=0.3)


def test_map_renders_a_costmap_in_the_vehicle_frame():
    m = PersistentMap(resolution=0.15, hits_to_occupy=1.0, decay=1.0)
    m.integrate([0.0], [3.0], (0.0, 0.0, 0.0))
    spec = GridSpec(60, 60, 0.15, -4.5, 0.0)
    cm = m.to_costmap((0.0, 1.0, 0.0), spec)         # vehicle advanced 1 m
    assert cm.cost_at(0.0, 2.0) >= LETHAL            # now 2 m ahead
    assert cm.cost_at(0.0, 5.0) < INSCRIBED


def test_memory_outlives_the_field_of_view():
    """Regression for the real bug: a rebuilt-every-frame map forgets what it passed."""
    m = PersistentMap(resolution=0.2, hits_to_occupy=1.0, decay=1.0)
    m.integrate([1.0], [1.0], (0.0, 0.0, 0.0))
    m.integrate([], [], (0.0, 3.0, 0.0))             # obstacle now behind us
    spec = GridSpec(60, 60, 0.15, -4.5, 0.0)
    assert len(m) == 1
    assert m.occupied_points().shape[0] == 1


# -------------------------------------------------------------------- world
def test_world_blocks_inside_obstacle_and_outside_bounds():
    w = World(obstacles=[Obstacle(0.0, 5.0, 1.0)])
    assert w.blocked(0.0, 5.0)
    assert not w.blocked(0.0, 8.0)
    assert w.blocked(99.0, 5.0)
    assert w.blocked(0.0, -1.0)


def test_clearance_is_distance_to_surface():
    w = World(obstacles=[Obstacle(0.0, 5.0, 1.0)])
    assert w.clearance(0.0, 8.0) == pytest.approx(2.0, abs=1e-6)
    assert np.isinf(World().clearance(0.0, 0.0))


def test_random_world_keeps_start_and_goal_clear():
    for seed in range(12):
        for d in ("easy", "medium", "hard"):
            w = random_world(seed, difficulty=d)
            assert not w.blocked(0.0, 0.5, margin=0.35)
            assert not w.blocked(*w.goal, margin=0.35)


def test_harder_worlds_have_more_obstacle_area():
    easy = random_world(3, difficulty="easy")
    hard = random_world(3, difficulty="hard")
    area = lambda w: sum(np.pi * o.radius ** 2 for o in w.obstacles)
    assert area(hard) > area(easy)


# ------------------------------------------------------------------ sensing
def test_sense_returns_vehicle_frame_hits():
    w = World(obstacles=[Obstacle(0.0, 4.0, 0.8)])
    xs, ys = sense(w, Vehicle(), GridSpec(90, 90, 0.12, -5.4, 0.0))
    assert xs.size > 0
    assert ys.mean() == pytest.approx(3.2, abs=0.6)    # front surface of the disc
    assert abs(xs.mean()) < 0.5


def test_sense_sees_nothing_in_an_empty_world():
    xs, ys = sense(World(), Vehicle(), GridSpec(90, 90, 0.12, -5.4, 0.0))
    assert xs.size == 0


def test_sense_respects_occlusion():
    """A far obstacle directly behind a near one must not be reported."""
    near = World(obstacles=[Obstacle(0.0, 3.0, 0.8), Obstacle(0.0, 6.0, 0.8)])
    xs, ys = sense(near, Vehicle(), GridSpec(90, 90, 0.12, -5.4, 0.0))
    straight = ys[np.abs(xs) < 0.2]
    assert straight.size > 0
    assert straight.max() < 5.0                        # nothing from the far disc


def test_sense_respects_limited_range():
    w = World(obstacles=[Obstacle(0.0, 12.0, 1.0)])
    xs, _ = sense(w, Vehicle(), GridSpec(90, 90, 0.12, -5.4, 0.0), max_range=6.0)
    assert xs.size == 0


# ----------------------------------------------------------------- episodes
def test_empty_world_is_reached_efficiently():
    r = run_episode(World(goal=(0.0, 8.0)), seed=0)
    assert r.success and not r.collided
    assert r.spl > 0.8


def test_episode_detects_collision():
    """A wall with no gap must end in a collision or a timeout, never success."""
    wall = [Obstacle(x, 5.0, 0.6) for x in np.arange(-9.0, 9.1, 0.5)]
    r = run_episode(World(obstacles=wall, goal=(0.0, 12.0)), seed=1, max_steps=120)
    assert not r.success


def test_episode_routes_around_a_single_obstacle():
    w = World(obstacles=[Obstacle(0.0, 5.0, 1.0)], goal=(0.0, 10.0))
    r = run_episode(w, seed=2)
    assert r.success
    assert r.path_length > r.shortest          # had to deviate


def test_spl_is_zero_on_failure_and_bounded_on_success():
    wall = [Obstacle(x, 5.0, 0.6) for x in np.arange(-9.0, 9.1, 0.5)]
    fail = run_episode(World(obstacles=wall, goal=(0.0, 12.0)), seed=3, max_steps=100)
    assert fail.spl == 0.0

    ok = run_episode(World(goal=(0.0, 6.0)), seed=4)
    assert 0.0 < ok.spl <= 1.0


def test_trajectory_is_recorded():
    r = run_episode(World(goal=(0.0, 6.0)), seed=5)
    assert r.trajectory.ndim == 2 and r.trajectory.shape[1] == 2
    assert len(r.trajectory) > 5


def test_map_memory_helps_on_cluttered_worlds():
    """Ablation, asserted: remembering what left the field of view must not hurt."""
    with_mem = [run_episode(random_world(s, difficulty="medium"), seed=s, remember=True)
                for s in range(8)]
    without = [run_episode(random_world(s, difficulty="medium"), seed=s, remember=False)
               for s in range(8)]
    assert sum(r.success for r in with_mem) >= sum(r.success for r in without)
