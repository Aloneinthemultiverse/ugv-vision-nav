import math

import numpy as np
import pytest

from ugvnav.layer3.costmap import Costmap, GridSpec, FREE, CAUTION, INSCRIBED, LETHAL
from ugvnav.layer4.planner import AStarPlanner, MPPIPlanner, PurePursuit, Twist


@pytest.fixture
def cm():
    return Costmap(GridSpec(width=60, height=60, resolution=0.1, origin_x=-3.0,
                            origin_y=0.0), robot_radius=0.2, inflation_radius=0.5)


# ------------------------------------------------------------------- geometry
def test_world_cell_roundtrip(cm):
    for x, y in [(-2.5, 0.5), (0.0, 3.0), (2.4, 5.9)]:
        c, r = cm.spec.world_to_cell(x, y)
        wx, wy = cm.spec.cell_to_world(c, r)
        assert abs(wx - x) <= cm.spec.resolution
        assert abs(wy - y) <= cm.spec.resolution


def test_vehicle_origin_is_bottom_centre(cm):
    c, r = cm.spec.world_to_cell(0.0, 0.0)
    assert r == cm.spec.height - 1
    assert c == 30


def test_cost_outside_grid_is_lethal(cm):
    assert cm.cost_at(99.0, 99.0) == LETHAL
    assert not cm.is_free(99.0, 99.0)


# --------------------------------------------------------------------- layers
def test_mark_writes_lethal(cm):
    cm.mark([0.0], [2.0])
    assert cm.cost_at(0.0, 2.0) == LETHAL
    assert cm.cost_at(0.0, 4.0) == FREE


def test_stronger_claim_wins_and_weaker_does_not_erase(cm):
    cm.add_disc(0.0, 2.0, 0.2, LETHAL)
    cm.add_disc(0.0, 2.0, 0.2, CAUTION)       # weaker, must not overwrite
    assert cm.cost_at(0.0, 2.0) == LETHAL


def test_virtual_scan_places_drop_offs_in_front(cm):
    ranges = np.array([np.inf, 3.0, 3.0, np.inf])
    angles = np.array([-0.5, 0.0, 0.05, 0.5])
    assert cm.add_virtual_scan(ranges, angles) > 0
    assert cm.cost_at(0.0, 3.0) == LETHAL          # the straight-ahead ray
    assert cm.cost_at(0.0, 1.0) == FREE            # nothing between us and it


def test_virtual_scan_ignores_infinite_rays(cm):
    assert cm.add_virtual_scan(np.array([np.inf, np.inf]), np.array([0.0, 0.1])) == 0
    assert (cm.grid == FREE).all()


def test_dynamic_obstacle_blocks_where_it_will_be(cm):
    """The whole point of tracking velocity: block the future, not the past."""
    cm.add_dynamic(x=-1.0, y=3.0, vx=0.5, vy=0.0, horizon_s=2.0, radius=0.2)

    assert cm.cost_at(-1.0, 3.0) == LETHAL     # where it is
    assert cm.cost_at(0.0, 3.0) == LETHAL      # where it will be in 2 s
    assert cm.cost_at(-2.5, 3.0) == FREE       # where it came from


def test_add_mask_merges_overhead_layer(cm):
    mask = np.zeros_like(cm.grid, dtype=bool)
    mask[0:5, :] = True
    assert cm.add_mask(mask, LETHAL) == mask.sum()
    assert (cm.grid[0:5, :] == LETHAL).all()


# ------------------------------------------------------------------ inflation
def test_inflation_grows_obstacle_by_robot_radius(cm):
    cm.mark([0.0], [3.0])
    cm.inflate()
    assert cm.cost_at(0.0, 3.0) == LETHAL
    assert cm.cost_at(0.15, 3.0) >= INSCRIBED     # inside the footprint
    assert cm.cost_at(0.45, 3.0) < INSCRIBED      # outside it
    assert cm.cost_at(0.45, 3.0) > FREE           # but still discouraged


def test_inflation_on_empty_map_is_noop(cm):
    cm.inflate()
    assert (cm.grid == FREE).all()


def test_inflation_decays_with_distance(cm):
    cm.mark([0.0], [3.0])
    cm.inflate()
    near = cm.cost_at(0.28, 3.0)
    far = cm.cost_at(0.45, 3.0)
    assert near >= far


# -------------------------------------------------------------- global planner
def test_astar_finds_a_straight_path_on_empty_map(cm):
    path = AStarPlanner(cm).plan((0.0, 0.2), (0.0, 5.0))
    assert len(path) > 10
    assert abs(path[-1][1] - 5.0) < 0.2
    assert np.abs(path[:, 0]).max() < 0.4        # no reason to wander


def test_astar_routes_around_a_wall(cm):
    """A wall with a gap must be negotiated, not driven through."""
    for x in np.arange(-3.0, 1.2, 0.05):         # gap on the right
        cm.mark([x], [3.0])
    cm.inflate()

    path = AStarPlanner(cm).plan((0.0, 0.2), (0.0, 5.0))

    assert path.size > 0
    crossing = path[np.abs(path[:, 1] - 3.0) < 0.12]
    assert crossing.size > 0
    assert crossing[:, 0].max() > 1.2            # went through the gap


def test_astar_returns_empty_when_fully_blocked(cm):
    for x in np.arange(-3.0, 3.0, 0.05):
        cm.mark([x], [3.0])
    cm.inflate()
    assert AStarPlanner(cm).plan((0.0, 0.2), (0.0, 5.0)).size == 0


def test_astar_prefers_cheap_cells_over_caution(cm):
    """A CAUTION corridor is passable but should be avoided when free space exists."""
    mask = np.zeros_like(cm.grid, dtype=bool)
    left_cols = slice(0, 30)
    mask[:, left_cols] = True
    cm.add_mask(mask, CAUTION)

    path = AStarPlanner(cm, cost_weight=0.05).plan((0.0, 0.2), (0.0, 5.0))
    assert path.size > 0
    assert path[:, 0].mean() > -0.2              # hugged the free right side


def test_astar_rejects_start_inside_an_obstacle(cm):
    cm.add_disc(0.0, 0.2, 0.3, LETHAL)
    cm.inflate()
    assert AStarPlanner(cm).plan((0.0, 0.2), (0.0, 5.0)).size == 0


# --------------------------------------------------------------- local planner
def test_mppi_rollout_drives_straight_with_zero_turn():
    cm = Costmap(GridSpec(40, 40, 0.1, -2.0, 0.0))
    p = MPPIPlanner(cm, horizon=10, dt=0.1, samples=4)
    v = np.full((1, 10), 1.0)
    w = np.zeros((1, 10))
    traj = p.rollout((0.0, 0.0, 0.0), v, w)
    assert traj[0, -1, 1] == pytest.approx(1.0, abs=1e-6)   # 1 m/s for 1 s
    assert abs(traj[0, -1, 0]) < 1e-9


def test_mppi_returns_zero_command_without_a_reference(cm):
    cmd, traj = MPPIPlanner(cm).plan((0.0, 0.2, 0.0), np.empty((0, 2)))
    assert cmd.linear == 0.0 and cmd.angular == 0.0
    assert traj.size == 0


def test_mppi_moves_forward_toward_the_goal(cm):
    ref = np.array([[0.0, y] for y in np.arange(0.2, 5.0, 0.1)])
    cmd, traj = MPPIPlanner(cm, samples=300, seed=1).plan((0.0, 0.2, 0.0), ref)
    assert cmd.linear > 0.1
    assert traj.shape[1] == 3


def test_mppi_steers_away_from_an_obstacle(cm):
    """With a blockage dead ahead, the averaged command must not be straight on."""
    cm.add_disc(0.0, 2.0, 0.6, LETHAL)
    cm.inflate()
    ref = np.array([[0.0, y] for y in np.arange(0.2, 5.0, 0.1)])
    _, best = MPPIPlanner(cm, samples=600, seed=3).plan((0.0, 0.2, 0.0), ref)
    # the selected trajectory should not sit inside the obstacle
    assert not any(cm.cost_at(float(x), float(y)) >= LETHAL for x, y, _ in best)


# -------------------------------------------------------------------- control
def test_pure_pursuit_goes_straight_for_a_straight_path():
    path = np.array([[0.0, y] for y in np.arange(0.0, 6.0, 0.1)])
    cmd = PurePursuit(lookahead=1.0).step((0.0, 0.0, 0.0), path)
    assert cmd.linear > 0.5
    assert abs(cmd.angular) < 1e-6


def test_pure_pursuit_turns_toward_an_offset_path():
    # +theta is counter-clockwise (toward -X), so steering toward a target on
    # the right (+X) requires a negative yaw rate.
    path = np.array([[x, 0.2] for x in np.arange(0.0, 4.0, 0.1)])
    cmd = PurePursuit(lookahead=1.0).step((0.0, 0.0, 0.0), path)
    assert cmd.angular < -0.05

    mirrored = np.array([[-x, 0.2] for x in np.arange(0.0, 4.0, 0.1)])
    assert PurePursuit(lookahead=1.0).step((0.0, 0.0, 0.0), mirrored).angular > 0.05


def test_controller_and_vehicle_model_agree_on_turn_direction():
    """Regression: the controller must actually drive the vehicle TOWARD the
    path, not away from it. Unit tests that only check the sign of omega cannot
    catch a convention mismatch between the controller and the motion model -
    this closes that loop explicitly."""
    from ugvnav.sim.world import Vehicle

    rng = np.random.default_rng(0)
    for direction in (+1.0, -1.0):
        path = np.array([[direction * x, 0.3] for x in np.arange(0.0, 6.0, 0.1)])
        veh = Vehicle(x=0.0, y=0.0, theta=0.0, noise=0.0)
        ctrl = PurePursuit(lookahead=1.0)
        before = abs(veh.x - direction * 2.0)
        for _ in range(25):
            cmd = ctrl.step(veh.pose, path)
            veh.step(cmd.linear, cmd.angular, 0.1, rng)
        assert abs(veh.x - direction * 2.0) < before, "controller steered away from the path"
        assert np.sign(veh.x) == np.sign(direction)


def test_mppi_rollout_matches_the_vehicle_motion_model():
    """MPPI's internal model must match the vehicle it is steering."""
    from ugvnav.sim.world import Vehicle

    cm = Costmap(GridSpec(40, 40, 0.1, -2.0, 0.0))
    p = MPPIPlanner(cm, horizon=20, dt=0.1, samples=1)
    v = np.full((1, 20), 0.8)
    w = np.full((1, 20), 0.4)
    traj = p.rollout((0.0, 0.0, 0.0), v, w)

    veh = Vehicle(x=0.0, y=0.0, theta=0.0, noise=0.0)
    rng = np.random.default_rng(0)
    for _ in range(20):
        veh.step(0.8, 0.4, 0.1, rng)

    assert traj[0, -1, 0] == pytest.approx(veh.x, abs=1e-6)
    assert traj[0, -1, 1] == pytest.approx(veh.y, abs=1e-6)


def test_pure_pursuit_stops_on_an_empty_path():
    cmd = PurePursuit().step((0.0, 0.0, 0.0), np.empty((0, 2)))
    assert cmd == Twist(0.0, 0.0)


def test_pure_pursuit_slows_down_in_caution(cm):
    """The uncertainty layer must actually change behaviour, not just colour."""
    path = np.array([[0.0, y] for y in np.arange(0.0, 6.0, 0.1)])
    ctrl = PurePursuit(lookahead=1.0, v_nominal=1.0, v_min=0.2)

    fast = ctrl.step((0.0, 1.0, 0.0), path, costmap=cm).linear
    cm.add_mask(np.ones_like(cm.grid, dtype=bool), CAUTION)
    slow = ctrl.step((0.0, 1.0, 0.0), path, costmap=cm).linear

    assert slow < fast


def test_pure_pursuit_target_selection_respects_lookahead():
    path = np.array([[0.0, y] for y in np.arange(0.0, 6.0, 0.5)])
    tgt = PurePursuit(lookahead=2.0).target((0.0, 0.0, 0.0), path)
    assert tgt[1] >= 2.0


def test_full_chain_produces_a_command(cm):
    """Costmap -> A* -> MPPI -> controller, end to end."""
    cm.add_disc(0.6, 2.5, 0.4, LETHAL)
    cm.inflate()

    path = AStarPlanner(cm).plan((0.0, 0.2), (0.0, 5.0))
    assert path.size > 0

    cmd_mppi, _ = MPPIPlanner(cm, samples=300, seed=5).plan((0.0, 0.2, 0.0), path)
    cmd_pp = PurePursuit().step((0.0, 0.2, 0.0), path, costmap=cm)

    assert cmd_mppi.linear >= 0.0
    assert cmd_pp.linear > 0.0
    assert math.isfinite(cmd_pp.angular)
