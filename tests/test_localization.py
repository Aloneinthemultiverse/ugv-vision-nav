import numpy as np
import pytest

from ugvnav.layer1.localization import (LoopClosureDetector, PoseFilter,
                                        PoseGraph, wrap_angle)


# ------------------------------------------------------------------ helpers
def drive_square(side=4.0, step=0.25):
    """Ground-truth trajectory: a closed square, returning exactly to start."""
    poses, x, y, th = [], 0.0, 0.0, 0.0
    for _ in range(4):
        for _ in range(int(side / step)):
            x += -np.sin(th) * step
            y += np.cos(th) * step
            poses.append((x, y, th))
        th = float(wrap_angle(th + np.pi / 2))
        poses.append((x, y, th))
    return np.array(poses)


# ----------------------------------------------------------------- utility
@pytest.mark.parametrize("a,expected", [
    (0.0, 0.0), (-np.pi, -np.pi), (-np.pi + 0.1, -np.pi + 0.1),
    (np.pi, -np.pi),          # interval is [-pi, pi): +pi wraps to -pi
    (3 * np.pi, -np.pi), (2 * np.pi, 0.0), (np.pi / 2, np.pi / 2),
])
def test_wrap_angle(a, expected):
    assert float(wrap_angle(a)) == pytest.approx(expected, abs=1e-9)


def test_wrap_angle_is_vectorised():
    out = wrap_angle(np.array([0.0, 3 * np.pi, -3 * np.pi]))
    assert out.shape == (3,)
    assert np.all(np.abs(out) <= np.pi + 1e-9)


# ------------------------------------------------------------------- EKF
def test_filter_starts_at_origin():
    assert np.allclose(PoseFilter().pose, 0.0)


def test_predict_drives_forward_along_y():
    """At theta = 0 the vehicle faces +Y, matching the costmap convention."""
    f = PoseFilter()
    for _ in range(10):
        f.predict(v=1.0, omega=0.0, dt=0.1)
    assert f.pose[1] == pytest.approx(1.0, abs=1e-6)
    assert abs(f.pose[0]) < 1e-9


def test_predict_turns_in_place():
    f = PoseFilter()
    f.predict(v=0.0, omega=np.pi / 2, dt=1.0)
    assert f.pose[2] == pytest.approx(np.pi / 2, abs=1e-6)
    assert np.allclose(f.pose[:2], 0.0)


def test_quarter_circle_matches_analytic_arc():
    """v = 1, omega = pi/2 for 1 s must trace a quarter circle of radius 2/pi."""
    f = PoseFilter(q=(1e-6, 1e-6, 1e-6))
    n, dt = 2000, 1.0 / 2000
    for _ in range(n):
        f.predict(v=1.0, omega=np.pi / 2, dt=dt)
    r = 1.0 / (np.pi / 2)
    assert f.pose[0] == pytest.approx(-r, abs=0.01)
    assert f.pose[1] == pytest.approx(r, abs=0.01)
    assert f.pose[2] == pytest.approx(np.pi / 2, abs=1e-3)


def test_covariance_grows_without_measurements():
    f = PoseFilter()
    start = np.trace(f.P)
    for _ in range(50):
        f.predict(1.0, 0.0, 0.1)
    assert np.trace(f.P) > start


def test_heading_update_reduces_heading_covariance():
    f = PoseFilter()
    for _ in range(30):
        f.predict(1.0, 0.1, 0.1)
    before = f.P[2, 2]
    f.update_heading(f.pose[2])
    assert f.P[2, 2] < before


def test_heading_update_pulls_toward_measurement():
    f = PoseFilter()
    f.predict(0.0, 1.0, 1.0)              # heading now 1.0 rad
    f.P[2, 2] = 1.0                       # very unsure about heading
    f.update_heading(0.0)
    assert abs(f.pose[2]) < 1.0


def test_visual_odometry_update_reduces_covariance():
    f = PoseFilter()
    for _ in range(20):
        f.predict(1.0, 0.0, 0.1)
    before = np.trace(f.P)
    f.update_visual_odometry(0.0, 0.1, 0.0)
    assert np.trace(f.P) < before


def test_imu_heading_corrects_wheel_slip():
    """Wheels over-report the turn; the IMU must pull heading back toward truth.

    The filter carries no gyro-bias state, so against a *systematic* input bias
    it settles at a steady-state lag rather than converging exactly. That is
    expected EKF behaviour and is documented as a known limitation; what the
    fusion must deliver is a large reduction, not perfection.
    """
    truth = np.pi / 4
    slipping = PoseFilter()
    fused = PoseFilter()
    for _ in range(40):
        slipping.predict(v=1.0, omega=(truth / 4) * 1.6, dt=0.1)   # 60% too fast
        fused.predict(v=1.0, omega=(truth / 4) * 1.6, dt=0.1)
        fused.update_heading(truth)

    err_slip = abs(float(wrap_angle(slipping.pose[2] - truth)))
    err_fused = abs(float(wrap_angle(fused.pose[2] - truth)))

    assert err_fused < err_slip
    assert err_fused < 0.5 * err_slip        # at least halves the heading error


def test_trusting_the_imu_more_shrinks_the_steady_state_lag():
    """Lowering IMU measurement noise tightens the fused heading, as it should."""
    truth = np.pi / 4
    loose = PoseFilter(r_yaw=0.2)
    tight = PoseFilter(r_yaw=0.005)
    for _ in range(40):
        for f in (loose, tight):
            f.predict(v=1.0, omega=(truth / 4) * 1.6, dt=0.1)
            f.update_heading(truth)
    assert abs(float(wrap_angle(tight.pose[2] - truth))) <            abs(float(wrap_angle(loose.pose[2] - truth)))


# ---------------------------------------------------------- loop closure
def test_loop_detector_ignores_recent_frames():
    d = LoopClosureDetector(min_gap=5)
    sig = np.ones(256, np.float32) / 16.0
    for _ in range(4):
        d.add(sig)
    assert d.detect(sig) is None          # not enough history yet


def test_loop_detector_recognises_a_revisit():
    rng = np.random.default_rng(0)
    d = LoopClosureDetector(min_gap=3, threshold=0.9)
    place = rng.random(256).astype(np.float32)
    place /= np.linalg.norm(place)

    d.add(place)                          # index 0: the place we will return to
    for _ in range(6):
        other = rng.random(256).astype(np.float32)
        d.add(other / np.linalg.norm(other))

    hit = d.detect(place)
    assert hit is not None
    assert hit[0] == 0
    assert hit[1] > 0.9


def test_loop_detector_rejects_unseen_place():
    rng = np.random.default_rng(1)
    d = LoopClosureDetector(min_gap=2, threshold=0.99)
    for _ in range(8):
        s = rng.random(256).astype(np.float32)
        d.add(s / np.linalg.norm(s))
    fresh = rng.random(256).astype(np.float32)
    assert d.detect(fresh / np.linalg.norm(fresh)) is None


def test_signature_is_deterministic_and_normalised():
    rng = np.random.default_rng(3)
    img = (rng.random((120, 160)) * 255).astype(np.uint8)
    a = LoopClosureDetector.signature(img)
    b = LoopClosureDetector.signature(img)
    assert a.shape == (256,)
    assert np.allclose(a, b)
    assert np.linalg.norm(a) == pytest.approx(1.0, abs=1e-5)


def test_signature_handles_featureless_image():
    sig = LoopClosureDetector.signature(np.zeros((80, 80), np.uint8))
    assert sig.shape == (256,) and not np.isnan(sig).any()


def test_same_scene_scores_higher_than_different_scene():
    import cv2
    rng = np.random.default_rng(5)
    a = cv2.GaussianBlur((rng.random((160, 200)) * 255).astype(np.uint8), (0, 0), 1.0)
    b = cv2.GaussianBlur((rng.random((160, 200)) * 255).astype(np.uint8), (0, 0), 1.0)
    a_shift = np.roll(a, 4, axis=1)       # same place, slightly moved

    sa, sb, sa2 = (LoopClosureDetector.signature(x) for x in (a, b, a_shift))
    assert float(sa @ sa2) > float(sa @ sb)


# ------------------------------------------------------------ pose graph
def test_relative_pose_of_identical_poses_is_zero():
    assert np.allclose(PoseGraph.relative([1.0, 2.0, 0.5], [1.0, 2.0, 0.5]), 0.0)


def test_relative_pose_is_expressed_in_the_first_frame():
    rel = PoseGraph.relative([0.0, 0.0, np.pi / 2], [1.0, 0.0, np.pi / 2])
    assert rel[0] == pytest.approx(0.0, abs=1e-9)
    assert rel[1] == pytest.approx(-1.0, abs=1e-9)
    assert rel[2] == pytest.approx(0.0, abs=1e-9)


def test_graph_with_consistent_edges_has_zero_residual():
    g = PoseGraph()
    truth = drive_square()
    for p in truth:
        g.add_node(p)
    for i in range(len(truth) - 1):
        g.add_edge(i, i + 1, PoseGraph.relative(truth[i], truth[i + 1]))
    assert g.residual() < 1e-12


def test_optimise_is_a_noop_on_a_consistent_graph():
    g = PoseGraph()
    truth = drive_square()
    for p in truth:
        g.add_node(p)
    for i in range(len(truth) - 1):
        g.add_edge(i, i + 1, PoseGraph.relative(truth[i], truth[i + 1]))
    before = np.array(g.nodes)
    g.optimise(iterations=5)
    assert np.allclose(np.array(g.nodes), before, atol=1e-6)


def test_loop_closure_reduces_drift_on_a_square():
    """The headline localization test: drive a closed loop with a biased yaw
    estimate, then close the loop and verify the trajectory snaps back."""
    truth = drive_square()
    rng = np.random.default_rng(7)

    g = PoseGraph()
    drifted = [np.array([0.0, 0.0, 0.0])]
    g.add_node(drifted[0])
    for i in range(len(truth) - 1):
        rel = PoseGraph.relative(truth[i], truth[i + 1]).copy()
        rel[2] += 0.02                            # systematic yaw bias
        rel[:2] += rng.normal(0, 0.004, 2)
        prev = drifted[-1]
        c, s = np.cos(prev[2]), np.sin(prev[2])
        nxt = np.array([prev[0] + c * rel[0] - s * rel[1],
                        prev[1] + s * rel[0] + c * rel[1],
                        float(wrap_angle(prev[2] + rel[2]))])
        drifted.append(nxt)
        g.add_node(nxt)
        g.add_edge(i, i + 1, rel)

    err_before = float(np.linalg.norm(drifted[-1][:2] - truth[-1][:2]))
    assert err_before > 0.3, "test needs meaningful drift to be meaningful"

    # the loop: last keyframe is the same place as the first
    g.add_edge(len(truth) - 1, 0, PoseGraph.relative(truth[-1], truth[0]), weight=50.0)
    g.optimise(iterations=40)

    err_after = float(np.linalg.norm(g.nodes[-1][:2] - g.nodes[0][:2]
                                     - (truth[-1][:2] - truth[0][:2])))
    assert err_after < err_before * 0.25


def test_optimise_handles_empty_and_single_node_graphs():
    assert PoseGraph().optimise() == 0.0
    g = PoseGraph(); g.add_node([0.0, 0.0, 0.0])
    assert g.optimise() == 0.0


def test_first_node_stays_anchored():
    g = PoseGraph()
    for p in ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]):
        g.add_node(p)
    g.add_edge(0, 1, [0.0, 1.5, 0.0])
    g.add_edge(1, 2, [0.0, 1.5, 0.0])
    g.optimise(iterations=20)
    assert np.allclose(g.nodes[0], [0.0, 0.0, 0.0], atol=1e-6)
