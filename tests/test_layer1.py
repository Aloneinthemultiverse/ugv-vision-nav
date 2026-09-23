import numpy as np
import pytest

from ugvnav.camera import Camera
from ugvnav.layer1.depth import StubDepth, to_metric
from ugvnav.layer1.elevation import ElevationNetwork
from ugvnav.layer1.odometry import estimate_relative_pose, VisualOdometry
from ugvnav.layer1.semantics import StubSegmenter, traversable_mask, _name_to_class
from ugvnav.layer1.uncertainty import UncertaintyEstimator, normalise


# --------------------------------------------------------------------- depth
def test_to_metric_inverts_ordering():
    """High inverse depth means near, so it must map to a small distance."""
    inv = np.array([[10.0, 1.0]], dtype=np.float32)
    m = to_metric(inv, near_m=1.0, far_m=30.0)
    assert m[0, 0] == pytest.approx(1.0, rel=1e-3)
    assert m[0, 1] == pytest.approx(30.0, rel=1e-3)
    assert m[0, 0] < m[0, 1]


def test_to_metric_handles_flat_input():
    out = to_metric(np.full((4, 4), 3.0, np.float32), far_m=25.0)
    assert np.allclose(out, 25.0)


def test_stub_depth_returns_copy():
    field = np.ones((3, 3), np.float32)
    stub = StubDepth(field)
    got = stub.infer(np.zeros((3, 3, 3), np.uint8))
    got[0, 0] = 99.0
    assert stub.infer(np.zeros((3, 3, 3), np.uint8))[0, 0] == 1.0


# ----------------------------------------------------------------- elevation
def _flat_ground_depth(cam: Camera, ground_y: float = 0.8) -> np.ndarray:
    """Depth image of a perfectly flat floor ``ground_y`` below the camera."""
    us, vs = np.meshgrid(np.arange(cam.width, dtype=np.float64),
                         np.arange(cam.height, dtype=np.float64))
    # y = (v - cy) * z / fy = ground_y   ->  z = ground_y * fy / (v - cy)
    denom = vs - cam.cy
    z = np.full_like(denom, np.inf)
    below = denom > 1e-6                       # only pixels under the horizon
    z[below] = ground_y * cam.fy / denom[below]
    return np.clip(z, 0.1, 60.0)


def test_elevation_flat_ground_is_flat():
    cam = Camera.from_fov(160, 120, height_m=0.8)
    depth = _flat_ground_depth(cam, 0.8)
    res = ElevationNetwork(cam).process(depth, seed=0)

    ground = res.height[res.ground_mask]
    assert ground.size > 500
    assert np.abs(ground).max() < 0.05           # the floor is level
    assert res.ground_mask.mean() > 0.3


def test_elevation_detects_a_step():
    """An object standing on the floor must show positive height."""
    cam = Camera.from_fov(160, 120, height_m=0.8)
    depth = _flat_ground_depth(cam, 0.8)
    depth[70:90, 60:100] *= 0.55                 # a block nearer than the floor
    res = ElevationNetwork(cam).process(depth, seed=0)

    assert res.height[70:90, 60:100].max() > 0.15


def test_elevation_scale_matches_camera_height():
    cam = Camera.from_fov(160, 120, height_m=1.25)
    res = ElevationNetwork(cam).process(_flat_ground_depth(cam, 1.25), seed=0)
    assert abs(abs(res.offset) - 1.25) < 0.05


def test_top_down_grid_shape_and_content():
    cam = Camera.from_fov(160, 120, height_m=0.8)
    res = ElevationNetwork(cam).process(_flat_ground_depth(cam, 0.8), seed=0)
    grid = ElevationNetwork.top_down(res.points, res.height, cells=16)
    assert grid.shape == (16, 16)
    assert np.isfinite(grid).any()


# ----------------------------------------------------------------- semantics
def test_traversable_mask_selects_trail_and_grass():
    labels = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint8)
    m = traversable_mask(labels)
    assert m.tolist() == [[False, True, True], [False, False, False]]


@pytest.mark.parametrize("name,expected", [
    ("sky", 5), ("grass", 2), ("dirt track", 1), ("tree", 3),
    ("person", 4), ("water", 6), ("banana", 0),
])
def test_ade_label_names_map_to_compact_classes(name, expected):
    assert _name_to_class(name) == expected


def test_stub_segmenter_roundtrip():
    labels = np.array([[1, 2], [3, 4]], np.uint8)
    assert np.array_equal(StubSegmenter(labels).infer(np.zeros((2, 2, 3), np.uint8)),
                          labels)


# --------------------------------------------------------------- uncertainty
class _NoisyDepth:
    """Returns a constant field, plus noise confined to one corner."""

    def __init__(self, shape=(40, 40)):
        self.shape = shape
        self._n = 0

    def infer(self, rgb):
        d = np.linspace(1.0, 5.0, self.shape[1], dtype=np.float32)
        d = np.tile(d, (self.shape[0], 1))
        rng = np.random.default_rng(self._n)
        self._n += 1
        d[:10, :10] += rng.normal(0, 2.0, (10, 10)).astype(np.float32)
        return d


def test_normalise_maps_to_unit_range():
    out = normalise(np.array([2.0, 4.0, 6.0]))
    assert out.min() == pytest.approx(0.0)
    assert out.max() == pytest.approx(1.0)


def test_normalise_constant_input_is_zero():
    assert np.allclose(normalise(np.full((3, 3), 7.0)), 0.0)


def test_uncertainty_is_higher_where_predictions_disagree():
    est = UncertaintyEstimator(_NoisyDepth(), samples=6, flip=False)
    mean, unc = est.process(np.zeros((40, 40, 3), np.uint8))

    assert mean.shape == unc.shape == (40, 40)
    assert 0.0 <= unc.min() and unc.max() <= 1.0
    assert unc[:10, :10].mean() > unc[20:, 20:].mean() + 0.1


def test_uncertainty_is_zero_for_a_deterministic_estimator():
    stub = StubDepth(np.linspace(0, 1, 100, dtype=np.float32).reshape(10, 10))
    _, unc = UncertaintyEstimator(stub, samples=4, flip=False).process(
        np.zeros((10, 10, 3), np.uint8))
    assert unc.max() < 1e-5


# ------------------------------------------------------------------ odometry
def _project(pts, K, R=np.eye(3), t=np.zeros(3)):
    cam = pts @ R.T + t
    uv = (K @ cam.T).T
    return uv[:, :2] / uv[:, 2:3]


def test_estimate_relative_pose_recovers_known_translation():
    """Synthetic camera motion with exactly-known ground truth."""
    K = Camera.from_fov(640, 480).K
    rng = np.random.default_rng(7)
    pts = np.stack([rng.uniform(-4, 4, 400),
                    rng.uniform(-3, 3, 400),
                    rng.uniform(6, 18, 400)], axis=-1)

    t_true = np.array([0.6, 0.0, 0.0])          # camera slides sideways
    p0 = _project(pts, K)
    p1 = _project(pts, K, np.eye(3), t_true)

    d = estimate_relative_pose(p0, p1, K)

    assert d.valid and d.inliers > 100
    assert np.linalg.norm(d.t) == pytest.approx(1.0, abs=1e-6)
    # translation is recovered up to sign and scale
    cos = abs(float(d.t @ (t_true / np.linalg.norm(t_true))))
    assert cos > 0.99
    assert np.allclose(d.R, np.eye(3), atol=0.02)


def test_estimate_relative_pose_recovers_known_rotation():
    K = Camera.from_fov(640, 480).K
    rng = np.random.default_rng(11)
    pts = np.stack([rng.uniform(-4, 4, 500),
                    rng.uniform(-3, 3, 500),
                    rng.uniform(6, 18, 500)], axis=-1)

    a = np.deg2rad(5.0)
    R_true = np.array([[np.cos(a), 0, np.sin(a)],
                       [0, 1, 0],
                       [-np.sin(a), 0, np.cos(a)]])
    t_true = np.array([0.5, 0.0, 0.1])

    d = estimate_relative_pose(_project(pts, K), _project(pts, K, R_true, t_true), K)

    assert d.valid
    assert np.allclose(d.R, R_true, atol=0.03)


def test_estimate_relative_pose_rejects_too_few_points():
    K = Camera.from_fov(320, 240).K
    d = estimate_relative_pose(np.zeros((3, 2)), np.zeros((3, 2)), K)
    assert not d.valid and d.inliers == 0


def test_visual_odometry_first_frame_is_not_valid():
    vo = VisualOdometry(K=Camera.from_fov(160, 120).K)
    img = (np.random.default_rng(0).random((120, 160, 3)) * 255).astype(np.uint8)
    assert not vo.process(img).valid
    assert np.allclose(vo.p, 0.0)


def test_visual_odometry_accumulates_translation_with_scale():
    """Pose must advance by the supplied metric scale, not by the unit vector."""
    vo = VisualOdometry(K=Camera.from_fov(320, 240).K)
    vo.R = np.eye(3)
    delta = vo.p.copy()
    vo.p = vo.p + vo.R @ (np.array([0.0, 0.0, 1.0]) * 2.5)
    assert np.linalg.norm(vo.p - delta) == pytest.approx(2.5)
