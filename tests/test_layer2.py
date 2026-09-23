import numpy as np
import pytest

from ugvnav.layer2.negative_obstacle import NegativeObstacleDetector
from ugvnav.layer2.dynamic import DynamicTracker, Track
from ugvnav.layer2.voxel import VoxelGrid


# ------------------------------------------------------- negative obstacles
def _depth_with_cliff(h=120, w=160, cliff_row=70, far=25.0):
    """Ground receding smoothly, then a sudden jump to far depth (a ditch)."""
    d = np.tile(np.linspace(2.0, 12.0, h, dtype=np.float32)[:, None], (1, w))
    d = d[::-1].copy()                       # near at the bottom of the image
    d[cliff_row - 6:cliff_row, 60:110] = far  # the hole
    return d


def test_negative_obstacle_fires_on_a_depth_discontinuity():
    det = NegativeObstacleDetector(grad_percentile=98.0, min_area=10)
    res = det.process(_depth_with_cliff())

    assert res.mask.any()
    rows = np.flatnonzero(res.mask.any(axis=1))
    assert 55 <= rows.min() <= 80            # detection sits at the cliff
    cols = np.flatnonzero(res.mask.any(axis=0))
    assert cols.min() >= 50 and cols.max() <= 120


def test_negative_obstacle_quiet_on_smooth_ground():
    """A smoothly receding floor must not be reported as a hazard."""
    smooth = np.tile(np.linspace(12.0, 2.0, 120, dtype=np.float32)[:, None], (1, 160))
    det = NegativeObstacleDetector(grad_percentile=99.9, min_area=200)
    res = det.process(smooth)
    assert res.mask.sum() < smooth.size * 0.02


def test_negative_obstacle_uses_height_below_plane():
    """Points below the ground plane count even without a depth edge."""
    depth = np.full((60, 60), 5.0, np.float32)
    height = np.zeros((60, 60), np.float32)
    height[20:30, 20:30] = -0.9               # a pit
    det = NegativeObstacleDetector(drop_m=0.25, grad_percentile=100.0, min_area=10)
    res = det.process(depth, height=height)
    assert res.mask[20:30, 20:30].mean() > 0.9
    assert res.mask[40:, 40:].sum() == 0


def test_negative_obstacle_respects_valid_mask():
    depth = _depth_with_cliff()
    valid = np.zeros_like(depth, dtype=bool)
    valid[90:, :] = True                      # exclude the cliff region
    res = NegativeObstacleDetector(min_area=5).process(depth, valid=valid)
    assert res.mask[:90].sum() == 0


def test_negative_obstacle_column_ranges_are_virtual_scan():
    det = NegativeObstacleDetector(grad_percentile=98.0, min_area=10)
    res = det.process(_depth_with_cliff())
    hit = np.isfinite(res.ranges)
    assert hit.any()
    assert np.all(res.ranges[hit] > 0)
    assert not hit[:40].any()                 # clear columns stay +inf


def test_despeckle_removes_tiny_blobs():
    depth = np.full((80, 80), 5.0, np.float32)
    height = np.zeros((80, 80), np.float32)
    height[10, 10] = -2.0                     # single stray pixel
    det = NegativeObstacleDetector(drop_m=0.25, grad_percentile=100.0, min_area=25)
    assert det.process(depth, height=height).mask.sum() == 0


# ------------------------------------------------------- dynamic obstacles
def _texture(h=120, w=160, seed=0):
    import cv2
    rng = np.random.default_rng(seed)
    img = (rng.random((h, w)) * 255).astype(np.uint8)
    return cv2.GaussianBlur(img, (0, 0), 1.2)


def test_ego_flow_matches_a_known_translation():
    H = np.array([[1.0, 0.0, 3.0],
                  [0.0, 1.0, -2.0],
                  [0.0, 0.0, 1.0]])
    flow = DynamicTracker.ego_flow((30, 40), H)
    assert np.allclose(flow[..., 0], 3.0)
    assert np.allclose(flow[..., 1], -2.0)


def test_ego_flow_identity_is_zero():
    assert np.allclose(DynamicTracker.ego_flow((20, 20), np.eye(3)), 0.0)


def test_dynamic_tracker_ignores_pure_ego_motion():
    """Shifting the whole scene must not produce any dynamic detection."""
    import cv2
    base = _texture(seed=1)
    shift = np.float32([[1, 0, 4], [0, 1, 0]])
    moved = cv2.warpAffine(base, shift, (base.shape[1], base.shape[0]),
                           borderMode=cv2.BORDER_REFLECT)
    H = np.array([[1, 0, 4], [0, 1, 0], [0, 0, 1]], dtype=np.float64)

    res = DynamicTracker(thresh_px=1.5, min_area=80).process(base, moved, H=H)
    assert res.mask.mean() < 0.05
    assert res.tracks == []


def test_dynamic_tracker_finds_an_independently_moving_object():
    import cv2
    base = _texture(seed=2)
    shift = np.float32([[1, 0, 3], [0, 1, 0]])
    moved = cv2.warpAffine(base, shift, (base.shape[1], base.shape[0]),
                           borderMode=cv2.BORDER_REFLECT)

    # a bright square that moves further than the background does
    patch = _texture(24, 24, seed=9) // 2 + 120
    base_obj, moved_obj = base.copy(), moved.copy()
    base_obj[40:64, 30:54] = patch
    moved_obj[40:64, 30 + 18:54 + 18] = patch     # +18 px, background only +3

    H = np.array([[1, 0, 3], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    res = DynamicTracker(thresh_px=2.0, min_area=40).process(base_obj, moved_obj, H=H)

    assert res.tracks, "expected at least one moving object"
    t = res.tracks[0]
    assert 25 <= t.cx <= 80 and 30 <= t.cy <= 80
    assert t.vx > 1.5                             # moving to the right
    assert t.speed > 1.5


def test_track_predicts_future_position():
    t = Track(cx=10.0, cy=20.0, vx=2.0, vy=-1.0, area=100, speed=2.24)
    assert t.predict(3.0) == (16.0, 17.0)


def test_background_homography_falls_back_to_identity():
    blank = np.zeros((60, 60), np.uint8)
    H = DynamicTracker().background_homography(blank, blank)
    assert np.allclose(H, np.eye(3))


# ------------------------------------------------------------------- voxels
def test_voxel_key_and_centre_roundtrip():
    g = VoxelGrid(resolution=0.5)
    k = g.key([1.2, -0.3, 2.9])
    assert k == (2, -1, 5)
    c = g.centre(k)
    assert g.key(c) == k


def test_voxel_rejects_bad_resolution():
    with pytest.raises(ValueError):
        VoxelGrid(resolution=0.0)


def test_voxel_needs_repeat_observations():
    g = VoxelGrid(resolution=0.2, hit_threshold=2)
    p = np.array([[1.0, 2.0, 0.5]])
    g.integrate(p)
    assert not g.occupied(p[0])          # one hit is not enough
    g.integrate(p)
    assert g.occupied(p[0])
    assert len(g) == 1


def test_voxel_applies_pose_transform():
    g = VoxelGrid(resolution=0.25, hit_threshold=1)
    pts = np.array([[0.0, 1.0, 0.0]])
    t = np.array([5.0, 0.0, 0.0])
    g.integrate(pts, pose_t=t)
    assert g.occupied([5.0, 1.0, 0.0])
    assert not g.occupied([0.0, 1.0, 0.0])


def test_voxel_discards_far_points():
    g = VoxelGrid(resolution=0.5, hit_threshold=1)
    assert g.integrate(np.array([[0.0, 100.0, 0.0]]), max_range=15.0) == 0
    assert len(g) == 0


def test_voxel_ignores_non_finite_points():
    g = VoxelGrid(resolution=0.5, hit_threshold=1)
    pts = np.array([[np.nan, 1.0, 0.0], [np.inf, 0.0, 0.0], [1.0, 1.0, 0.0]])
    assert g.integrate(pts) == 1


def test_overhead_blocked_detects_a_branch():
    """A branch at 1.2 m blocks a 1.5 m vehicle but clears a 1.0 m one."""
    g = VoxelGrid(resolution=0.2, hit_threshold=1)
    g.integrate(np.array([[2.0, 3.0, 1.2]]))

    assert g.overhead_blocked(2.0, 3.0, robot_height=1.5)
    assert not g.overhead_blocked(2.0, 3.0, robot_height=1.0)
    assert not g.overhead_blocked(0.0, 0.0, robot_height=1.5)   # elsewhere clear


def test_overhead_ignores_the_ground_itself():
    g = VoxelGrid(resolution=0.2, hit_threshold=1)
    g.integrate(np.array([[1.0, 1.0, 0.0]]))       # ground point
    assert not g.overhead_blocked(1.0, 1.0, robot_height=1.5, floor_clearance=0.25)


def test_clearance_map_shape_and_hit():
    g = VoxelGrid(resolution=0.5, hit_threshold=1)
    g.integrate(np.array([[0.0, 6.0, 1.0]]))
    m = g.clearance_map(cells=16, robot_height=1.5)
    assert m.shape == (16, 16)
    assert m.any()
