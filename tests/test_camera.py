import numpy as np
import pytest

from ugvnav.camera import Camera, fit_ground_plane, plane_heights


@pytest.fixture
def cam():
    return Camera.from_fov(320, 240, hfov_deg=70.0, height_m=0.8)


def test_intrinsics_matrix_shape(cam):
    K = cam.K
    assert K.shape == (3, 3)
    assert K[2, 2] == 1.0
    assert K[0, 2] == pytest.approx(160.0)
    assert K[1, 2] == pytest.approx(120.0)


def test_unproject_then_project_is_identity(cam):
    """Back-projection and projection must be exact inverses."""
    depth = np.random.default_rng(0).uniform(1.0, 20.0, (240, 320))
    pts = cam.unproject(depth)
    uv = cam.project(pts)

    us, vs = np.meshgrid(np.arange(320.0), np.arange(240.0))
    assert np.allclose(uv[..., 0], us, atol=1e-6)
    assert np.allclose(uv[..., 1], vs, atol=1e-6)


def test_unproject_preserves_depth_as_z(cam):
    depth = np.full((10, 10), 4.2)
    pts = cam.unproject(depth)
    assert np.allclose(pts[..., 2], 4.2)


def test_principal_point_maps_to_optical_axis(cam):
    depth = np.full((240, 320), 5.0)
    pts = cam.unproject(depth)
    assert pts[120, 160, 0] == pytest.approx(0.0, abs=1e-9)
    assert pts[120, 160, 1] == pytest.approx(0.0, abs=1e-9)


def test_project_rejects_points_behind_camera(cam):
    behind = np.array([[[1.0, 1.0, -3.0]]])
    assert np.isnan(cam.project(behind)).all()


def test_fit_ground_plane_recovers_known_plane():
    """A synthetic plane y = 1.5 must be recovered exactly."""
    rng = np.random.default_rng(1)
    x = rng.uniform(-5, 5, 4000)
    z = rng.uniform(1, 20, 4000)
    y = np.full_like(x, 1.5)
    pts = np.stack([x, y, z], axis=-1)

    n, d, inl = fit_ground_plane(pts, seed=3)

    # plane is y = 1.5  ->  normal along Y, offset 1.5 with the sign convention
    assert abs(abs(n[1]) - 1.0) < 1e-6
    assert abs(n[0]) < 1e-6 and abs(n[2]) < 1e-6
    assert plane_heights(pts, n, d).max() < 1e-6
    assert inl.sum() > 3900


def test_fit_ground_plane_is_robust_to_outliers():
    rng = np.random.default_rng(2)
    n_in = 3000
    ground = np.stack([rng.uniform(-5, 5, n_in),
                       np.full(n_in, 2.0),
                       rng.uniform(1, 20, n_in)], axis=-1)
    junk = rng.uniform(-5, 5, (600, 3)) + np.array([0.0, -4.0, 10.0])
    pts = np.concatenate([ground, junk])

    n, d, inl = fit_ground_plane(pts, seed=5)

    assert inl[:n_in].mean() > 0.95      # ground kept
    assert inl[n_in:].mean() < 0.20      # junk mostly rejected


def test_plane_heights_sign_is_positive_above_ground():
    """Points above the ground plane must report positive height."""
    rng = np.random.default_rng(4)
    n_pts = 2000
    pts = np.stack([rng.uniform(-5, 5, n_pts),
                    np.full(n_pts, 2.0),
                    rng.uniform(1, 20, n_pts)], axis=-1)
    n, d, _ = fit_ground_plane(pts, seed=1)

    # camera frame has Y down, so "up" is -Y
    above = np.array([[0.0, 2.0 - 0.5, 8.0]])
    below = np.array([[0.0, 2.0 + 0.5, 8.0]])
    assert plane_heights(above, n, d)[0] > 0.4
    assert plane_heights(below, n, d)[0] < -0.4


def test_fit_ground_plane_needs_three_points():
    with pytest.raises(ValueError):
        fit_ground_plane(np.zeros((2, 3)))
