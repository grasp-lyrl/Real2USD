"""Projection math for the Go2 lidar->depth synthesis (data/rosbag.py)."""

import numpy as np

from r2s3d_core.lidar_depth import project_cloud_to_depth

K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])
H, W = 480, 640


def _pixel(Xc, Yc, Zc):
    u = K[0, 0] * Xc / Zc + K[0, 2]
    v = K[1, 1] * Yc / Zc + K[1, 2]
    return int(np.floor(u)), int(np.floor(v))


def test_identity_projection_places_point_at_expected_pixel():
    # camera == world (T_cam_world = I); a point 2 m in front, offset right/down.
    p = np.array([[0.4, 0.2, 2.0]])
    depth = project_cloud_to_depth(p, np.eye(4), K, H, W)
    u, v = _pixel(*p[0])
    assert abs(depth[v, u] - 2.0) < 1e-4
    assert (depth > 0).sum() == 1  # exactly one pixel set


def test_nearest_point_wins_zbuffer():
    # two points on the same ray (same pixel), different depth -> nearest kept.
    near = np.array([2.0, 1.0, 4.0])          # Xc/Zc, Yc/Zc fixed
    far = near * (7.0 / 4.0)                   # same direction, farther (Z=7)
    depth = project_cloud_to_depth(np.stack([far, near]), np.eye(4), K, H, W)
    u, v = _pixel(*near)
    assert abs(depth[v, u] - 4.0) < 1e-4       # min depth survives, not 7


def test_points_behind_and_outside_are_dropped():
    behind = np.array([[0.1, 0.1, -1.0]])      # Z<0
    outside = np.array([[100.0, 0.0, 1.0]])    # projects far off-image
    depth = project_cloud_to_depth(np.vstack([behind, outside]), np.eye(4), K, H, W)
    assert (depth > 0).sum() == 0


def test_pose_transform_applied():
    # world point at (0,0,5); camera translated to (0,0,3) looking +z along world.
    # T_world_cam has t=(0,0,3); T_cam_world maps world->cam so point lands at Z=2.
    from r2s3d_core.frames import make_T, invert
    T_world_cam = make_T(np.eye(3), [0, 0, 3.0])
    depth = project_cloud_to_depth(np.array([[0.0, 0.0, 5.0]]), invert(T_world_cam), K, H, W)
    assert abs(depth[240, 320] - 2.0) < 1e-4


def test_empty_cloud_returns_zeros():
    depth = project_cloud_to_depth(np.empty((0, 3)), np.eye(4), K, H, W)
    assert depth.shape == (H, W)
    assert not depth.any()
