"""Replica backend tests. Skipped unless the dataset is present locally.

The core invariant: frames back-project into the room reconstruction mesh bounds,
which validates intrinsics, depth scale, camera convention, and world frame all at
once (see data/replica.py module docstring for the derivation).
"""

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from r2s3d_core.data.registry import data_root, make_source

_ROOT = data_root() / "replica"
_HAVE = (_ROOT / "Replica" / "room0" / "traj.txt").is_file()
pytestmark = pytest.mark.skipif(not _HAVE, reason=f"Replica not downloaded at {_ROOT}")


def test_frame_shapes_and_intrinsics():
    src = make_source("replica", "room0", stride=500, load_gt=False)
    assert len(src) > 0
    f = next(iter(src))
    assert f.rgb.dtype == np.uint8 and f.rgb.shape[2] == 3
    assert f.depth.shape == f.rgb.shape[:2]
    assert f.K[0, 0] == pytest.approx(600.0)
    # indoor depth in meters
    d = f.depth[f.depth > 0]
    assert 0.1 < np.median(d) < 15.0
    # pose is a valid rigid transform
    R = f.T_world_cam[:3, :3]
    assert np.abs(R @ R.T - np.eye(3)).max() < 1e-6
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-6)


def test_backprojection_within_mesh_bounds():
    mesh_ply = _ROOT / "Replica" / "room0_mesh.ply"
    if not mesh_ply.is_file():
        pytest.skip("room0_mesh.ply not present")
    m = trimesh.load(str(mesh_ply), process=False)
    src = make_source("replica", "room0", stride=1, load_gt=False)
    f = next(iter(src))
    H, W = f.depth.shape
    fx, fy, cx, cy = f.K[0, 0], f.K[1, 1], f.K[0, 2], f.K[1, 2]
    js, is_ = np.where(f.depth > 0)
    z = f.depth[js, is_]
    pc = np.stack([(is_ - cx) * z / fx, (js - cy) * z / fy, z], axis=1)
    pw = (f.T_world_cam[:3, :3] @ pc.T).T + f.T_world_cam[:3, 3]
    inside = ((pw >= m.bounds[0] - 0.1) & (pw <= m.bounds[1] + 0.1)).all(1).mean()
    assert inside > 0.99
