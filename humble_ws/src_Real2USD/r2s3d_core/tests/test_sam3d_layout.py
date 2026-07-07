"""Unit tests for the SAM3D-layout baseline's pure, non-gated pieces:
placement composition math and mask/best-view selection. SAM3D inference itself is
a human-gated external dependency and is not exercised here.
"""

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from r2s3d_core.baselines import sam3d_layout as B
from r2s3d_core.data.base import Frame, GTObject


def _cam_frame(T_world_cam, H=480, W=640, f=300.0):
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    return Frame(rgb=np.zeros((H, W, 3), np.uint8),
                 depth=np.ones((H, W), np.float32),
                 K=K, T_world_cam=T_world_cam, stamp=0.0, frame_id=0)


def test_place_identity_translation():
    mesh = trimesh.creation.box(extents=[1, 1, 1])
    posed, T, ext = B.place_from_sam3d(
        mesh, scale=1.0, quat_wxyz=[1, 0, 0, 0], translation=[0, 0, 2.0],
        T_world_cam=np.eye(4),
    )
    # pointmap translation [0,0,2] -> camera (diag(-1,-1,1)) -> [0,0,2]; world = cam
    np.testing.assert_allclose(T[:3, 3], [0, 0, 2.0], atol=1e-6)
    np.testing.assert_allclose(np.sort(ext), [1, 1, 1], atol=1e-6)


def test_place_isotropic_scale():
    mesh = trimesh.creation.box(extents=[1, 1, 1])
    _, T, ext = B.place_from_sam3d(mesh, 2.0, [1, 0, 0, 0], [0, 0, 3.0], np.eye(4))
    np.testing.assert_allclose(np.sort(ext), [2, 2, 2], atol=1e-6)
    # translation is applied after scale -> not scaled
    np.testing.assert_allclose(T[:3, 3], [0, 0, 3.0], atol=1e-6)


def test_place_anisotropic_scale():
    mesh = trimesh.creation.box(extents=[1, 1, 1])
    _, _, ext = B.place_from_sam3d(mesh, [2, 1, 1], [1, 0, 0, 0], [0, 0, 2.0], np.eye(4))
    np.testing.assert_allclose(np.sort(ext), [1, 1, 2], atol=1e-6)


def test_place_respects_world_cam():
    # move the camera 5m in world +x; object should shift with it
    mesh = trimesh.creation.box(extents=[1, 1, 1])
    Twc = np.eye(4)
    Twc[0, 3] = 5.0
    _, T, _ = B.place_from_sam3d(mesh, 1.0, [1, 0, 0, 0], [0, 0, 2.0], Twc)
    np.testing.assert_allclose(T[:3, 3], [5.0, 0, 2.0], atol=1e-6)


def _box_gt(center, extents):
    T = np.eye(4)
    T[:3, 3] = center
    m = trimesh.creation.box(extents=extents)
    m.apply_transform(T)
    return GTObject(1, "chair", T, np.asarray(extents, float), m)


def test_render_mask_visible_and_border():
    g = _box_gt([0, 0, 3.0], [1, 1, 1])  # centered 3m in front
    # camera at origin looking +z (OpenCV optical): world == cam
    fr = _cam_frame(np.eye(4))
    mask = B.render_instance_mask(g.mesh, fr)
    assert mask is not None and mask.any()
    area, border = B._view_score(mask)
    assert 0 < area < 1
    assert not border


def test_render_mask_behind_camera_is_none():
    g = _box_gt([0, 0, -3.0], [1, 1, 1])  # behind the camera
    fr = _cam_frame(np.eye(4))
    assert B.render_instance_mask(g.mesh, fr) is None


def test_select_best_view_prefers_centered_nonborder():
    g = _box_gt([0, 0, 3.0], [1, 1, 1])
    centered = _cam_frame(np.eye(4))
    # a camera shifted so the object projects to the image edge
    Twc = np.eye(4)
    Twc[0, 3] = -2.6  # object appears far to one side
    edge = _cam_frame(Twc)
    idx = B.select_best_view(g, [edge, centered])
    assert idx == 1  # the centered, non-border view


def test_masked_depth_cloud_roundtrip():
    g = _box_gt([0, 0, 3.0], [1, 1, 1])
    fr = _cam_frame(np.eye(4))
    # give the frame a constant depth so back-projection is well-defined
    fr = fr._replace(depth=np.full_like(fr.depth, 3.0))
    mask = B.render_instance_mask(g.mesh, fr)
    cloud = B._masked_depth_cloud(fr, mask)
    assert len(cloud) > 0
    assert np.allclose(cloud[:, 2], 3.0, atol=1e-5)  # z == depth, world==cam
