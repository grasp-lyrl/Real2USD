"""ProcTHOR backend: pure-function units + a gated AI2-THOR round-trip.

The round-trip is the load-bearing transform test (CLAUDE.md discipline): masked
rendered depth, back-projected through T_world_cam + K, must land inside the object's
GT oriented box. It is skipped when ai2thor/prior (the `procthor` extra) or a render
display are unavailable, so the core suite stays light.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from r2s3d_core.data.procthor import (
    intrinsics_from_fov,
    thor_camera_to_world,
    _obb_from_corner_points,
    _M_WU,
)


# ------------------------------------------------------------ pure units

def test_intrinsics_vertical_fov():
    # 90 deg vertical FOV on a 480-tall image => fy = (H/2)/tan(45) = 240
    K = intrinsics_from_fov(480, 640, 90.0, fov_axis="vertical")
    assert np.isclose(K[1, 1], 240.0)
    assert np.isclose(K[0, 0], 240.0)  # square pixels
    assert np.isclose(K[0, 2], 320.0) and np.isclose(K[1, 2], 240.0)
    # horizontal option uses width for the focal base
    Kh = intrinsics_from_fov(480, 640, 90.0, fov_axis="horizontal")
    assert np.isclose(Kh[0, 0], 320.0)


def test_camera_to_world_is_proper_rotation():
    for yaw in (0.0, 90.0, 137.0, 270.0):
        for horizon in (0.0, 30.0, -15.0):
            T = thor_camera_to_world({"x": 1.0, "y": 1.5, "z": -2.0}, yaw, horizon)
            R = T[:3, :3]
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
            assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)  # right-handed
    # translation is the Unity->world permutation of the camera position
    T = thor_camera_to_world({"x": 1.0, "y": 1.5, "z": -2.0}, 0.0, 0.0)
    np.testing.assert_allclose(T[:3, 3], _M_WU @ np.array([1.0, 1.5, -2.0]), atol=1e-9)


def test_camera_optical_axes_at_identity_pose():
    # yaw=0, horizon=0: camera looks along Unity +Z (=> our world +Y), optical z-forward.
    T = thor_camera_to_world({"x": 0.0, "y": 0.0, "z": 0.0}, 0.0, 0.0)
    R = T[:3, :3]
    fwd_world = R @ np.array([0.0, 0.0, 1.0])   # optical +z
    down_world = R @ np.array([0.0, 1.0, 0.0])  # optical +y (down)
    np.testing.assert_allclose(fwd_world, [0.0, 1.0, 0.0], atol=1e-9)   # Unity +Z -> our +Y
    np.testing.assert_allclose(down_world, [0.0, 0.0, -1.0], atol=1e-9)  # optical down -> our -Z


def test_obb_from_corner_points_recovers_rotated_box():
    ext = np.array([0.4, 0.8, 1.2])
    R = Rotation.from_euler("z", 35, degrees=True).as_matrix()
    center = np.array([1.0, -2.0, 0.5])
    # 8 corners of the box in world frame
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    corners = (R @ (signs * ext / 2).T).T + center
    T, rec_ext = _obb_from_corner_points(corners)
    # extents recovered up to axis permutation
    np.testing.assert_allclose(np.sort(rec_ext), np.sort(ext), atol=1e-9)
    np.testing.assert_allclose(T[:3, 3], center, atol=1e-9)
    # recovered frame is orthonormal + right-handed
    np.testing.assert_allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(T[:3, :3]), 1.0, atol=1e-9)


# ------------------------------------------------------------ gated round-trip

def _have_thor():
    try:
        import ai2thor  # noqa: F401
        import prior  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _have_thor(), reason="procthor extra (ai2thor/prior) not installed")
def test_backprojection_inside_gt_obb():
    """Masked depth -> world must land inside the GT OBB (vertical FOV), beating horizontal."""
    import prior
    from ai2thor.controller import Controller
    from scipy.ndimage import binary_erosion
    from r2s3d_core.data.procthor import _THOR_FAR

    house = prior.load_dataset("procthor-10k")["train"][137]
    W, H = 640, 480
    try:
        c = Controller(scene=house, renderDepthImage=True, renderInstanceSegmentation=True,
                       width=W, height=H, gridSize=0.25, quality="Low")
    except Exception as ex:  # pragma: no cover - no display / GPU on this host
        pytest.skip(f"AI2-THOR could not start a renderer: {ex}")
    try:
        fov = c.last_event.metadata["fov"]
        obb = {}
        for o in c.last_event.metadata["objects"]:
            b = o.get("objectOrientedBoundingBox")
            if b and b.get("cornerPoints"):
                cw = (_M_WU @ np.asarray(b["cornerPoints"], float).T).T
                obb[o["objectId"]] = _obb_from_corner_points(cw)
        rp = c.step(action="GetReachablePositions").metadata["actionReturn"]
        rp = sorted(rp, key=lambda p: (round(p["x"], 3), round(p["z"], 3)))[::120][:4]

        def containment(axis):
            K = intrinsics_from_fov(H, W, fov, fov_axis=axis)
            Kinv = np.linalg.inv(K)
            fr = []
            for pos in rp:
                for yaw in (0, 90, 180, 270):
                    ev = c.step(action="Teleport", position=pos, rotation=dict(x=0, y=yaw, z=0),
                                horizon=0, standing=True)
                    if not ev.metadata["lastActionSuccess"]:
                        continue
                    depth = np.asarray(ev.depth_frame, np.float32)
                    T = thor_camera_to_world(ev.metadata["cameraPosition"], yaw, 0)
                    for oid, m in ev.instance_masks.items():
                        if oid not in obb or m.sum() < 300:
                            continue
                        m = binary_erosion(m, iterations=2)
                        ys, xs = np.nonzero(m)
                        if len(xs) < 40:
                            continue
                        d = depth[ys, xs]
                        ok = np.isfinite(d) & (d > 0) & (d < _THOR_FAR)
                        if ok.sum() < 30:
                            continue
                        ys, xs, d = ys[ok], xs[ok], d[ok]
                        pix = np.stack([xs, ys, np.ones_like(xs)], 0).astype(float)
                        world = (T[:3, :3] @ ((Kinv @ pix) * d)) + T[:3, 3:4]
                        To, ext = obb[oid]
                        local = To[:3, :3].T @ (world - To[:3, 3:4])
                        fr.append(np.mean(np.all(np.abs(local) <= (ext[:, None] / 2 + 0.1), axis=0)))
            return np.array(fr)

        vert = containment("vertical")
        horiz = containment("horizontal")
        assert len(vert) > 20
        # vertical FOV: masked depth lands solidly inside the GT boxes ...
        assert np.median(vert) > 0.6, f"median containment {np.median(vert):.3f} too low"
        # ... and clearly beats the wrong (horizontal) intrinsics choice.
        assert np.median(vert) > np.median(horiz) + 0.2
    finally:
        c.stop()
