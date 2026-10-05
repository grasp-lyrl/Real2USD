"""Phase 1: frames.py round-trips + regression against v1 ply_frame_utils.

The regression is the load-bearing test: the consolidated chain must reproduce v1's
placement math exactly, so Phase 0 numbers are unchanged after the refactor.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from r2s3d_core import frames


# ------------------------------------------------------------ round-trips

def test_quat_wxyz_matches_scipy():
    q_wxyz = np.array([0.9, 0.1, 0.2, 0.3])
    q_wxyz /= np.linalg.norm(q_wxyz)
    R = frames.quat_wxyz_to_R(q_wxyz)
    R_scipy = Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()
    np.testing.assert_allclose(R, R_scipy, atol=1e-12)
    # orthonormal
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)


def test_make_T_invert_roundtrip():
    R = Rotation.from_euler("xyz", [0.3, -1.1, 0.7]).as_matrix()
    t = np.array([1.5, -2.0, 0.25])
    T = frames.make_T(R, t)
    np.testing.assert_allclose(frames.invert(T) @ T, np.eye(4), atol=1e-9)
    np.testing.assert_allclose(frames.invert(frames.invert(T)), T, atol=1e-9)


def test_xyzw_wxyz_consistency():
    q_xyzw = np.array([0.1, 0.2, 0.3, 0.9])
    q_xyzw /= np.linalg.norm(q_xyzw)
    R1 = frames.quat_xyzw_to_R(q_xyzw)
    R2 = frames.quat_wxyz_to_R([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
    np.testing.assert_allclose(R1, R2, atol=1e-12)


# ---------------------------------------------- regression vs v1 ply_frame_utils

def _load_v1():
    p = Path(__file__).resolve().parents[2] / "real2sam3d" / "real2sam3d" / "ply_frame_utils.py"
    if not p.is_file():
        return None
    spec = importlib.util.spec_from_file_location("v1_ply_frame_utils", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_V1 = _load_v1()
_need_v1 = pytest.mark.skipif(_V1 is None, reason="v1 ply_frame_utils.py not found")

_SCALE = np.array([1.2, 0.8, 2.0])
_QUAT_WXYZ = np.array([0.9, 0.1, 0.2, 0.3]) / np.linalg.norm([0.9, 0.1, 0.2, 0.3])
_TRANS = np.array([0.5, -0.3, 3.0])
_ODOM = {"position": [1.0, 2.0, 0.5], "orientation": [0.0, 0.0, 0.3826834, 0.9238795]}  # 45deg yaw xyzw


@_need_v1
def test_T_cam_raw_matches_v1():
    got = frames.T_cam_raw(_SCALE, _QUAT_WXYZ, _TRANS)
    ref = _V1.build_T_raw_to_cam(_SCALE, _QUAT_WXYZ, _TRANS)
    np.testing.assert_allclose(got, ref, atol=1e-9)


@_need_v1
def test_go2_chain_matches_v1():
    got_cam = frames.T_odom_cam_go2(_ODOM, odom_quat="xyzw")
    ref_cam = _V1.build_T_cam_to_odom(_ODOM)
    np.testing.assert_allclose(got_cam, ref_cam, atol=1e-9)
    got_raw = frames.T_odom_raw_go2(_ODOM, _SCALE, _QUAT_WXYZ, _TRANS, odom_quat="xyzw")
    ref_raw = _V1.build_T_raw_to_odom(_ODOM, _SCALE, _QUAT_WXYZ, _TRANS)
    np.testing.assert_allclose(got_raw, ref_raw, atol=1e-9)


def test_go2_extrinsic_config_loads():
    cal = frames.load_go2_calibration()
    assert cal["go2"]["t_body_cam"] == [0.285, 0.0, 0.01]
