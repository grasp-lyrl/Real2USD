"""Single source of truth for coordinate-frame transforms (Phase 1).

Consolidates the frame math that v1 scattered across `ply_frame_utils.py` (5
hardcoded rotation constants) and three ROS nodes (duplicated camera extrinsics).
ROS-free and unit-tested; ROS nodes should import from here instead of redefining.

Convention (matches r2s3d_core.data.base.Frame)
------------------------------------------------
* Column vectors. ``T_a_b`` is a 4x4 mapping points from frame ``b`` to frame ``a``:
  ``p_a = T_a_b @ [p_b; 1]``. Compose right-to-left: ``T_a_c = T_a_b @ T_b_c``.
* Camera frame = OpenCV optical: x right, y down, z forward.
* World frame = gravity-aligned Z-up.
* Quaternions: ``wxyz`` is scalar-first (SAM3D / PyTorch3D); ``xyzw`` is scalar-last
  (ROS / scipy).

SAM3D shape chain (raw canonical mesh -> pointmap -> OpenCV camera) is verified equal
to v1 ``ply_frame_utils.build_T_raw_to_cam`` (see tests). The Go2 body chain
(camera -> world/odom) uses the robot extrinsic from ``config/go2_calibration.yaml``
and mirrors v1 ``build_T_cam_to_odom`` for the eventual ROS wrapper.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import yaml

# --- SAM3D / PyTorch3D chain constants (row-vector originals from v1, applied here
#     in column form). Names per docs/PHASE_SPECS.md. ---
R_FLIP_Z = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64)
R_ZUP_YUP = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float64)    # Y-up -> Z-up
R_CAM_PT3D = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64)  # pointmap -> OpenCV cam

# Go2 demo_go2 world<-cam rotation (kept for parity with v1 naming/usage).
R_WORLD_CAM_GO2 = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)

_CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"


# ----------------------------------------------------------------- helpers

def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Assemble a 4x4 from rotation (3x3) and translation (3,)."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64).ravel()
    return T


def invert(T: np.ndarray) -> np.ndarray:
    """Inverse of a rigid (or affine) 4x4 transform."""
    R = T[:3, :3]
    t = T[:3, 3]
    Ri = np.linalg.inv(R)
    return make_T(Ri, -Ri @ t)


def quat_wxyz_to_R(q) -> np.ndarray:
    """Scalar-first quaternion (w,x,y,z) -> 3x3 rotation."""
    q = np.asarray(q, dtype=np.float64).ravel()
    w, x, y, z = q
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n > 0:
        w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def quat_xyzw_to_R(q) -> np.ndarray:
    """Scalar-last quaternion (x,y,z,w) -> 3x3 rotation (ROS/scipy)."""
    q = np.asarray(q, dtype=np.float64).ravel()
    return quat_wxyz_to_R([q[3], q[0], q[1], q[2]])


# --------------------------------------------------- SAM3D shape chain

def T_cam_raw(scale, quat_wxyz, translation) -> np.ndarray:
    """Affine 4x4 mapping a raw SAM3D mesh vertex into the OpenCV camera frame.

    Column-form of v1's row-vector chain
    ``verts @ R_FLIP_Z @ R_ZUP_YUP * s @ R(wxyz) + t`` then ``@ R_CAM_PT3D``:

        p_cam = R_CAM_PT3D.T @ ( t + R.T @ diag(s) @ R_ZUP_YUP.T @ R_FLIP_Z.T @ p_raw )

    Scale may be anisotropic, so this is a general affine, not rigid. Verified equal
    to ``ply_frame_utils.build_T_raw_to_cam`` in tests.
    """
    s = np.asarray(scale, dtype=np.float64).reshape(-1)
    s = np.repeat(s, 3) if s.size == 1 else s[:3]
    R = quat_wxyz_to_R(quat_wxyz)
    t = np.asarray(translation, dtype=np.float64).reshape(3)
    A = R_CAM_PT3D.T @ R.T @ np.diag(s) @ R_ZUP_YUP.T @ R_FLIP_Z.T
    b = R_CAM_PT3D.T @ t
    return make_T(A, b)


# --------------------------------------------------- Go2 body chain

def load_go2_calibration(path: Optional[Path] = None) -> dict:
    path = Path(path) if path else _CONFIG_DIR / "go2_calibration.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def T_odom_cam_go2(odom: dict, init_odom: Optional[dict] = None,
                   odom_quat: str = "xyzw", t_body_cam=(0.285, 0.0, 0.01)) -> np.ndarray:
    """4x4 camera -> odom for the Go2, mirroring v1 ``build_T_cam_to_odom``.

    ``odom`` = {position:[x,y,z], orientation: quaternion}. ``init_odom`` (optional)
    makes the odom position first-step-relative (subtract [x,y,0]). ``t_body_cam`` is
    the camera offset in the odom body frame (config/go2_calibration.yaml).
    """
    t_odom = np.asarray(odom["position"], dtype=np.float64).copy()
    if init_odom is not None:
        ti = init_odom["position"]
        t_odom -= np.array([ti[0], ti[1], 0.0], dtype=np.float64)
    R_odom = quat_wxyz_to_R(odom["orientation"]) if odom_quat == "wxyz" else quat_xyzw_to_R(odom["orientation"])
    t_wtc = np.asarray(t_body_cam, dtype=np.float64)
    # v1: p_world = R_WORLD_CAM_GO2 @ p_cam + t_wtc ; p_odom = R_odom @ p_world + t_odom
    R = R_odom @ R_WORLD_CAM_GO2
    t = R_odom @ t_wtc + t_odom
    return make_T(R, t)


def T_odom_raw_go2(odom: dict, scale, quat_wxyz, translation,
                   init_odom: Optional[dict] = None, **kw) -> np.ndarray:
    """4x4 raw SAM3D mesh -> Go2 odom = T_odom_cam @ T_cam_raw (v1 build_T_raw_to_odom)."""
    return T_odom_cam_go2(odom, init_odom=init_odom, **kw) @ T_cam_raw(scale, quat_wxyz, translation)
