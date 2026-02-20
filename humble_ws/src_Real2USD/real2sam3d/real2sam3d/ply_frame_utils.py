"""
Frame conversion for SAM3D + Go2 (injector). Used by sam3d_injector_node.

- SAM3D gives object pose in camera frame (scale, rotation wxyz, translation).
- Odom = robot pose in world (position + orientation xyzw by default).
- demo_go2 constants: R_world_to_cam, t_world_to_camera (names kept as in demo).
- demo_go2 normalizes by init_odom: subtract first step (x,y,0) so world is first-step-relative.

Chain: raw mesh -> (flip Z, Y-up->Z-up, scale, rotate, translate) -> pointmap -> R_pytorch3d_to_cam -> camera
       -> world via (R_world_to_cam, t_world_to_camera) -> odom (with init_odom subtracted).

Set REAL2SAM3D_ODOM_QUAT_WXYZ=1 if go2_odom_orientation is (w,x,y,z).
Set REAL2SAM3D_DEBUG_TRANSFORM=1 to log one line per slot.
Set REAL2SAM3D_DEBUG_DUMP=1 to write transform_debug.json per slot.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

DEBUG_TRANSFORM = os.environ.get("REAL2SAM3D_DEBUG_TRANSFORM", "").strip().lower() in ("1", "true", "yes")
ODOM_QUAT_WXYZ = os.environ.get("REAL2SAM3D_ODOM_QUAT_WXYZ", "").strip().lower() in ("1", "true", "yes")
DEBUG_DUMP = os.environ.get("REAL2SAM3D_DEBUG_DUMP", "").strip().lower() in ("1", "true", "yes")


def _quat_xyzw_to_rotation_matrix(q: List[float]) -> np.ndarray:
    """Quaternion [qx, qy, qz, qw] (ROS/scipy) to 3x3."""
    qx, qy, qz, qw = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n > 0:
        qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float64)


def _quat_to_rotation_matrix(q: List[float], convention: str = "xyzw") -> np.ndarray:
    """Quat to 3x3. convention: 'xyzw' (ROS) or 'wxyz' (scalar first)."""
    q = [float(q[i]) for i in range(4)]
    if convention.lower() == "wxyz":
        q = [q[1], q[2], q[3], q[0]]
    return _quat_xyzw_to_rotation_matrix(q)


def _quat_wxyz_to_rotation_matrix(rotation: np.ndarray) -> np.ndarray:
    """Quaternion (w,x,y,z) to 3x3 (PyTorch3D / SAM3D)."""
    q = np.asarray(rotation, dtype=np.float64).ravel()
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n > 0:
        w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


# SAM3D / demo_go2 conventions
R_flip_z = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64)
R_yup_to_zup = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float64)  # demo_go2: R_zup_to_yup.T
R_pytorch3d_to_cam = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64)

# demo_go2 constants (keep same naming and usage)
R_world_to_cam = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)
t_world_to_camera = np.array([0.285, 0.0, 0.01], dtype=np.float64)


def transform_mesh_vertices(
    vertices: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """
    demo_go2 transform_mesh_vertices equivalent in numpy (row-vector form).
    This mirrors Transform3d(scale->rotate->translate) for affine points:
      verts = verts * s
      verts = verts @ R_mat
      verts = verts + t
    Returns (N,3) in pointmap frame.
    """
    verts = np.asarray(vertices, dtype=np.float64)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError("vertices must be (N, 3)")
    # 1) Flip Z, 2) Y-up -> Z-up
    verts = verts @ R_flip_z
    verts = verts @ R_yup_to_zup
    # 3) scale -> rotate -> translate (Transform3d order)
    s = np.asarray(scale, dtype=np.float64).ravel()
    if s.size == 1:
        s = np.array([s[0], s[0], s[0]], dtype=np.float64)
    else:
        s = s[:3]
    verts = verts * s
    R_mat = _quat_wxyz_to_rotation_matrix(rotation)
    # Important: demo_go2 / pytorch3d parity uses @ R_mat here (not transpose).
    verts = verts @ R_mat
    t = np.asarray(translation, dtype=np.float64).ravel()[:3]
    verts = verts + t
    return verts


def _demo_go2_cam_to_odom_vertices(
    verts_cam: np.ndarray,
    odom: dict,
    init_odom: Optional[dict] = None,
) -> np.ndarray:
    """Literal demo_go2 row-vector camera->world->odom."""
    t_odom = np.array(odom["position"], dtype=np.float64)
    if init_odom is not None:
        ti = init_odom["position"]
        t_odom = t_odom - np.array([ti[0], ti[1], 0.0], dtype=np.float64)
    conv = "wxyz" if ODOM_QUAT_WXYZ else "xyzw"
    R_odom = _quat_to_rotation_matrix(odom["orientation"], convention=conv)
    verts_world = verts_cam @ R_world_to_cam.T + t_world_to_camera
    verts_odom = verts_world @ R_odom.T + t_odom
    return verts_odom


def _affine_from_mapped_basis(mapped: np.ndarray) -> np.ndarray:
    """
    Build 4x4 (column convention) from mapped points of [0, e1, e2, e3] under an affine transform.
    mapped shape: (4,3), in same order: origin, x-basis, y-basis, z-basis.
    """
    p0 = mapped[0]
    px = mapped[1] - p0
    py = mapped[2] - p0
    pz = mapped[3] - p0
    T = np.eye(4)
    T[:3, :3] = np.column_stack([px, py, pz])
    T[:3, 3] = p0
    return T


def _write_transform_debug_dump(
    job_dir: Path,
    odom: dict,
    init_odom: Optional[dict],
    scale: Optional[np.ndarray],
    rot: Optional[np.ndarray],
    trans: Optional[np.ndarray],
    T_cam_to_odom: np.ndarray,
    T_raw_to_cam: Optional[np.ndarray],
    T_raw_to_odom: Optional[np.ndarray],
) -> None:
    """
    Write transform_debug.json with line-by-line numeric data for one slot.
    This is intended for debugging mismatches between methods.
    """
    try:
        basis = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        report: Dict[str, Any] = {
            "constants": {
                "R_flip_z": R_flip_z.tolist(),
                "R_yup_to_zup": R_yup_to_zup.tolist(),
                "R_pytorch3d_to_cam": R_pytorch3d_to_cam.tolist(),
                "R_world_to_cam": R_world_to_cam.tolist(),
                "t_world_to_camera": t_world_to_camera.tolist(),
            },
            "inputs": {
                "odom_position": odom.get("position"),
                "odom_orientation": odom.get("orientation"),
                "init_odom_position": init_odom.get("position") if isinstance(init_odom, dict) else None,
                "sam3d_scale": scale.tolist() if isinstance(scale, np.ndarray) else None,
                "sam3d_rotation_wxyz": rot.tolist() if isinstance(rot, np.ndarray) else None,
                "sam3d_translation": trans.tolist() if isinstance(trans, np.ndarray) else None,
                "odom_quat_convention": "wxyz" if ODOM_QUAT_WXYZ else "xyzw",
            },
            "matrices": {
                "T_cam_to_odom": T_cam_to_odom.tolist(),
                "T_raw_to_cam": T_raw_to_cam.tolist() if isinstance(T_raw_to_cam, np.ndarray) else None,
                "T_raw_to_odom": T_raw_to_odom.tolist() if isinstance(T_raw_to_odom, np.ndarray) else None,
            },
        }

        if isinstance(scale, np.ndarray) and isinstance(rot, np.ndarray) and isinstance(trans, np.ndarray):
            # Direct chain on basis points
            pts_pointmap = transform_mesh_vertices(basis, rot, trans, scale)
            pts_cam = pts_pointmap @ R_pytorch3d_to_cam
            pts_odom_direct = transform_glb_to_world(pts_cam, odom, init_odom=init_odom)

            # Matrix chain on same basis
            ones = np.ones((basis.shape[0], 1), dtype=np.float64)
            basis_h = np.hstack([basis, ones])
            pts_odom_matrix = (T_raw_to_odom @ basis_h.T).T[:, :3] if isinstance(T_raw_to_odom, np.ndarray) else None

            report["basis_test"] = {
                "raw_basis_points": basis.tolist(),
                "pointmap_from_raw": pts_pointmap.tolist(),
                "cam_from_pointmap": pts_cam.tolist(),
                "odom_direct_chain": pts_odom_direct.tolist(),
                "odom_matrix_chain": pts_odom_matrix.tolist() if isinstance(pts_odom_matrix, np.ndarray) else None,
                "odom_direct_minus_matrix": (pts_odom_direct - pts_odom_matrix).tolist() if isinstance(pts_odom_matrix, np.ndarray) else None,
            }

        out_path = job_dir / "transform_debug.json"
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
    except Exception:
        # Debug dump should never break pipeline.
        return


def transform_glb_to_world(
    verts_cam: np.ndarray,
    odom: dict,
    init_odom: Optional[dict] = None,
) -> np.ndarray:
    """Exact demo_go2 math (row-vector form): cam -> world -> odom."""
    return _demo_go2_cam_to_odom_vertices(verts_cam, odom, init_odom=init_odom)


def build_T_cam_to_odom(odom: dict, init_odom: Optional[dict] = None) -> np.ndarray:
    """4x4 camera->odom from literal demo_go2 vertex ops (no derived algebra)."""
    basis = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    mapped = _demo_go2_cam_to_odom_vertices(basis, odom, init_odom=init_odom)
    return _affine_from_mapped_basis(mapped)


def build_T_raw_to_cam(scale: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """4x4 raw->camera from literal demo_go2 vertex ops (no derived algebra)."""
    basis = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    verts_pointmap = transform_mesh_vertices(basis, rotation, translation, scale)
    verts_cam = verts_pointmap @ R_pytorch3d_to_cam
    return _affine_from_mapped_basis(verts_cam)


def build_T_raw_to_odom(
    odom: dict,
    scale: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    init_odom: Optional[dict] = None,
) -> np.ndarray:
    """4x4 raw -> odom. T_cam_to_odom @ T_raw_to_cam."""
    return build_T_cam_to_odom(odom, init_odom=init_odom) @ build_T_raw_to_cam(scale, rotation, translation)


def pose_from_transform_4x4(T: np.ndarray) -> Tuple[List[float], List[float]]:
    """(position [x,y,z], orientation [qx,qy,qz,qw]) from 4x4."""
    from scipy.spatial.transform import Rotation as R_scipy
    return (T[:3, 3].tolist(), R_scipy.from_matrix(T[:3, :3]).as_quat().tolist())


def apply_sam3d_transforms_to_slot(job_dir: Path, init_odom: Optional[dict] = None) -> bool:
    """
    Load pose.json (frame=sam3d_raw); run demo_go2-style pipeline; write only transforms to pose.json.
    Writes: cam_to_odom_*, transform_odom_from_raw, initial_position, initial_orientation.
    If init_odom is provided, odom position is made relative to init (x,y,0) so output matches demo_go2.
    """
    pose_path = job_dir / "pose.json"
    if not pose_path.exists():
        return False
    try:
        with open(pose_path) as f:
            pose = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    frame = pose.get("frame")
    if frame not in ("sam3d_raw", "z_up_odom"):
        return False

    odom = {
        "position": pose["go2_odom_position"],
        "orientation": pose["go2_odom_orientation"],
    }
    scale = pose.get("sam3d_scale")
    rot = pose.get("sam3d_rotation")
    trans = pose.get("sam3d_translation")

    T_cam_to_odom = build_T_cam_to_odom(odom, init_odom=init_odom)
    pose["cam_to_odom_rotation"] = [float(x) for x in T_cam_to_odom[:3, :3].ravel()]
    pose["cam_to_odom_translation"] = T_cam_to_odom[:3, 3].tolist()
    # Persist explicit 4x4 for camera->odom so downstream can compose robustly.
    pose["transform_odom_from_cam"] = T_cam_to_odom.tolist()
    pose["transform_pipeline"] = "demo_go2_literal_transform3d"
    pose["odom_quat_convention"] = "wxyz" if ODOM_QUAT_WXYZ else "xyzw"
    pose["init_odom_used"] = init_odom["position"] if isinstance(init_odom, dict) and "position" in init_odom else None

    T_raw_to_cam = None
    T_raw_to_odom = None
    if scale is not None and rot is not None and trans is not None:
        scale = np.asarray(scale, dtype=np.float64)
        rot = np.asarray(rot, dtype=np.float64)
        trans = np.asarray(trans, dtype=np.float64)
        T_raw_to_cam = build_T_raw_to_cam(scale, rot, trans)
        pose["transform_cam_from_raw"] = T_raw_to_cam.tolist()
        T_raw_to_odom = build_T_raw_to_odom(odom, scale, rot, trans, init_odom=init_odom)
        pose["transform_odom_from_raw"] = T_raw_to_odom.tolist()
        position, orientation = pose_from_transform_4x4(T_raw_to_odom)
        pose["initial_position"] = position
        pose["initial_orientation"] = orientation
    else:
        pose["transform_cam_from_raw"] = None
        pose["transform_odom_from_raw"] = None
        pose["initial_position"] = None
        pose["initial_orientation"] = None

    pose["frame"] = "z_up_odom"
    pose["mesh_frame"] = "sam3d_raw"
    with open(pose_path, "w") as f:
        json.dump(pose, f, indent=2)

    if DEBUG_DUMP:
        _write_transform_debug_dump(
            job_dir=job_dir,
            odom=odom,
            init_odom=init_odom,
            scale=scale if isinstance(scale, np.ndarray) else None,
            rot=rot if isinstance(rot, np.ndarray) else None,
            trans=trans if isinstance(trans, np.ndarray) else None,
            T_cam_to_odom=T_cam_to_odom,
            T_raw_to_cam=T_raw_to_cam,
            T_raw_to_odom=T_raw_to_odom,
        )

    if DEBUG_TRANSFORM:
        job_id = pose.get("job_id", job_dir.name)
        print("[DEBUG_TRANSFORM] job_id=%s odom_pos=%s initial_position=%s" % (job_id, odom.get("position"), pose.get("initial_position")))
    return True
