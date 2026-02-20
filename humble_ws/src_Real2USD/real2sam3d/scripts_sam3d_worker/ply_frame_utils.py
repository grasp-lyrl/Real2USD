"""
PLY helpers for SAM3D worker.

- vertices_y_up_to_z_up: Convert vertices from Y-up (glTF/Blender) to Z-up (world/ROS).
- center_ply_to_origin: Read PLY, center at origin, optionally convert to Z-up, write simple PLY.
- transform_ply_camera_to_world_local: (Legacy) Transform camera-frame PLY to world
  then object-local using odom; kept for reference.

Requires: numpy, plyfile (pip install plyfile).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def vertices_y_up_to_z_up(vertices: np.ndarray) -> np.ndarray:
    """
    Convert vertices from Y-up (glTF/Blender: X right, Y up, Z back) to Z-up world frame
    (X right, Y forward, Z up). So: (x, y, z)_yup -> (x, -z, y)_zup.
    vertices: (N, 3). Returns (N, 3) in Z-up.
    """
    v = np.asarray(vertices, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError("vertices must be (N, 3)")
    return np.column_stack([v[:, 0], -v[:, 2], v[:, 1]])


def _quat_to_rotation_matrix(q: List[float]) -> np.ndarray:
    """Convert quaternion [qx, qy, qz, qw] to 3x3 rotation matrix."""
    qx, qy, qz, qw = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float64)


def build_world_T_camera(
    odom: dict,
    R_world_to_cam: Optional[np.ndarray] = None,
    t_world_to_camera: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Build 4x4 transform from camera frame to world (odom) frame.

    odom: dict with "position" [x,y,z] and "orientation" [qx,qy,qz,qw] (ROS).
    R_world_to_cam: optional 3x3 world-to-camera rotation (when robot at identity).
    t_world_to_camera: optional [x,y,z] camera position in world when robot at identity.
    When both are None, odom is treated as camera pose. When set (e.g. go2),
    world_T_camera = [R_odom @ R_world_to_cam.T | R_odom @ t_world_to_camera + t_odom].
    """
    t_odom = np.array(odom["position"], dtype=np.float64)
    R_odom = _quat_to_rotation_matrix(odom["orientation"])
    if R_world_to_cam is None and t_world_to_camera is None:
        R = R_odom
        t = t_odom
    else:
        R_w2c = np.asarray(R_world_to_cam, dtype=np.float64) if R_world_to_cam is not None else np.eye(3)
        t_cam = np.asarray(t_world_to_camera or [0, 0, 0], dtype=np.float64)
        R = R_odom @ R_w2c.T
        t = R_odom @ t_cam + t_odom
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def transform_vertices_camera_to_world_local(
    vertices: np.ndarray,
    odom: dict,
    world_centroid: Optional[np.ndarray] = None,
    R_world_to_cam: Optional[np.ndarray] = None,
    t_world_to_camera: Optional[List[float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform mesh vertices from camera frame to world, then to object-local
    (centroid at origin). Use for GLB/mesh export so it matches the PLY frame.

    vertices: (N, 3) in camera frame.
    world_centroid: if provided (e.g. from PLY transform), use it so GLB and PLY share the same origin.
    Returns (vertices_local, world_centroid).
    """
    verts = np.asarray(vertices, dtype=np.float64)
    if verts.ndim == 2 and verts.shape[1] == 3:
        ones = np.ones((verts.shape[0], 1), dtype=np.float64)
        verts_h = np.hstack([verts, ones])
    else:
        raise ValueError("vertices must be (N, 3)")
    world_T_cam = build_world_T_camera(odom, R_world_to_cam=R_world_to_cam, t_world_to_camera=t_world_to_camera)
    verts_world = (world_T_cam @ verts_h.T).T[:, :3]
    if world_centroid is None:
        world_centroid = np.mean(verts_world, axis=0)
    verts_local = verts_world - np.asarray(world_centroid, dtype=np.float64)
    return verts_local, world_centroid


# Unitree Go2 front camera (demo_go2 convention): world-to-camera rotation and camera position in world when robot at identity
GO2_R_WORLD_TO_CAM = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)
GO2_T_WORLD_TO_CAMERA = np.array([0.285, 0.0, 0.01], dtype=np.float64)


def initial_pose_from_sam3d_output(output: Dict[str, Any], odom: dict) -> Optional[Tuple[List[float], List[float]]]:
    """
    Compute object pose in odom frame from SAM3D output (translation, rotation in camera/pointmap frame) and robot odom.
    Matches demo_go2 transform: camera -> world -> odom.
    Returns (position [x,y,z], orientation [qx,qy,qz,qw]) or None if output missing required keys.
    """
    if "translation" not in output or "rotation" not in output:
        return None
    trans = output["translation"]
    rot = output["rotation"]
    if hasattr(trans, "cpu"):
        trans = np.asarray(trans.cpu().numpy(), dtype=np.float64).ravel()[:3]
    else:
        trans = np.asarray(trans, dtype=np.float64).ravel()[:3]
    if len(trans) < 3:
        return None
    T = build_world_T_camera(
        odom,
        R_world_to_cam=GO2_R_WORLD_TO_CAM,
        t_world_to_camera=GO2_T_WORLD_TO_CAMERA.tolist(),
    )
    p_cam_h = np.array([trans[0], trans[1], trans[2], 1.0], dtype=np.float64)
    position_odom = (T @ p_cam_h)[:3]

    if hasattr(rot, "cpu"):
        q = np.asarray(rot.cpu().numpy(), dtype=np.float64).ravel()
    else:
        q = np.asarray(rot, dtype=np.float64).ravel()
    if len(q) < 4:
        return None
    # PyTorch3D / SAM3D use w,x,y,z; _quat_to_rotation_matrix expects [qx,qy,qz,qw]
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    R_obj_cam = _quat_to_rotation_matrix([x, y, z, w])
    R_cam_to_odom = T[:3, :3]
    R_obj_odom = R_cam_to_odom @ R_obj_cam
    from scipy.spatial.transform import Rotation as R_scipy
    quat_odom = R_scipy.from_matrix(R_obj_odom).as_quat()  # x,y,z,w
    return (position_odom.tolist(), quat_odom.tolist())


def _get_element_names(ply) -> List[str]:
    """Return list of element names (plyfile uses .elements, not .element_names)."""
    try:
        from plyfile import PlyData
    except ImportError:
        return []
    if not isinstance(ply, PlyData):
        return []
    return [el.name for el in ply.elements]


def _find_vertex_element(ply) -> Optional[str]:
    """Find first element that has 'x', 'y', 'z' properties (e.g. 'vertex')."""
    for name in _get_element_names(ply):
        el = ply[name]
        if hasattr(el.data, 'dtype') and el.data.dtype.names:
            names = set(el.data.dtype.names)
            if 'x' in names and 'y' in names and 'z' in names:
                return name
    return None


def center_ply_to_origin(
    ply_path: Path,
    out_path: Optional[Path] = None,
    z_up: bool = True,
) -> np.ndarray:
    """
    Read PLY (e.g. SAM3D gaussian splat), center vertices at origin, and write
    a simple PLY with only vertex x,y,z (and optional red,green,blue). No odom
    or world transform; object is at origin for later registration.

    If z_up=True (default), convert from Y-up to Z-up world frame before writing.
    Sanitizes NaN/inf so the output is readable by Open3D and other tools.
    When out_path is the same as ply_path, writes to a temp file then replaces.
    Returns centroid (3,) of the original points (before centering), in same frame as written.
    """
    try:
        from plyfile import PlyData, PlyElement
    except ImportError as e:
        raise RuntimeError(
            "plyfile is required. Install with: pip install plyfile"
        ) from e

    ply_path = Path(ply_path)
    out_path = Path(out_path) if out_path is not None else ply_path

    ply = PlyData.read(str(ply_path))
    vertex_key = _find_vertex_element(ply)
    if vertex_key is None:
        raise ValueError(
            f"PLY {ply_path} has no element with x,y,z. "
            f"Elements: {_get_element_names(ply)}"
        )

    vert = ply[vertex_key].data
    x = np.asarray(vert["x"], dtype=np.float64).copy()
    y = np.asarray(vert["y"], dtype=np.float64).copy()
    z = np.asarray(vert["z"], dtype=np.float64).copy()
    pts = np.stack([x, y, z], axis=1)

    # Sanitize: replace nan/inf so we don't write broken PLY
    bad = ~np.isfinite(pts)
    if np.any(bad):
        pts = np.where(bad, 0.0, pts)

    centroid = np.mean(pts, axis=0)
    pts_local = pts - centroid
    if z_up:
        pts_local = vertices_y_up_to_z_up(pts_local)

    names = set(vert.dtype.names) if vert.dtype.names else set()
    has_rgb = "red" in names and "green" in names and "blue" in names
    if has_rgb:
        r = np.asarray(vert["red"]).ravel()
        g = np.asarray(vert["green"]).ravel()
        b = np.asarray(vert["blue"]).ravel()
        if r.size != pts_local.shape[0]:
            has_rgb = False
        elif np.issubdtype(r.dtype, np.floating):
            r = (np.clip(r, 0, 1) * 255).astype(np.uint8)
            g = (np.clip(g, 0, 1) * 255).astype(np.uint8)
            b = (np.clip(b, 0, 1) * 255).astype(np.uint8)
        else:
            r = r.astype(np.uint8)
            g = g.astype(np.uint8)
            b = b.astype(np.uint8)

    if has_rgb:
        dtype = [
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
            ("red", np.uint8),
            ("green", np.uint8),
            ("blue", np.uint8),
        ]
        data = np.empty(pts_local.shape[0], dtype=dtype)
        data["x"] = pts_local[:, 0].astype(np.float32)
        data["y"] = pts_local[:, 1].astype(np.float32)
        data["z"] = pts_local[:, 2].astype(np.float32)
        data["red"] = r
        data["green"] = g
        data["blue"] = b
    else:
        dtype = [("x", np.float32), ("y", np.float32), ("z", np.float32)]
        data = np.empty(pts_local.shape[0], dtype=dtype)
        data["x"] = pts_local[:, 0].astype(np.float32)
        data["y"] = pts_local[:, 1].astype(np.float32)
        data["z"] = pts_local[:, 2].astype(np.float32)

    el = PlyElement.describe(data, "vertex")
    # When overwriting the source file, write to temp then replace so we never truncate the file
    same_file = out_path.resolve() == ply_path.resolve()
    if same_file:
        import os
        import tempfile
        fd, tmp_path = tempfile.mkstemp(suffix=".ply", dir=out_path.parent, prefix=".center_ply_")
        try:
            os.close(fd)
            PlyData([el], text=True).write(tmp_path)
            os.replace(tmp_path, str(out_path))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    else:
        PlyData([el], text=True).write(str(out_path))

    return centroid


def transform_ply_camera_to_world_local(
    ply_path: Path,
    odom: dict,
    out_path: Optional[Path] = None,
    R_world_to_cam: Optional[np.ndarray] = None,
    t_world_to_camera: Optional[List[float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read PLY (camera-frame vertices), transform to world, then to object-local
    (centroid at origin). Overwrite or write to out_path. Preserves all vertex
    properties; only x,y,z are transformed.

    Optional R_world_to_cam and t_world_to_camera implement the same camera-offset
    chain as demo_go2 transform_glb_to_world (e.g. for go2: t_world_to_camera
    is the camera position in world when robot is at origin).

    Returns:
        world_centroid: (3,) position of object center in world (for pose.json).
        local_vertices: (N,3) vertices in object-local frame (for verification).
    """
    try:
        from plyfile import PlyData
    except ImportError as e:
        raise RuntimeError(
            "plyfile is required for camera→world PLY transform. "
            "Install with: pip install plyfile"
        ) from e

    ply_path = Path(ply_path)
    out_path = Path(out_path) if out_path is not None else ply_path

    ply = PlyData.read(str(ply_path))
    vertex_key = _find_vertex_element(ply)
    if vertex_key is None:
        raise ValueError(
            f"PLY {ply_path} has no element with x,y,z properties. "
            f"Elements: {_get_element_names(ply)}"
        )

    vert = ply[vertex_key].data
    x = np.asarray(vert['x'], dtype=np.float64)
    y = np.asarray(vert['y'], dtype=np.float64)
    z = np.asarray(vert['z'], dtype=np.float64)
    pts_cam = np.stack([x, y, z], axis=1)  # (N, 3)

    world_T_cam = build_world_T_camera(
        odom,
        R_world_to_cam=R_world_to_cam,
        t_world_to_camera=t_world_to_camera,
    )
    ones = np.ones((pts_cam.shape[0], 1), dtype=np.float64)
    pts_cam_h = np.hstack([pts_cam, ones])  # (N, 4)
    pts_world = (world_T_cam @ pts_cam_h.T).T[:, :3]  # (N, 3)

    world_centroid = np.mean(pts_world, axis=0)
    pts_local = pts_world - world_centroid

    # Update vertex x,y,z in place; preserve all other elements and format
    vert['x'] = pts_local[:, 0]
    vert['y'] = pts_local[:, 1]
    vert['z'] = pts_local[:, 2]
    ply.write(str(out_path))

    return world_centroid, pts_local
