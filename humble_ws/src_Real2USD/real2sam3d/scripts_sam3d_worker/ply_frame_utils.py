"""
PLY helpers for SAM3D worker.

- vertices_y_up_to_z_up: Convert vertices from Y-up (glTF/Blender) to Z-up (world/ROS).
- center_ply_to_origin: Read PLY, center at origin, optionally convert to Z-up, write simple PLY.
- transform_ply_camera_to_world_local: (Legacy) Transform camera-frame PLY to world
  then object-local using odom; kept for reference.

Frame convention (demo_go2-aligned). All conversions live here and in the worker only; downstream (e.g. simple_scene_buffer_node) applies pose as-is, no extra transform.
- Odom and point clouds use *Z-up* (x forward, y left, z up).
- object.glb is written in *Z-up at origin* (camera_frame_to_zup after transform_raw_glb_to_camera_frame with at_origin=True).
- pose.json stores go2_odom_* and initial_position / initial_orientation in *Z-up odom* (initial_pose_from_sam3d_output). Mesh frame is z_up_origin; placement pose is applied so v_odom = R @ v + t.

Quaternion conventions (easy to mix up):
- **Odom / ROS / scipy:** (x, y, z, w) = scalar last = *xyzw*. We use this for go2_odom_orientation.
- **SAM3D / PyTorch3D:** (w, x, y, z) = scalar first = *wxyz*. We use this for sam3d_rotation.
If your odom source uses wxyz, set env REAL2SAM3D_ODOM_QUAT_WXYZ=1 or pass odom_quat_wxyz=True where supported.
Requires: numpy, plyfile (pip install plyfile).
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# If set, go2_odom_orientation is (w,x,y,z); we convert to xyzw before building R.
ODOM_QUAT_WXYZ = os.environ.get("REAL2SAM3D_ODOM_QUAT_WXYZ", "").strip().lower() in ("1", "true", "yes")


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


def _quat_xyzw_to_rotation_matrix(q: List[float]) -> np.ndarray:
    """Convert quaternion [qx, qy, qz, qw] (scalar last, ROS/scipy) to 3x3 rotation matrix."""
    qx, qy, qz, qw = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float64)


def _quat_to_rotation_matrix(q: List[float], convention: Optional[str] = None) -> np.ndarray:
    """Convert quaternion to 3x3 rotation matrix.
    convention: 'xyzw' (ROS/scipy: scalar last) or 'wxyz' (scalar first).
    If None, uses ODOM_QUAT_WXYZ env (True -> wxyz).
    """
    q = [float(q[i]) for i in range(4)]
    if (convention or ("wxyz" if ODOM_QUAT_WXYZ else "xyzw")).lower() == "wxyz":
        q = [q[1], q[2], q[3], q[0]]  # wxyz -> xyzw
    return _quat_xyzw_to_rotation_matrix(q)


def build_world_T_camera(
    odom: dict,
    R_world_to_cam: Optional[np.ndarray] = None,
    t_world_to_camera: Optional[List[float]] = None,
    odom_quat_wxyz: Optional[bool] = None,
) -> np.ndarray:
    """
    Build 4x4 transform from camera frame to world (odom) frame.

    odom: dict with "position" [x,y,z] and "orientation" (xyzw by default, or wxyz if odom_quat_wxyz=True).
    R_world_to_cam: optional 3x3 world-to-camera rotation (when robot at identity).
    t_world_to_camera: optional [x,y,z] camera position in world when robot at identity.
    odom_quat_wxyz: if True, orientation is (w,x,y,z). If None, uses env REAL2SAM3D_ODOM_QUAT_WXYZ.
    When both R/t are None, odom is treated as camera pose. When set (e.g. go2),
    world_T_camera = [R_odom @ R_world_to_cam.T | R_odom @ t_world_to_camera + t_odom].
    """
    t_odom = np.array(odom["position"], dtype=np.float64)
    use_wxyz = odom_quat_wxyz if odom_quat_wxyz is not None else ODOM_QUAT_WXYZ
    R_odom = _quat_to_rotation_matrix(odom["orientation"], convention="wxyz" if use_wxyz else "xyzw")
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
    Transform mesh vertices from camera frame to world (odom) Z-up, then to object-local
    (centroid at origin). Use for GLB export so the mesh is in Z-up, consistent with go2 odom.

    vertices: (N, 3) in camera frame (x right, y down, z forward).
    odom: dict with "position" [x,y,z] and "orientation" [qx,qy,qz,qw] (ROS, Z-up).
    world_centroid: if provided, use it so GLB and PLY share the same origin.
    Returns (vertices_local, world_centroid) with vertices_local in Z-up odom frame at origin.
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

# --- demo_go2.py parity: raw GLB (Y-up) → camera frame (x right, y down, z forward) ---
# https://github.com/christopher-hsu/sam-3d-objects/blob/main/demo_go2.py
R_FLIP_Z = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64)
# Y-up (glTF) → Z-up (PyTorch3D): (x,y,z)_yup -> (-x, z, y)_zup
R_YUP_TO_ZUP = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float64)
# Pointmap convention [-X,-Y,Z] → camera (x right, y down, z forward)
R_PYTORCH3D_TO_CAM = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64)

# Camera (x right, y down, z forward) → Z-up world (x forward, y left, z up)
# Maps: camera right → world -Y, camera down → world -Z, camera forward → world X,
# so camera "up" (-Y) → world +Z. Row vectors: v_zup = v_cam @ R_CAM_TO_ZUP.T
R_CAM_TO_ZUP = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)


def _quat_wxyz_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Quaternion (w,x,y,z) to 3x3 rotation matrix (e.g. PyTorch3D / SAM3D)."""
    q = np.asarray(q, dtype=np.float64).ravel()
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def transform_raw_glb_to_camera_frame(
    vertices: np.ndarray,
    scale: np.ndarray,
    rotation_quat_wxyz: np.ndarray,
    translation: np.ndarray,
    at_origin: bool = True,
) -> np.ndarray:
    """
    Transform raw SAM3D GLB to camera frame (demo_go2.py transform_glb convention).

    Same chain as demo_go2: flip Z, Y-up→Z-up, scale, rotation, [optional] translation,
    then R_pytorch3d_to_cam. If at_origin=True we skip adding translation so the mesh
    stays at origin (SAM3D outputs at origin); pose then places it via initial_pose.
    """
    verts = np.asarray(vertices, dtype=np.float64)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError("vertices must be (N, 3)")
    scale = np.asarray(scale, dtype=np.float64).ravel()
    if scale.size == 1:
        scale = np.array([scale[0], scale[0], scale[0]], dtype=np.float64)
    else:
        scale = scale[:3]
    trans = np.asarray(translation, dtype=np.float64).ravel()[:3]
    R_quat = _quat_wxyz_to_rotation_matrix(rotation_quat_wxyz)

    # demo_go2 transform_mesh_vertices: flip Z, Y-up→Z-up, scale, rotate, translate
    verts = verts @ R_FLIP_Z.T
    verts = verts @ R_YUP_TO_ZUP.T
    verts = scale * verts
    verts = verts @ R_quat.T
    if not at_origin:
        verts = verts + trans
    # pointmap → camera frame (demo_go2 transform_glb)
    verts = verts @ R_PYTORCH3D_TO_CAM.T
    return verts


def camera_frame_to_zup(vertices: np.ndarray) -> np.ndarray:
    """
    Convert vertices from camera frame (x right, y down, z forward) to Z-up (x forward, y left, z up)
    so the mesh matches odom and point cloud convention. Use after transform_raw_glb_to_camera_frame.
    """
    verts = np.asarray(vertices, dtype=np.float64)
    return verts @ R_CAM_TO_ZUP.T


def initial_pose_from_sam3d_output(output: Dict[str, Any], odom: dict) -> Optional[Tuple[List[float], List[float]]]:
    """
    Compute object pose in Z-up odom from SAM3D output and robot odom (demo_go2 convention).

    SAM3D gives translation and rotation in pointmap/PyTorch3D frame. We convert both
    to camera (R_pytorch3d_to_cam), then to odom (build_world_T_camera; Z-up: x forward, y left, z up),
    so the initial pose matches the mesh from transform_raw_glb_to_camera_frame(..., at_origin=True)
    and is valid as initialization for registration.
    Returns (position [x,y,z], orientation [qx,qy,qz,qw]) in Z-up odom or None if output missing required keys.
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
    # Translation is in pointmap frame (same as demo_go2 transform_mesh_vertices).
    # Convert to camera: p_cam = R_pytorch3d_to_cam @ p_pointmap
    p_pointmap = np.asarray(trans, dtype=np.float64)
    p_cam = R_PYTORCH3D_TO_CAM @ p_pointmap

    T = build_world_T_camera(
        odom,
        R_world_to_cam=GO2_R_WORLD_TO_CAM,
        t_world_to_camera=GO2_T_WORLD_TO_CAMERA.tolist(),
    )
    p_cam_h = np.array([p_cam[0], p_cam[1], p_cam[2], 1.0], dtype=np.float64)
    position_odom = (T @ p_cam_h)[:3]

    if hasattr(rot, "cpu"):
        q = np.asarray(rot.cpu().numpy(), dtype=np.float64).ravel()
    else:
        q = np.asarray(rot, dtype=np.float64).ravel()
    if len(q) < 4:
        return None
    # Rotation from SAM3D is in pointmap frame (same as in transform_glb before R_pytorch3d_to_cam).
    # R_obj_cam = R_pytorch3d_to_cam @ R_obj_pointmap.
    # Mesh is exported in Z-up (camera_frame_to_zup), so pose orientation must be
    # R_obj_odom = R_cam_to_odom @ R_obj_cam @ R_CAM_TO_ZUP.T so that
    # v_odom = R_obj_odom @ v_zup + t matches R_cam_to_odom @ R_obj_cam @ v_cam + t
    # (with v_cam = v_zup @ R_CAM_TO_ZUP in row-vector convention).
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    R_obj_pointmap = _quat_to_rotation_matrix([x, y, z, w], convention="xyzw")  # SAM3D gives wxyz; we pass xyzw
    R_obj_cam = R_PYTORCH3D_TO_CAM @ R_obj_pointmap
    R_cam_to_odom = T[:3, :3]
    R_obj_odom = R_cam_to_odom @ R_obj_cam @ R_CAM_TO_ZUP.T
    from scipy.spatial.transform import Rotation as R_scipy
    quat_odom = R_scipy.from_matrix(R_obj_odom).as_quat()  # x,y,z,w
    return (position_odom.tolist(), quat_odom.tolist())


def demo_go2_transform_vertices_to_odom(
    vertices_raw_yup: np.ndarray,
    scale: np.ndarray,
    rotation_quat_wxyz: np.ndarray,
    translation: np.ndarray,
    odom: dict,
    init_odom: Optional[dict] = None,
    odom_quat_wxyz: Optional[bool] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply demo_go2 transform chain exactly (row-vector math) and return:
      (vertices_cam, vertices_odom)

    demo_go2 equivalent:
      verts_pointmap = transform_mesh_vertices(...)
      verts_cam = verts_pointmap @ R_pytorch3d_to_cam
      t_rel = odom.t - [init_odom.t[0], init_odom.t[1], 0]
      verts_world = verts_cam @ R_world_to_cam.T + t_world_to_camera
      verts_odom = verts_world @ R_world_to_odom.T + t_rel
    """
    # Use the same primitives as demo_go2: quaternion_to_matrix + Transform3d.
    try:
        import torch
        from pytorch3d.transforms import quaternion_to_matrix, Transform3d
    except ImportError as e:
        raise RuntimeError(
            "demo_go2 parity path requires torch + pytorch3d in this env."
        ) from e

    verts_np = np.asarray(vertices_raw_yup, dtype=np.float32)
    if verts_np.ndim != 2 or verts_np.shape[1] != 3:
        raise ValueError("vertices must be (N, 3)")

    vertices = torch.tensor(verts_np, dtype=torch.float32).unsqueeze(0)  # [1, N, 3]
    device = vertices.device
    vertices = vertices @ torch.tensor(R_FLIP_Z, dtype=torch.float32, device=device)
    vertices = vertices @ torch.tensor(R_YUP_TO_ZUP, dtype=torch.float32, device=device)

    s_np = np.asarray(scale, dtype=np.float32).ravel()
    if s_np.size == 1:
        s_np = np.array([s_np[0], s_np[0], s_np[0]], dtype=np.float32)
    else:
        s_np = s_np[:3]
    q_wxyz = torch.tensor(np.asarray(rotation_quat_wxyz, dtype=np.float32).ravel()[:4], dtype=torch.float32, device=device).unsqueeze(0)
    t_np = np.asarray(translation, dtype=np.float32).ravel()[:3]
    t_t = torch.tensor(t_np, dtype=torch.float32, device=device)

    R_mat = quaternion_to_matrix(q_wxyz)
    tfm = Transform3d(dtype=vertices.dtype, device=device)
    tfm = tfm.scale(torch.tensor(s_np, dtype=torch.float32, device=device)).rotate(R_mat).translate(
        float(t_t[0]), float(t_t[1]), float(t_t[2])
    )
    verts_pointmap = tfm.transform_points(vertices)[0]
    verts_cam_t = verts_pointmap @ torch.tensor(R_PYTORCH3D_TO_CAM, dtype=torch.float32, device=device)
    verts_cam = verts_cam_t.cpu().numpy().astype(np.float64)

    # 3) demo_go2 transform_glb_to_world
    t_odom = np.array(odom["position"], dtype=np.float64)
    if init_odom is not None:
        ti = np.array(init_odom["position"], dtype=np.float64)
        t_odom = t_odom - np.array([ti[0], ti[1], 0.0], dtype=np.float64)
    use_wxyz = odom_quat_wxyz if odom_quat_wxyz is not None else ODOM_QUAT_WXYZ
    R_odom = _quat_to_rotation_matrix(odom["orientation"], convention="wxyz" if use_wxyz else "xyzw")
    verts_world = verts_cam @ GO2_R_WORLD_TO_CAM.T + GO2_T_WORLD_TO_CAMERA
    verts_odom = verts_world @ R_odom.T + t_odom
    return verts_cam, verts_odom


def demo_go2_build_T_raw_to_odom(
    scale: np.ndarray,
    rotation_quat_wxyz: np.ndarray,
    translation: np.ndarray,
    odom: dict,
    init_odom: Optional[dict] = None,
    odom_quat_wxyz: Optional[bool] = None,
) -> np.ndarray:
    """
    Build 4x4 raw->odom by mapping basis points through demo_go2_transform_vertices_to_odom.
    This avoids any algebraic derivation and guarantees parity with the row-vector vertex path.
    """
    basis = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    _cam, mapped = demo_go2_transform_vertices_to_odom(
        basis,
        scale=scale,
        rotation_quat_wxyz=rotation_quat_wxyz,
        translation=translation,
        odom=odom,
        init_odom=init_odom,
        odom_quat_wxyz=odom_quat_wxyz,
    )
    p0 = mapped[0]
    px = mapped[1] - p0
    py = mapped[2] - p0
    pz = mapped[3] - p0
    T = np.eye(4)
    T[:3, :3] = np.column_stack([px, py, pz])
    T[:3, 3] = p0
    return T


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
