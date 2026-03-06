"""
Standalone GLB export: write a joint scene GLB from a list of scene-graph-style objects.
No ROS/rclpy dependency — use this from evaluations or other non-ROS scripts.

Same behavior as simple_scene_buffer_node.write_joint_glb_from_list_standalone.
Optionally includes accumulated RealSense (or lidar) point cloud from run_dir when provided.
"""

from pathlib import Path
from typing import Any, List, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

try:
    from real2sam3d.ply_frame_utils import R_flip_z, R_yup_to_zup
    T_RAW_TO_ZUP = np.eye(4, dtype=np.float64)
    T_RAW_TO_ZUP[:3, :3] = (np.asarray(R_flip_z) @ np.asarray(R_yup_to_zup)).T
except Exception:
    T_RAW_TO_ZUP = np.eye(4, dtype=np.float64)

# Subdirs under run_dir where we look for realsense/lidar pointcloud npy (in order)
_POINTCLOUD_SUBDIRS = [
    Path("diagnostics") / "pointclouds" / "realsense",
    Path("pointscloud") / "realsense",
    Path("pointcloud") / "realsense",
]


def _pose_to_matrix(position_xyz, quat_xyzw):
    """Build 4x4 transform from position (3,) and quaternion (x,y,z,w)."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_quat(np.asarray(quat_xyzw, dtype=np.float64).ravel()[:4]).as_matrix()
    T[:3, 3] = np.asarray(position_xyz, dtype=np.float64).ravel()[:3]
    return T


def _voxel_downsample_points(pts: np.ndarray, voxel_size: float) -> np.ndarray:
    """Reduce point cloud by keeping one point per voxel. Uses open3d if available (faster), else numpy."""
    if voxel_size <= 0 or pts is None or len(pts) == 0:
        return pts
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2:
        return pts
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd = pcd.voxel_down_sample(voxel_size)
        return np.asarray(pcd.points, dtype=np.float64)
    except Exception:
        pass
    # Numpy fallback: one point per voxel
    voxel_idx = np.floor(pts[:, :3] / voxel_size).astype(np.int64)
    order = np.lexsort((voxel_idx[:, 2], voxel_idx[:, 1], voxel_idx[:, 0]))
    voxel_idx_ordered = voxel_idx[order]
    keep = np.ones(len(voxel_idx_ordered), dtype=bool)
    keep[1:] = np.any(voxel_idx_ordered[1:] != voxel_idx_ordered[:-1], axis=1)
    return pts[order[keep]]


def _load_accumulated_pointcloud_npy(
    run_dir: Optional[Path] = None,
    pointcloud_dir: Optional[Path] = None,
    voxel_size: Optional[float] = 0.02,
    use_all_files: bool = False,
) -> Optional[np.ndarray]:
    """Load accumulated RealSense point cloud.
    Each realsense_*.npy file is already a full accumulated point cloud (periodic or shutdown snapshot).
    - If pointcloud_dir is set: look only in that directory for realsense_*.npy (use when you have the realsense folder path).
    - Else if run_dir is set: look in run_dir/diagnostics/pointclouds/realsense, run_dir/pointscloud/realsense, run_dir/pointcloud/realsense.
    - If use_all_files False (default): use the single file with the most points (usually shutdown or last periodic).
    - If use_all_files True: load and concatenate all realsense_*.npy, then voxel-downsample to merge duplicates.
    If voxel_size > 0, downsampling is applied to speed up GLB export (default 0.02 = 2cm).
    Returns Nx3 float64 in odom frame, or None if not found.
    """
    all_loaded: List[tuple] = []
    if pointcloud_dir is not None:
        d = Path(pointcloud_dir)
        if d.is_dir():
            files = sorted(d.glob("realsense_*.npy"), key=lambda p: p.stat().st_mtime)
            for p in files:
                try:
                    pts = np.load(str(p), allow_pickle=False)
                    if hasattr(pts, "shape") and len(pts.shape) >= 2 and pts.shape[-1] >= 3:
                        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
                        all_loaded.append((p, pts))
                except Exception:
                    continue
    if not all_loaded and run_dir is not None:
        run_dir = Path(run_dir)
        for sub in _POINTCLOUD_SUBDIRS:
            d = run_dir / sub
            if not d.is_dir():
                continue
            files = sorted(d.glob("realsense_*.npy"), key=lambda p: p.stat().st_mtime)
            for p in files:
                try:
                    pts = np.load(str(p), allow_pickle=False)
                    if hasattr(pts, "shape") and len(pts.shape) >= 2 and pts.shape[-1] >= 3:
                        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
                        all_loaded.append((p, pts))
                except Exception:
                    continue
            if all_loaded:
                break
    if not all_loaded:
        return None
    if use_all_files and len(all_loaded) > 1:
        best_pts = np.vstack([pts for _, pts in all_loaded])
    else:
        # Single file: use the one with most points (typically shutdown or last periodic)
        best_pts = max(all_loaded, key=lambda x: x[1].shape[0])[1]
    if voxel_size and voxel_size > 0 and len(best_pts) > 1000:
        best_pts = _voxel_downsample_points(best_pts, voxel_size)
    return best_pts


def _load_meshes_from_path(data_path: str) -> Optional[List[Any]]:
    """Load trimesh geometry from data_path (GLB or PLY). Returns list of Trimesh or None."""
    try:
        import trimesh
    except ImportError:
        return None
    path = Path(data_path)
    if not path.exists():
        return None
    suffix = path.suffix.lower()
    if suffix == ".ply":
        glb = path.parent / "object.glb"
        if glb.exists():
            path = glb
            suffix = ".glb"
    if suffix != ".glb":
        return None
    try:
        # Fast path: single-mesh GLB (common for object.glb) avoids Scene parsing
        single = trimesh.load(str(path), process=False, force="mesh")
        if isinstance(single, trimesh.Trimesh):
            return [single]
    except Exception:
        pass
    try:
        scene = trimesh.load(str(path), process=False)
        if isinstance(scene, trimesh.Scene):
            meshes = []
            for node_name in scene.graph.nodes_geometry:
                try:
                    node_tf, geom_name = scene.graph[node_name]
                    geom = scene.geometry.get(geom_name)
                    if isinstance(geom, trimesh.Trimesh):
                        g = geom.copy()
                        g.apply_transform(np.asarray(node_tf, dtype=np.float64))
                        meshes.append(g)
                except Exception:
                    continue
            return meshes if meshes else None
        if isinstance(scene, trimesh.Trimesh):
            return [scene]
    except Exception:
        pass
    return None


def write_joint_glb_from_list_standalone(
    glb_path: Path,
    scene_list: List[dict],
    run_dir: Optional[Path] = None,
    realsense_voxel_size: Optional[float] = 0.02,
    realsense_use_all_files: bool = False,
    realsense_z_offset: float = 0.0,
    log=None,
) -> None:
    """Write joint GLB from scene_list. No rclpy. Each item: data_path, position, orientation; optional transform_odom_from_raw (4x4).
    If run_dir is set, loads accumulated RealSense point cloud from run_dir and adds it to the scene (same odom frame).
    realsense_voxel_size: voxel size in meters for downsampling the point cloud (faster export). Default 0.02 (2cm). Set None to disable.
    realsense_use_all_files: if True, concatenate all realsense_*.npy files then downsample; if False (default), use the single file with most points.
    realsense_z_offset: translation in meters applied to point cloud z (e.g. -0.2 to move down by 0.2). Default 0.
    """
    try:
        import trimesh
    except ImportError:
        if log:
            log.warn("trimesh not available; skipping joint GLB")
        return
    scene = trimesh.Scene()
    added = 0
    for item in scene_list:
        path = item.get("data_path")
        if not path:
            continue
        T_raw = item.get("transform_odom_from_raw")
        if T_raw is not None:
            T = np.array(T_raw, dtype=np.float64)
            if T.shape != (4, 4):
                T = np.eye(4, dtype=np.float64)
        else:
            pos = np.array(item["position"], dtype=np.float64)
            quat = np.array(item["orientation"], dtype=np.float64)
            T = _pose_to_matrix(pos, quat) @ T_RAW_TO_ZUP
        oid = item.get("id", added)
        meshes = _load_meshes_from_path(path)
        if not meshes:
            continue
        for i, mesh in enumerate(meshes):
            name = "obj_%s" % oid if len(meshes) == 1 else "obj_%s_%d" % (oid, i)
            scene.add_geometry(mesh.copy(), transform=T, node_name=name)
            added += 1

    # Optionally add accumulated RealSense point cloud from run_dir (same odom frame)
    if run_dir is not None:
        pts = _load_accumulated_pointcloud_npy(
            run_dir, voxel_size=realsense_voxel_size, use_all_files=realsense_use_all_files
        )
        if pts is not None and pts.size > 0:
            try:
                pts = np.asarray(pts, dtype=np.float64)
                if realsense_z_offset != 0.0:
                    pts = pts.copy()
                    pts[:, 2] += realsense_z_offset
                pc = trimesh.points.PointCloud(vertices=pts)
                scene.add_geometry(pc, transform=np.eye(4), node_name="realsense_accumulated")
                if log:
                    msg = "Included accumulated RealSense point cloud (%d points)" % len(pts)
                    if realsense_z_offset != 0.0:
                        msg += " z_offset=%.2fm" % realsense_z_offset
                    if realsense_use_all_files:
                        msg += " (merged all realsense_*.npy)"
                    if realsense_voxel_size and realsense_voxel_size > 0:
                        msg += " voxel-downsampled at %.0fmm" % (realsense_voxel_size * 1000)
                    log.info(msg + " from %s" % run_dir)
            except Exception as e:
                if log:
                    log.warn("Failed to add RealSense point cloud to scene: %s" % e)

    if added == 0 and not scene.geometry:
        return
    glb_path = Path(glb_path)
    glb_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(glb_path))
    if log:
        log.info("Wrote joint GLB %s (%d meshes)" % (glb_path, added))
