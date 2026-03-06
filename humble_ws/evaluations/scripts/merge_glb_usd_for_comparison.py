#!/usr/bin/env python3
"""
Merge all GLB and USD files in a directory into one GLB for side-by-side comparison
(size, shape, etc.). USD is converted to mesh and placed alongside GLBs in a single file.

Usage (from evaluations/ or with PYTHONPATH including real2sam3d if you use pxr fallback):

  python scripts/merge_glb_usd_for_comparison.py --input-dir chairs --out chairs/comparison.glb

  # Optional: spacing between models (meters), default 2.0
  python scripts/merge_glb_usd_for_comparison.py --input-dir chairs --out chairs/comparison.glb --spacing 3.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Any

import numpy as np
from scipy.spatial.transform import Rotation as R

# Z-up (USD) to Y-up (GLB): -90° around X then 180° around X so the chair sits upright
# (Without the 180°, the USD chair ends up upside-down.)
_R_ZUP_TO_YUP = np.eye(4, dtype=np.float64)
_R_ZUP_TO_YUP[1, 1] = 0
_R_ZUP_TO_YUP[1, 2] = -1
_R_ZUP_TO_YUP[2, 1] = 1
_R_ZUP_TO_YUP[2, 2] = 0
_R_180_X = np.eye(4, dtype=np.float64)
_R_180_X[1, 1] = -1
_R_180_X[2, 2] = -1
_T_ZUP_TO_YUP = _R_180_X @ _R_ZUP_TO_YUP

# Depth unproject: same as sam3d_glb_registration_bridge (Unitree Go2 camera in odom)
_T_CAM_IN_ODOM = np.array([0.285, 0.0, 0.01], dtype=np.float64)
_GROUND_PLANE_HEIGHT = 0.1
_DEPTH_LEFT_X = -3.0  # place depth point cloud to the left of USD (origin)

_SCRIPT_DIR = Path(__file__).resolve().parent
_EVAL_ROOT = _SCRIPT_DIR.parent
if str(_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_EVAL_ROOT))


def _load_glb_meshes(path: Path):
    """Load all meshes from a GLB as list of (trimesh.Trimesh, transform 4x4 or None)."""
    try:
        import trimesh
    except ImportError:
        return None
    if not path.exists() or path.suffix.lower() != ".glb":
        return None
    try:
        single = trimesh.load(str(path), process=False, force="mesh")
        if isinstance(single, trimesh.Trimesh):
            return [(single, np.eye(4))]
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
                        meshes.append((geom, np.asarray(node_tf, dtype=np.float64)))
                except Exception:
                    continue
            return meshes if meshes else None
        if isinstance(scene, trimesh.Trimesh):
            return [(scene, np.eye(4))]
    except Exception:
        pass
    return None


def _usd_meshes_via_trimesh(path: Path) -> Optional[List[Tuple[Any, np.ndarray]]]:
    """Load USD via trimesh; return list of (mesh, 4x4 transform) or None."""
    try:
        import trimesh
    except ImportError:
        return None
    if not path.exists() or path.suffix.lower() not in (".usd", ".usda", ".usdc"):
        return None
    try:
        scene = trimesh.load_scene(str(path))
        if scene is None or not hasattr(scene, "geometry"):
            return None
        meshes = []
        if hasattr(scene, "graph") and hasattr(scene.graph, "nodes_geometry"):
            for node_name in scene.graph.nodes_geometry:
                try:
                    node_tf, geom_name = scene.graph[node_name]
                    geom = scene.geometry.get(geom_name)
                    if isinstance(geom, trimesh.Trimesh):
                        meshes.append((geom, np.asarray(node_tf, dtype=np.float64)))
                except Exception:
                    continue
        else:
            for geom in getattr(scene, "geometry", {}).values():
                if isinstance(geom, trimesh.Trimesh):
                    meshes.append((geom, np.eye(4)))
        return meshes if meshes else None
    except Exception:
        return None


def _usd_meshes_via_pxr(path: Path) -> Optional[List[Tuple[Any, np.ndarray]]]:
    """Load USD via pxr (UsdGeom), build trimesh per mesh. Returns list of (trimesh, 4x4)."""
    try:
        from pxr import Usd, UsdGeom
        import trimesh
    except ImportError:
        return None
    if not path.exists() or path.suffix.lower() not in (".usd", ".usda", ".usdc"):
        return None
    try:
        stage = Usd.Stage.Open(str(path))
        if not stage:
            return None
        meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
        result = []
        for prim in stage.TraverseAll():
            if not prim.IsA(UsdGeom.Mesh):
                continue
            mesh_prim = UsdGeom.Mesh(prim)
            points_attr = mesh_prim.GetPointsAttr()
            if not points_attr:
                continue
            points = np.array(points_attr.Get(), dtype=np.float64) * meters_per_unit
            if len(points) == 0:
                continue
            xform = UsdGeom.Xformable(prim)
            world = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            m = np.eye(4, dtype=np.float64)
            for i in range(4):
                for j in range(4):
                    m[i, j] = world[i][j]
            counts_attr = mesh_prim.GetFaceVertexCountsAttr()
            indices_attr = mesh_prim.GetFaceVertexIndicesAttr()
            if not counts_attr or not indices_attr:
                continue
            counts = list(counts_attr.Get())
            indices = list(indices_attr.Get())
            # Triangulate: each face with 3 verts -> one tri; 4 verts -> two tris
            tri_faces = []
            idx = 0
            for c in counts:
                if c == 3:
                    tri_faces.append([indices[idx], indices[idx + 1], indices[idx + 2]])
                    idx += 3
                elif c == 4:
                    tri_faces.append([indices[idx], indices[idx + 1], indices[idx + 2]])
                    tri_faces.append([indices[idx], indices[idx + 2], indices[idx + 3]])
                    idx += 4
                else:
                    # Fan triangulation for n-gons
                    for k in range(1, c - 1):
                        tri_faces.append([indices[idx], indices[idx + k], indices[idx + k + 1]])
                    idx += c
            if not tri_faces:
                continue
            faces = np.array(tri_faces, dtype=np.int32)
            tm = trimesh.Trimesh(vertices=points, faces=faces)
            result.append((tm, m))
        return result if result else None
    except Exception:
        return None


def _voxel_downsample_points(pts: np.ndarray, voxel_size: float) -> np.ndarray:
    """Reduce point cloud by keeping one point per voxel (numpy fallback, no open3d)."""
    if voxel_size <= 0 or pts is None or len(pts) == 0:
        return pts
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2:
        return pts
    voxel_idx = np.floor(pts[:, :3] / voxel_size).astype(np.int64)
    order = np.lexsort((voxel_idx[:, 2], voxel_idx[:, 1], voxel_idx[:, 0]))
    voxel_idx_ordered = voxel_idx[order]
    keep = np.ones(len(voxel_idx_ordered), dtype=bool)
    keep[1:] = np.any(voxel_idx_ordered[1:] != voxel_idx_ordered[:-1], axis=1)
    return pts[order[keep]]


def _load_depth_point_cloud_from_job_dir(
    job_dir: Path,
    use_mask: bool = True,
) -> Optional[np.ndarray]:
    """
    Build point cloud in odom frame (Z-up) from depth.npy + meta.json + optional mask.png.
    Same logic as sam3d_glb_registration_bridge_node._segment_point_cloud_from_job_dir_with_reason.
    Returns Nx3 float64 or None.
    """
    depth_path = job_dir / "depth.npy"
    meta_path = job_dir / "meta.json"
    if not depth_path.exists() or not meta_path.exists():
        return None
    try:
        depth_crop = np.load(str(depth_path)).astype(np.float64)
    except Exception:
        return None
    if depth_crop.size == 0:
        return None
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except Exception:
        return None
    crop_bbox = meta.get("crop_bbox")
    cam = meta.get("camera_info") or {}
    odom = meta.get("odometry") or {}
    if not (crop_bbox and len(crop_bbox) >= 4):
        return None
    k_flat = cam.get("k") or cam.get("K")
    if k_flat is None or len(k_flat) != 9:
        return None
    if odom.get("position") is None:
        return None

    x_min, y_min = int(crop_bbox[0]), int(crop_bbox[1])
    K = np.array(k_flat, dtype=np.float64).reshape(3, 3)
    mask_crop = None
    if use_mask:
        mask_path = job_dir / "mask.png"
        if mask_path.exists():
            try:
                import cv2
                mask_crop = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            except Exception:
                pass
        if mask_crop is not None and mask_crop.size > 0:
            if mask_crop.shape != depth_crop.shape:
                try:
                    import cv2
                    mask_crop = cv2.resize(
                        mask_crop, (depth_crop.shape[1], depth_crop.shape[0]), interpolation=cv2.INTER_NEAREST
                    )
                except Exception:
                    mask_crop = None
            if mask_crop is not None:
                vc, uc = np.where((mask_crop > 0) & (depth_crop > 1e-6))
            else:
                vc, uc = np.where(depth_crop > 1e-6)
        else:
            vc, uc = np.where(depth_crop > 1e-6)
    else:
        vc, uc = np.where(depth_crop > 1e-6)
    if len(vc) == 0:
        return None
    u_full = (x_min + uc).astype(np.float64)
    v_full = (y_min + vc).astype(np.float64)
    Z = depth_crop[vc, uc]
    uv1 = np.stack([u_full, v_full, np.ones_like(u_full, dtype=np.float64)], axis=0)
    xyz_cam = (np.linalg.inv(K) @ uv1) * Z
    points_cam = xyz_cam.T

    R_static_odom_to_cam = R.from_euler("xyz", np.radians([-90, 0, 90]), degrees=False).as_matrix()
    R_additional = R.from_euler("xyz", np.radians([0, 0, 180]), degrees=False).as_matrix()
    R_odom_to_cam = R_additional @ R_static_odom_to_cam
    T_odom_from_cam = np.eye(4, dtype=np.float64)
    T_odom_from_cam[:3, :3] = R_odom_to_cam
    T_odom_from_cam[:3, 3] = _T_CAM_IN_ODOM
    t = np.array(odom["position"], dtype=np.float64)
    q = np.array(odom["orientation"], dtype=np.float64)
    R_world_from_odom = R.from_quat(q).as_matrix()
    T_world_from_odom = np.eye(4, dtype=np.float64)
    T_world_from_odom[:3, :3] = R_world_from_odom
    T_world_from_odom[:3, 3] = t
    T_world_from_cam = T_world_from_odom @ T_odom_from_cam
    ones = np.ones((points_cam.shape[0], 1), dtype=np.float64)
    h = np.hstack([points_cam, ones]).T
    points_odom = (T_world_from_cam @ h).T[:, :3]
    points_odom = points_odom[points_odom[:, 2] > 0]
    if len(points_odom) < 30:
        return None
    points_odom = points_odom[points_odom[:, 2] > _GROUND_PLANE_HEIGHT]
    if len(points_odom) < 30:
        return None
    return points_odom.astype(np.float64)


def _centroid_of_transformed_meshes(pairs: List[Tuple[Any, np.ndarray]]) -> np.ndarray:
    """Compute centroid (3,) of all mesh vertices in world space for the given (mesh, 4x4) pairs."""
    all_pts = []
    for mesh, tf in pairs:
        R = tf[:3, :3]
        t = tf[:3, 3]
        pts = np.asarray(mesh.vertices, dtype=np.float64) @ R.T + t
        all_pts.append(pts)
    all_pts = np.vstack(all_pts)
    return np.mean(all_pts, axis=0)


def load_usd_meshes(path: Path) -> Optional[List[Tuple[Any, np.ndarray]]]:
    """Load USD as list of (mesh, 4x4 transform). Tries trimesh first, then pxr.
    All USD meshes are transformed Z-up -> Y-up and translated so centroid is at origin."""
    out = _usd_meshes_via_trimesh(path)
    if out is None:
        out = _usd_meshes_via_pxr(path)
    if out is None:
        return None
    # Apply Z-up to Y-up so USD matches GLB orientation
    pairs = [(m, _T_ZUP_TO_YUP @ t) for m, t in out]
    # Translate so centroid is at origin
    centroid = _centroid_of_transformed_meshes(pairs)
    T_shift = np.eye(4, dtype=np.float64)
    T_shift[:3, 3] = -centroid
    return [(m, T_shift @ t) for m, t in pairs]


def merge_to_one_glb(
    input_dir: Path,
    output_glb: Path,
    spacing: float = 2.0,
    exclude_patterns: Optional[List[str]] = None,
    include_depth: bool = True,
    usd_scale: float = 1.0,
) -> None:
    """Collect all .glb and .usd in input_dir, place side-by-side along X, export one GLB."""
    try:
        import trimesh
    except ImportError:
        raise SystemExit("trimesh is required. pip install trimesh")

    input_dir = Path(input_dir).resolve()
    output_glb = Path(output_glb).resolve()
    exclude_patterns = exclude_patterns or []

    glb_files = sorted(input_dir.glob("*.glb"))
    # Don't include the output file itself (avoids merging previous comparison and duplicating everything)
    if output_glb.parent.samefile(input_dir) and output_glb.suffix.lower() == ".glb":
        glb_files = [f for f in glb_files if f.resolve() != output_glb]
    usd_files = sorted(input_dir.glob("*.usd")) + sorted(input_dir.glob("*.usda")) + sorted(input_dir.glob("*.usdc"))
    if output_glb.parent.samefile(input_dir) and output_glb.suffix.lower() in (".usd", ".usda", ".usdc"):
        usd_files = [f for f in usd_files if f.resolve() != output_glb]
    # Deduplicate .usd / .usda / .usdc same stem
    seen = set()
    usd_files = [f for f in usd_files if (f.stem not in seen and (seen.add(f.stem) or True))]

    all_entries = []  # (list of (mesh/pointcloud, 4x4), display_name)
    cursor_x = 0.0

    # Depth point cloud first (left of everything) if depth.npy + meta.json exist
    if include_depth:
        depth_points_odom = _load_depth_point_cloud_from_job_dir(input_dir, use_mask=True)
        if depth_points_odom is None:
            depth_points_odom = _load_depth_point_cloud_from_job_dir(input_dir, use_mask=False)
        if depth_points_odom is not None and len(depth_points_odom) > 0:
            # Odom is Z-up; convert to Y-up and place centroid at (depth_left_x, 0, 0)
            ones = np.ones((len(depth_points_odom), 1), dtype=np.float64)
            h = np.hstack([depth_points_odom, ones]).T  # (4, N)
            pts_yup = (_T_ZUP_TO_YUP @ h).T[:, :3]
            centroid = np.mean(pts_yup, axis=0)
            pts_centered = pts_yup - centroid
            pts_final = pts_centered + np.array([_DEPTH_LEFT_X, 0.0, 0.0], dtype=np.float64)
            pts_final = _voxel_downsample_points(pts_final, 0.02)
            if len(pts_final) > 0:
                pc = trimesh.points.PointCloud(vertices=pts_final)
                all_entries.insert(0, ([(pc, np.eye(4))], "depth"))

    # GLBs first at origin, then USD
    for f in glb_files:
        if any(p in f.name for p in exclude_patterns):
            continue
        pairs = _load_glb_meshes(f)
        if not pairs:
            continue
        # Compute bbox of all meshes in this file for spacing
        max_extent = 0.0
        for mesh, tf in pairs:
            mesh_copy = mesh.copy()
            mesh_copy.apply_transform(tf)
            ext = np.max(mesh_copy.bounds[1] - mesh_copy.bounds[0])
            max_extent = max(max_extent, ext)
        offset = np.eye(4, dtype=np.float64)
        offset[0, 3] = cursor_x
        all_entries.append(([(m, offset @ t) for m, t in pairs], f.stem))
        cursor_x += max(max_extent, 0.5) + spacing

    # Scale matrix for USD (uniform scale)
    S_usd = np.eye(4, dtype=np.float64)
    if usd_scale != 1.0:
        S_usd[0, 0] = S_usd[1, 1] = S_usd[2, 2] = usd_scale
    for f in usd_files:
        if any(p in f.name for p in exclude_patterns):
            continue
        pairs = load_usd_meshes(f)
        if not pairs:
            continue
        max_extent = 0.0
        for mesh, tf in pairs:
            mesh_copy = mesh.copy()
            mesh_copy.apply_transform(S_usd @ tf)
            ext = np.max(mesh_copy.bounds[1] - mesh_copy.bounds[0])
            max_extent = max(max_extent, ext)
        offset = np.eye(4, dtype=np.float64)
        offset[0, 3] = cursor_x
        all_entries.append(([(m, offset @ S_usd @ t) for m, t in pairs], f.stem))
        cursor_x += max(max_extent, 0.5) + spacing

    if not all_entries:
        raise SystemExit("No GLB or USD meshes found in %s" % input_dir)

    scene = trimesh.Scene()
    for (pairs, name) in all_entries:
        for i, (mesh, transform) in enumerate(pairs):
            node_name = "%s_%d" % (name, i) if len(pairs) > 1 else name
            scene.add_geometry(mesh.copy(), transform=transform, node_name=node_name)

    output_glb.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(output_glb))
    print("Wrote %s (%d assets)" % (output_glb, len(all_entries)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge GLB and USD files in a directory into one GLB for comparison."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=_EVAL_ROOT / "chairs",
        help="Directory containing .glb and .usd files (default: evaluations/chairs)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output GLB path (default: <input-dir>/comparison.glb)",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=2.0,
        help="Spacing in meters between models along X (default: 2.0)",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="*",
        default=[],
        help="Substrings in filenames to exclude",
    )
    parser.add_argument(
        "--no-depth",
        action="store_true",
        help="Do not include depth.npy point cloud (even if depth.npy + meta.json exist)",
    )
    parser.add_argument(
        "--usd-scale",
        type=float,
        default=1.0,
        metavar="FACTOR",
        help="Scale factor for USD meshes (default: 1.0). Use e.g. 1.2 to enlarge.",
    )
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    output_glb = args.out or (input_dir / "comparison.glb")
    if not input_dir.is_dir():
        raise SystemExit("Not a directory: %s" % input_dir)
    merge_to_one_glb(
        input_dir,
        output_glb,
        spacing=args.spacing,
        exclude_patterns=args.exclude or None,
        include_depth=not args.no_depth,
        usd_scale=args.usd_scale,
    )


if __name__ == "__main__":
    main()
