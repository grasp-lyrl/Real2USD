"""
Convert USDA parse_usda_data output to prediction box list for run_eval / metrics_3d.
Uses same OBB convention as plot_overlays: boundingBoxMin/Max (world AABB) + orient (w,x,y,z)
→ center, obb_center, dimensions, aabb_min/aabb_max, obb_rotation_matrix.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy.spatial.transform import Rotation as R


def quat_wxyz_to_rotation_matrix(orientation: list) -> np.ndarray:
    """USDA uses quat (w, x, y, z). scipy Rotation.from_quat expects (x, y, z, w)."""
    o = np.asarray(orientation, dtype=np.float64).ravel()
    if len(o) < 4:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = float(o[0]), float(o[1]), float(o[2]), float(o[3])
    return R.from_quat([x, y, z, w]).as_matrix()


def usda_data_to_prediction_boxes(
    usda_data: Dict[str, List[Dict]],
    normalize_yaw_rad: float = 0.0,
) -> List[Dict]:
    """
    Convert USDA parse_usda_data output to list of prediction boxes (same format as run_eval).
    Each box has: id, label, label_canonical, center, obb_center, dimensions, obb_dimensions,
    aabb_min, aabb_max, and optionally obb_rotation_matrix.
    If normalize_yaw_rad != 0, all boxes are rotated by -normalize_yaw_rad around Z (e.g. for GT frame alignment).
    """
    out: List[Dict] = []
    R_z = None
    if abs(normalize_yaw_rad) > 1e-9:
        c, s = np.cos(-normalize_yaw_rad), np.sin(-normalize_yaw_rad)
        R_z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)

    for label, boxes in usda_data.items():
        for box in boxes:
            bmin = np.asarray(box["bbox_min"], dtype=np.float64)
            bmax = np.asarray(box["bbox_max"], dtype=np.float64)
            center = ((bmin + bmax) * 0.5)
            dims = (bmax - bmin)
            aabb_min = bmin.copy()
            aabb_max = bmax.copy()
            orientation = box.get("orientation")
            if orientation and len(orientation) >= 4:
                R_mat = quat_wxyz_to_rotation_matrix(orientation).astype(np.float64)
            else:
                R_mat = np.eye(3, dtype=np.float64)

            if R_z is not None:
                center = R_z @ center
                aabb_min = R_z @ aabb_min
                aabb_max = R_z @ aabb_max
                R_mat = R_z @ R_mat

            base = {
                "id": box.get("id"),
                "label": label,
                "label_canonical": label,
                "center": center.tolist(),
                "obb_center": center.tolist(),
                "dimensions": dims.tolist(),
                "obb_dimensions": dims.tolist(),
                "aabb_min": aabb_min.tolist(),
                "aabb_max": aabb_max.tolist(),
            }
            out.append({**base, "obb_rotation_matrix": R_mat.tolist()})
    return out
