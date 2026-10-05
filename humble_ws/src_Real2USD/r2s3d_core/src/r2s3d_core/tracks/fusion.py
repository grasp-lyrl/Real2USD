"""Multi-view geometry for tracks: masked-depth back-projection, a voxel-hashed fused
cloud, and cloud<->image projection helpers used by the associator.

PHASE_SPECS §Phase 2: "Fused cloud: 1 cm voxel hash, per-voxel max 1 point,
mask-filtered depth only." All clouds are in the gravity-aligned Z-up world frame.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ..data.base import Frame
from ..eval import geometry as geo


def backproject_mask(frame: Frame, mask: np.ndarray) -> np.ndarray:
    """Back-project masked, valid-depth pixels of ``frame`` into the world frame.

    (Same math as ``baselines.sam3d_layout._masked_depth_cloud`` — kept here so the
    tracker has no baseline dependency.)
    """
    H, W = frame.depth.shape[:2]
    fx, fy, cx, cy = frame.K[0, 0], frame.K[1, 1], frame.K[0, 2], frame.K[1, 2]
    m = np.asarray(mask, bool)
    ys, xs = np.where(m & (frame.depth > 0) & np.isfinite(frame.depth))
    if len(xs) == 0:
        return np.zeros((0, 3), np.float64)
    z = frame.depth[ys, xs].astype(np.float64)
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    pc = np.stack([x, y, z], axis=1)
    return (frame.T_world_cam[:3, :3] @ pc.T).T + frame.T_world_cam[:3, 3]


def mask_centroid_world(frame: Frame, mask: np.ndarray) -> Optional[np.ndarray]:
    """World centroid of a detection from its masked median depth (robust to
    outliers). Uses the median point, not the mean."""
    pts = backproject_mask(frame, mask)
    if len(pts) == 0:
        return None
    return np.median(pts, axis=0)


class VoxelCloud:
    """1 cm voxel hash: at most one point per occupied voxel (voxel-center)."""

    def __init__(self, voxel: float = 0.01):
        self.voxel = float(voxel)
        self._voxels: set = set()

    def add(self, points: np.ndarray) -> None:
        if len(points) == 0:
            return
        keys = np.floor(np.asarray(points) / self.voxel).astype(np.int64)
        for k in map(tuple, keys):
            self._voxels.add(k)

    @property
    def n(self) -> int:
        return len(self._voxels)

    def points(self) -> np.ndarray:
        if not self._voxels:
            return np.zeros((0, 3), np.float64)
        arr = np.array(list(self._voxels), dtype=np.float64)
        return (arr + 0.5) * self.voxel

    def keys(self) -> set:
        return self._voxels


def cloud_overlap(a: "VoxelCloud", b: "VoxelCloud") -> float:
    """Fraction of the smaller cloud's voxels shared with the other (0..1)."""
    ka, kb = a.keys(), b.keys()
    if not ka or not kb:
        return 0.0
    return len(ka & kb) / min(len(ka), len(kb))


def aabb_box(points: np.ndarray) -> Optional[geo.Box]:
    """Axis-aligned box (as a geo.Box with identity rotation) around a cloud."""
    if len(points) < 2:
        return None
    lo, hi = points.min(0), points.max(0)
    ext = np.maximum(hi - lo, 1e-3)
    return geo.Box(center=0.5 * (lo + hi), R=np.eye(3), extents=ext)


def cloud_iou(a: np.ndarray, b: np.ndarray) -> float:
    """OBB (axis-aligned) IoU between two clouds — cheap merge gate."""
    ba, bb = aabb_box(a), aabb_box(b)
    if ba is None or bb is None:
        return 0.0
    return geo.obb_iou(ba, bb)


def project_cloud_mask(points_world: np.ndarray, frame: Frame,
                       fill_hull: bool = True) -> Optional[np.ndarray]:
    """Project a world cloud into ``frame`` and return its silhouette mask.

    Points behind the camera or off-image are dropped. With ``fill_hull`` the convex
    hull of the projected points is filled (a track's cloud is sparse, so raw splats
    under-cover); otherwise individual pixels are set.
    """
    H, W = frame.depth.shape[:2]
    if len(points_world) == 0:
        return None
    T_cam_world = np.linalg.inv(frame.T_world_cam)
    pc = (T_cam_world[:3, :3] @ points_world.T).T + T_cam_world[:3, 3]
    z = pc[:, 2]
    front = z > 1e-6
    if not front.any():
        return None
    fx, fy, cx, cy = frame.K[0, 0], frame.K[1, 1], frame.K[0, 2], frame.K[1, 2]
    u = (fx * pc[front, 0] / z[front] + cx)
    v = (fy * pc[front, 1] / z[front] + cy)
    px = np.stack([u, v], axis=1)
    inb = (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)
    px = px[inb]
    if len(px) < 3:
        return None
    mask = np.zeros((H, W), np.uint8)
    if fill_hull:
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(px)
            poly = px[hull.vertices].astype(np.int32)
            cv2.fillPoly(mask, [poly], 1)
        except Exception:
            mask[px[:, 1].astype(int), px[:, 0].astype(int)] = 1
    else:
        mask[px[:, 1].astype(int), px[:, 0].astype(int)] = 1
    return mask.astype(bool)


def mask_iou(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return inter / union if union > 0 else 0.0
