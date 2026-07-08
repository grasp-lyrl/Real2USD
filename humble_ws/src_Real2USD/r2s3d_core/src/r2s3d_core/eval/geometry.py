"""Geometry primitives for the evaluation harness.

Oriented-box IoU (exact, via halfspace intersection + convex hull) is ported from
``humble_ws/evaluations/metrics_3d.py`` so v1 and v2 report the same IoU. Adds the
box representation used throughout the metrics, surface sampling, Chamfer/F-score,
and symmetry-aware rotation error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection


@dataclass
class Box:
    """An oriented 3D bounding box in the world frame.

    center: (3,) box center.
    R: (3,3) rotation, columns are the box axes in world.
    extents: (3,) full side lengths (not half).
    """

    center: np.ndarray
    R: np.ndarray
    extents: np.ndarray

    @classmethod
    def from_pose(cls, T_world_obj: np.ndarray, extents: np.ndarray) -> "Box":
        T = np.asarray(T_world_obj, dtype=np.float64)
        return cls(center=T[:3, 3].copy(), R=T[:3, :3].copy(), extents=np.asarray(extents, dtype=np.float64))

    @property
    def volume(self) -> float:
        return float(np.prod(np.maximum(self.extents, 0.0)))

    def corners(self) -> np.ndarray:
        h = 0.5 * self.extents
        signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float64)
        local = signs * h
        return (self.R @ local.T).T + self.center


# --------------------------------------------------------------------- OBB IoU

def _box_halfspaces(center: np.ndarray, dims: np.ndarray, rot: np.ndarray) -> np.ndarray:
    h = np.maximum(dims * 0.5, 0.0)
    hs = []
    for i in range(3):
        n = rot[:, i]
        b_pos = float(h[i] + np.dot(n, center))
        b_neg = float(h[i] - np.dot(n, center))
        hs.append([float(n[0]), float(n[1]), float(n[2]), -b_pos])
        hs.append([float(-n[0]), float(-n[1]), float(-n[2]), -b_neg])
    return np.asarray(hs, dtype=np.float64)


def _find_interior_point(halfspaces: np.ndarray) -> Optional[np.ndarray]:
    A = halfspaces[:, :3]
    d = halfspaces[:, 3]
    m = A.shape[0]
    A_ub = np.hstack([A, np.ones((m, 1))])
    b_ub = -d
    c = np.array([0.0, 0.0, 0.0, -1.0])
    bounds = [(None, None)] * 4
    res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success or res.x[3] <= 1e-9:
        return None
    return res.x[:3]


def obb_iou(a: Box, b: Box) -> float:
    """Exact IoU of two oriented boxes."""
    va, vb = a.volume, b.volume
    if va <= 0.0 or vb <= 0.0:
        return 0.0
    halfspaces = np.vstack([
        _box_halfspaces(a.center, a.extents, a.R),
        _box_halfspaces(b.center, b.extents, b.R),
    ])
    interior = _find_interior_point(halfspaces)
    if interior is None:
        return 0.0
    try:
        hs = HalfspaceIntersection(halfspaces, interior_point=interior)
        verts = np.asarray(hs.intersections, dtype=np.float64)
        if verts.shape[0] < 4:
            return 0.0
        inter = float(ConvexHull(verts).volume)
    except Exception:
        return 0.0
    union = va + vb - inter
    return inter / union if union > 0.0 else 0.0


# ----------------------------------------------------------- rotation + scale

def rotation_geodesic_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Geodesic angle (degrees) between two rotations."""
    Rrel = Ra.T @ Rb
    cos = (np.trace(Rrel) - 1.0) * 0.5
    cos = float(np.clip(cos, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def _Rz(theta_rad: float) -> np.ndarray:
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# Scan2CAD-style symmetry groups about the world up (Z) axis.
_SYMMETRY_YAWS = {
    "none": [0.0],
    "c2": [0.0, np.pi],
    "c4": [0.0, np.pi / 2, np.pi, 3 * np.pi / 2],
    "inf": list(np.linspace(0.0, 2 * np.pi, 180, endpoint=False)),
}


def rotation_error_deg(R_pred: np.ndarray, R_gt: np.ndarray, symmetry: str = "none") -> float:
    """Symmetry-aware rotation error: min geodesic over the class's yaw symmetry group.

    Symmetry rotations are applied about the world up-axis in the GT object frame
    (i.e. GT looks identical under R_gt @ Rz(theta) for theta in the group).
    """
    yaws = _SYMMETRY_YAWS.get(symmetry, _SYMMETRY_YAWS["none"])
    return min(rotation_geodesic_deg(R_pred, R_gt @ _Rz(t)) for t in yaws)


def scale_ratio_error(extents_pred: np.ndarray, extents_gt: np.ndarray) -> np.ndarray:
    """Per-axis |s_pred/s_gt - 1|."""
    gt = np.maximum(np.asarray(extents_gt, dtype=np.float64), 1e-9)
    pred = np.asarray(extents_pred, dtype=np.float64)
    return np.abs(pred / gt - 1.0)


def _cube_rotations() -> list:
    """The 24 proper rotations of a cube (signed permutation matrices, det +1)."""
    import itertools

    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            M = np.zeros((3, 3))
            for i, p in enumerate(perm):
                M[i, p] = signs[i]
            if abs(np.linalg.det(M) - 1.0) < 1e-6:
                mats.append(M)
    return mats


_CUBE_ROTATIONS = _cube_rotations()


def box_pose_error(R_pred, ext_pred, R_gt, ext_gt, symmetry: str = "none"):
    """Axis-labeling-invariant orientation + scale error for two oriented boxes.

    A box's principal axes have no intrinsic labels/signs, so the min-volume OBB of a
    predicted mesh may name its axes in any order relative to canonical GT axes.
    Comparing raw rotation matrices then wildly overstates the error (e.g. a correctly
    oriented object reads as ~180 deg). This resolves the correspondence by choosing
    the cube symmetry g that best aligns the predicted frame to GT (on top of the
    class's yaw symmetry group), and reports the per-axis scale ratio error under that
    same correspondence — so rotation and scale stay physically consistent.

    Returns ``(rotation_deg, scale_err_per_axis)``.
    """
    Rp = np.asarray(R_pred, dtype=np.float64)
    Rg = np.asarray(R_gt, dtype=np.float64)
    ep = np.asarray(ext_pred, dtype=np.float64)
    eg = np.maximum(np.asarray(ext_gt, dtype=np.float64), 1e-9)
    yaws = _SYMMETRY_YAWS.get(symmetry, _SYMMETRY_YAWS["none"])
    Rg_sym = [Rg @ _Rz(t) for t in yaws]

    best_rot, best_ext = 180.0, np.abs(ep)
    for g in _CUBE_ROTATIONS:
        Ra = Rp @ g
        rot = min(rotation_geodesic_deg(Ra, Rgs) for Rgs in Rg_sym)
        if rot < best_rot:
            best_rot = rot
            best_ext = np.abs(g.T @ ep)  # extents reordered to GT axes
    scale_err = np.abs(best_ext / eg - 1.0)
    return best_rot, scale_err


# ---------------------------------------------------------- Chamfer / F-score

def sample_surface(mesh, n: int = 10000, seed: int = 0) -> np.ndarray:
    """Sample ~n points on a trimesh surface (deterministic)."""
    import trimesh  # local import to keep geometry import light

    if mesh is None or len(getattr(mesh, "faces", [])) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    rng = np.random.RandomState(seed)
    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=rng.randint(0, 2**31 - 1))
    return np.asarray(pts, dtype=np.float64)


def chamfer_and_fscore(pts_pred: np.ndarray, pts_gt: np.ndarray, taus=(0.05, 0.02)):
    """Chamfer-L1 (symmetric mean NN distance) and F-score at each tau.

    Returns dict with ``chamfer_l1`` and ``fscore@<tau>`` keys.
    """
    from scipy.spatial import cKDTree

    out = {"chamfer_l1": float("nan")}
    for tau in taus:
        out[f"fscore@{tau}"] = float("nan")
    if len(pts_pred) == 0 or len(pts_gt) == 0:
        return out

    tp = cKDTree(pts_pred)
    tg = cKDTree(pts_gt)
    d_pred_to_gt, _ = tg.query(pts_pred)   # each pred point to nearest gt
    d_gt_to_pred, _ = tp.query(pts_gt)     # each gt point to nearest pred
    out["chamfer_l1"] = float(np.mean(d_pred_to_gt) + np.mean(d_gt_to_pred))
    for tau in taus:
        precision = float(np.mean(d_pred_to_gt < tau))
        recall = float(np.mean(d_gt_to_pred < tau))
        f = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        out[f"fscore@{tau}"] = f
    return out
