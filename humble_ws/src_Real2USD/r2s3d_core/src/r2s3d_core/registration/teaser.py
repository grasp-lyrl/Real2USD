"""TEASER++ Sim(3) registration: correspondence-based, scale-aware, outlier-robust.

Rigid ICP recovers pose only — it cannot rescale a mesh, so it stalls on SAM3D's
residual ~0.3 scale error (see docs/STATUS.md). TEASER++ estimates a **similarity**
transform (uniform scale + rotation + translation) from putative FPFH feature
correspondences, certifiably robust to gross outlier matches. This is the Phase-3 lever
for that scale error.

Pipeline: voxel-downsample both clouds → FPFH features (open3d) → mutual-NN feature
correspondences → TEASER++ robust Sim(3) solve. Returns a 4x4 that applies as
``p' = s*R*p + t`` (so it scales the source mesh into the target).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def _fpfh(pts: np.ndarray, voxel: float):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(pts, float)))
    down = pcd.voxel_down_sample(voxel)
    down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.0, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5.0, max_nn=100))
    return np.asarray(down.points), np.asarray(fpfh.data).T  # (M,3), (M,33)


def _mutual_correspondences(fa: np.ndarray, fb: np.ndarray) -> np.ndarray:
    """Indices (i in A, j in B) that are mutual nearest neighbours in FPFH space."""
    from scipy.spatial import cKDTree

    tb, ta = cKDTree(fb), cKDTree(fa)
    _, a2b = tb.query(fa)
    _, b2a = ta.query(fb)
    return np.array([[i, a2b[i]] for i in range(len(fa)) if b2a[a2b[i]] == i], dtype=int)


def register_teaser(source_pts: np.ndarray, target_pts: np.ndarray, voxel: float = 0.03,
                    noise_bound: Optional[float] = None, estimate_scaling: bool = True
                    ) -> Tuple[np.ndarray, dict]:
    """Sim(3) aligning ``source_pts`` onto ``target_pts`` via FPFH + TEASER++.

    Returns ``(T_4x4, info)`` where ``T`` applies as ``s*R*p + t``. On failure (too few
    correspondences / solver error) returns identity with ``info['ok'] = False`` — loud,
    never a silent no-op.
    """
    import teaserpp_python

    nb = float(noise_bound if noise_bound is not None else voxel)
    src_pts, src_f = _fpfh(source_pts, voxel)
    dst_pts, dst_f = _fpfh(target_pts, voxel)
    if len(src_pts) < 3 or len(dst_pts) < 3:
        return np.eye(4), {"ok": False, "reason": "too_few_points",
                           "n_src": len(src_pts), "n_dst": len(dst_pts)}

    pairs = _mutual_correspondences(src_f, dst_f)
    if len(pairs) < 3:
        return np.eye(4), {"ok": False, "reason": "too_few_correspondences",
                           "n_corr": int(len(pairs))}

    S = src_pts[pairs[:, 0]].T  # (3, N)
    D = dst_pts[pairs[:, 1]].T

    p = teaserpp_python.RobustRegistrationSolver.Params()
    p.cbar2 = 1.0
    p.noise_bound = nb
    p.estimate_scaling = estimate_scaling
    p.rotation_estimation_algorithm = (
        teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS)
    p.rotation_gnc_factor = 1.4
    p.rotation_max_iterations = 100
    p.rotation_cost_threshold = 1e-12
    try:
        solver = teaserpp_python.RobustRegistrationSolver(p)
        solver.solve(S, D)
        sol = solver.getSolution()
    except Exception as e:  # pragma: no cover - solver-internal failures
        return np.eye(4), {"ok": False, "reason": f"solver_error: {e}", "n_corr": int(len(pairs))}

    scale = float(sol.scale) if estimate_scaling else 1.0
    R = np.asarray(sol.rotation, float)
    t = np.asarray(sol.translation, float)
    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = t
    return T, {"ok": True, "scale": scale, "n_corr": int(len(pairs)),
               "n_src": int(len(src_pts)), "n_dst": int(len(dst_pts))}
