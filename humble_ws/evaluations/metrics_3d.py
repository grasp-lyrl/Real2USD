from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection


def aabb_iou_3d(a_min: np.ndarray, a_max: np.ndarray, b_min: np.ndarray, b_max: np.ndarray) -> float:
    inter_min = np.maximum(a_min, b_min)
    inter_max = np.minimum(a_max, b_max)
    inter_dims = np.maximum(0.0, inter_max - inter_min)
    inter_vol = float(np.prod(inter_dims))
    if inter_vol <= 0.0:
        return 0.0
    a_vol = float(np.prod(np.maximum(0.0, a_max - a_min)))
    b_vol = float(np.prod(np.maximum(0.0, b_max - b_min)))
    union = a_vol + b_vol - inter_vol
    return inter_vol / union if union > 0 else 0.0


def point_in_aabb_3d(point: np.ndarray, box_min: np.ndarray, box_max: np.ndarray) -> bool:
    return bool(np.all(point >= box_min) and np.all(point <= box_max))


def strict_relaxed_from_aabb(pred: Dict, gt: Dict) -> Tuple[bool, bool]:
    pred_c = np.asarray(pred["center"], dtype=np.float64)
    gt_c = np.asarray(gt["center"], dtype=np.float64)
    pred_min = np.asarray(pred["aabb_min"], dtype=np.float64)
    pred_max = np.asarray(pred["aabb_max"], dtype=np.float64)
    gt_min = np.asarray(gt["aabb_min"], dtype=np.float64)
    gt_max = np.asarray(gt["aabb_max"], dtype=np.float64)
    pred_in_gt = point_in_aabb_3d(pred_c, gt_min, gt_max)
    gt_in_pred = point_in_aabb_3d(gt_c, pred_min, pred_max)
    return pred_in_gt and gt_in_pred, pred_in_gt or gt_in_pred


def _box_volume_from_dims(dimensions: np.ndarray) -> float:
    return float(np.prod(np.maximum(dimensions, 0.0)))


def _box_halfspaces(center: np.ndarray, dims: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """
    Return halfspaces as rows [a, b, c, d] for inequalities: a*x + b*y + c*z + d <= 0.
    For OBB: abs(R^T (x-c)) <= h.
    """
    h = np.maximum(dims * 0.5, 0.0)
    hs = []
    for i in range(3):
        n = rot[:, i]
        b_pos = float(h[i] + np.dot(n, center))
        b_neg = float(h[i] - np.dot(n, center))
        # n . x <= b_pos
        hs.append([float(n[0]), float(n[1]), float(n[2]), -b_pos])
        # -n . x <= b_neg
        hs.append([float(-n[0]), float(-n[1]), float(-n[2]), -b_neg])
    return np.asarray(hs, dtype=np.float64)


def _find_interior_point(halfspaces: np.ndarray) -> np.ndarray:
    """
    Solve max t s.t. A x + d + t <= 0.
    If max t > 0, x is strictly interior.
    """
    A = halfspaces[:, :3]
    d = halfspaces[:, 3]
    m = A.shape[0]
    A_ub = np.hstack([A, np.ones((m, 1), dtype=np.float64)])
    b_ub = -d
    c = np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float64)  # maximize t
    bounds = [(None, None), (None, None), (None, None), (None, None)]
    res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        return None
    x = res.x[:3]
    t = res.x[3]
    if t <= 1e-9:
        return None
    return x


def obb_iou_3d(
    a_center: np.ndarray,
    a_dims: np.ndarray,
    a_rot: np.ndarray,
    b_center: np.ndarray,
    b_dims: np.ndarray,
    b_rot: np.ndarray,
) -> float:
    vol_a = _box_volume_from_dims(a_dims)
    vol_b = _box_volume_from_dims(b_dims)
    if vol_a <= 0.0 or vol_b <= 0.0:
        return 0.0

    halfspaces = np.vstack([
        _box_halfspaces(a_center, a_dims, a_rot),
        _box_halfspaces(b_center, b_dims, b_rot),
    ])
    interior = _find_interior_point(halfspaces)
    if interior is None:
        return 0.0

    try:
        hs = HalfspaceIntersection(halfspaces, interior_point=interior)
        verts = np.asarray(hs.intersections, dtype=np.float64)
        if verts.shape[0] < 4:
            return 0.0
        inter_vol = float(ConvexHull(verts).volume)
    except Exception:
        return 0.0

    union = vol_a + vol_b - inter_vol
    return inter_vol / union if union > 0.0 else 0.0


def pair_iou_3d(pred: Dict, gt: Dict, iou_mode: str = "aabb") -> float:
    mode = (iou_mode or "aabb").strip().lower()
    if mode == "obb":
        if (
            pred.get("obb_center") is not None
            and pred.get("obb_dimensions") is not None
            and pred.get("obb_rotation_matrix") is not None
            and gt.get("obb_center") is not None
            and gt.get("obb_dimensions") is not None
            and gt.get("obb_rotation_matrix") is not None
        ):
            return obb_iou_3d(
                np.asarray(pred["obb_center"], dtype=np.float64),
                np.asarray(pred["obb_dimensions"], dtype=np.float64),
                np.asarray(pred["obb_rotation_matrix"], dtype=np.float64),
                np.asarray(gt["obb_center"], dtype=np.float64),
                np.asarray(gt["obb_dimensions"], dtype=np.float64),
                np.asarray(gt["obb_rotation_matrix"], dtype=np.float64),
            )
    return aabb_iou_3d(
        np.asarray(pred["aabb_min"], dtype=np.float64),
        np.asarray(pred["aabb_max"], dtype=np.float64),
        np.asarray(gt["aabb_min"], dtype=np.float64),
        np.asarray(gt["aabb_max"], dtype=np.float64),
    )


def greedy_match(
    preds: List[Dict],
    gts: List[Dict],
    iou_threshold: float,
    require_label_match: bool = True,
    iou_mode: str = "aabb",
) -> Tuple[List[Dict], List[int], List[int], List[Dict]]:
    candidates = []
    for pi, pred in enumerate(preds):
        for gi, gt in enumerate(gts):
            if require_label_match and pred.get("label_canonical") != gt.get("label_canonical"):
                continue
            iou = pair_iou_3d(pred, gt, iou_mode=iou_mode)
            if iou >= iou_threshold:
                strict, relaxed = strict_relaxed_from_aabb(pred, gt)
                candidates.append(
                    {
                        "pred_idx": pi,
                        "gt_idx": gi,
                        "iou": iou,
                        "strict": strict,
                        "relaxed": relaxed,
                        "centroid_error_m": float(
                            np.linalg.norm(np.asarray(pred["center"]) - np.asarray(gt["center"]))
                        ),
                    }
                )
    candidates.sort(key=lambda x: x["iou"], reverse=True)

    used_p, used_g, matches = set(), set(), []
    for c in candidates:
        if c["pred_idx"] in used_p or c["gt_idx"] in used_g:
            continue
        matches.append(c)
        used_p.add(c["pred_idx"])
        used_g.add(c["gt_idx"])

    unmatched_pred = [i for i in range(len(preds)) if i not in used_p]
    unmatched_gt = [i for i in range(len(gts)) if i not in used_g]
    return matches, unmatched_pred, unmatched_gt, candidates
