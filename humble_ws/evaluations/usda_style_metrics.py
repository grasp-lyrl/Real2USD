"""
USDA-style metrics: many-to-many 2D (XY) overlap matching, 2D IoU, strict/relaxed by 2D centroid containment.
Use this to get numbers comparable to usda_labeled_bbox_eval.py from the same pred/GT data as run_eval.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def _aabb_2d_min_max(box: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Return (min_xy, max_xy) for a box with aabb_min/aabb_max or center + dimensions."""
    if "aabb_min" in box and "aabb_max" in box:
        mn = np.asarray(box["aabb_min"], dtype=np.float64)[:2]
        mx = np.asarray(box["aabb_max"], dtype=np.float64)[:2]
        return mn, mx
    c = np.asarray(box["center"], dtype=np.float64)[:2]
    # If only center, use a tiny box (avoid zero area)
    d = np.asarray(box.get("dimensions", [0.01, 0.01, 0.01]), dtype=np.float64)[:2]
    return c - d / 2, c + d / 2


def _center_2d(box: Dict) -> np.ndarray:
    if "center" in box:
        return np.asarray(box["center"], dtype=np.float64)[:2]
    mn, mx = _aabb_2d_min_max(box)
    return (mn + mx) / 2.0


def _iou_2d_axis_aligned(min1: np.ndarray, max1: np.ndarray, min2: np.ndarray, max2: np.ndarray) -> float:
    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)
    if np.any(inter_min >= inter_max):
        return 0.0
    inter_area = float(np.prod(inter_max - inter_min))
    area1 = float(np.prod(np.maximum(max1 - min1, 0.0)))
    area2 = float(np.prod(np.maximum(max2 - min2, 0.0)))
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0.0


def _overlap_2d(min1: np.ndarray, max1: np.ndarray, min2: np.ndarray, max2: np.ndarray) -> bool:
    return not (
        max1[0] < min2[0] or max2[0] < min1[0] or max1[1] < min2[1] or max2[1] < min1[1]
    )


def _point_in_box_2d(p: np.ndarray, min_xy: np.ndarray, max_xy: np.ndarray) -> bool:
    return bool(np.all(p >= min_xy) and np.all(p <= max_xy))


def compute_usda_style_metrics(
    preds: List[Dict],
    gts: List[Dict],
    require_label_match: bool = False,
) -> Dict:
    """
    Compute USDA-style metrics from the same pred/GT format as run_eval (center, aabb_min, aabb_max).

    - Matching: many-to-many; every (pred, GT) pair with 2D (XY) axis-aligned overlap is counted.
    - IoU: 2D axis-aligned (XY).
    - Strict: both centroids (2D) inside the other's box.
    - Relaxed: at least one centroid (2D) inside the other's box.
    - strict_accuracy_usda = strict_pair_count / n_gt (fraction; can exceed 1.0).
    - relaxed_accuracy_usda = relaxed_pair_count / n_gt (fraction; can exceed 1.0).
    - Per-class: same metrics restricted to pairs where pred and GT have the same label_canonical.

    Returns a dict with keys suitable for comparison to usda_labeled_bbox_eval overlap_statistics
    (using fraction 0-1 for accuracy, not percent), plus a "per_class" dict keyed by label.
    """
    n_gt = len(gts)
    n_pred = len(preds)

    def _canonical(box: Dict) -> str:
        return (box.get("label_canonical") or "").strip()

    pred_min = []
    pred_max = []
    pred_center = []
    pred_label = []
    for p in preds:
        mn, mx = _aabb_2d_min_max(p)
        pred_min.append(mn)
        pred_max.append(mx)
        pred_center.append(_center_2d(p))
        pred_label.append(_canonical(p))
    gt_min = []
    gt_max = []
    gt_center = []
    gt_label = []
    for g in gts:
        mn, mx = _aabb_2d_min_max(g)
        gt_min.append(mn)
        gt_max.append(mx)
        gt_center.append(_center_2d(g))
        gt_label.append(_canonical(g))

    strict_pairs = 0
    relaxed_pairs = 0
    iou_values = []

    # Per-class: only count pairs where both pred and GT have the same label
    per_class: Dict[str, Dict] = {}
    all_labels = set(pred_label) | set(gt_label)
    for lbl in all_labels:
        per_class[lbl] = {
            "n_gt": 0,
            "n_pred": 0,
            "strict_pair_count": 0,
            "relaxed_pair_count": 0,
            "iou_values": [],
        }
    for g in gts:
        lbl = _canonical(g)
        per_class[lbl]["n_gt"] = per_class[lbl].get("n_gt", 0) + 1
    for p in preds:
        lbl = _canonical(p)
        per_class[lbl]["n_pred"] = per_class[lbl].get("n_pred", 0) + 1

    for pi in range(n_pred):
        for gi in range(n_gt):
            plbl = pred_label[pi]
            glbl = gt_label[gi]
            if require_label_match and plbl != glbl:
                continue
            mn1, mx1 = pred_min[pi], pred_max[pi]
            mn2, mx2 = gt_min[gi], gt_max[gi]
            if not _overlap_2d(mn1, mx1, mn2, mx2):
                continue
            c1 = pred_center[pi]
            c2 = gt_center[gi]
            strict = _point_in_box_2d(c1, mn2, mx2) and _point_in_box_2d(c2, mn1, mx1)
            relaxed = _point_in_box_2d(c1, mn2, mx2) or _point_in_box_2d(c2, mn1, mx1)
            iou = _iou_2d_axis_aligned(mn1, mx1, mn2, mx2)

            strict_pairs += 1 if strict else 0
            relaxed_pairs += 1 if relaxed else 0
            iou_values.append(iou)

            # Per-class: only when both same label
            if plbl == glbl:
                per_class[plbl]["strict_pair_count"] += 1 if strict else 0
                per_class[plbl]["relaxed_pair_count"] += 1 if relaxed else 0
                per_class[plbl]["iou_values"].append(iou)

    strict_accuracy_usda = (strict_pairs / n_gt) if n_gt > 0 else 0.0
    relaxed_accuracy_usda = (relaxed_pairs / n_gt) if n_gt > 0 else 0.0
    mean_iou_2d = float(np.mean(iou_values)) if iou_values else 0.0
    f1_relaxed_usda = (
        2.0 * relaxed_accuracy_usda * mean_iou_2d / (relaxed_accuracy_usda + mean_iou_2d)
        if (relaxed_accuracy_usda + mean_iou_2d) > 0
        else 0.0
    )

    # Finalize per-class: add accuracy and mean IoU, drop raw iou list
    per_class_out: Dict[str, Dict] = {}
    for lbl, pc in per_class.items():
        n_gt_c = pc["n_gt"]
        ious = pc["iou_values"]
        strict_c = pc["strict_pair_count"]
        relaxed_c = pc["relaxed_pair_count"]
        strict_acc_c = (strict_c / n_gt_c) if n_gt_c > 0 else 0.0
        relaxed_acc_c = (relaxed_c / n_gt_c) if n_gt_c > 0 else 0.0
        mean_iou_c = float(np.mean(ious)) if ious else 0.0
        f1_c = (
            2.0 * relaxed_acc_c * mean_iou_c / (relaxed_acc_c + mean_iou_c)
            if (relaxed_acc_c + mean_iou_c) > 0
            else 0.0
        )
        per_class_out[lbl] = {
            "n_gt": pc["n_gt"],
            "n_pred": pc["n_pred"],
            "strict_pair_count": strict_c,
            "relaxed_pair_count": relaxed_c,
            "strict_accuracy_usda": float(strict_acc_c),
            "relaxed_accuracy_usda": float(relaxed_acc_c),
            "mean_iou_2d": float(mean_iou_c),
            "f1_relaxed_usda": float(f1_c),
        }
    return {
        "strict_accuracy_usda": float(strict_accuracy_usda),
        "relaxed_accuracy_usda": float(relaxed_accuracy_usda),
        "mean_iou_2d": float(mean_iou_2d),
        "f1_relaxed_usda": float(f1_relaxed_usda),
        "strict_pair_count": strict_pairs,
        "relaxed_pair_count": relaxed_pairs,
        "overlapping_pair_count": len(iou_values),
        "n_pred": n_pred,
        "n_gt": n_gt,
        "per_class": per_class_out,
    }
