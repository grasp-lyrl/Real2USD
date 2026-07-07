"""Placement-accuracy metrics for asset-centric scene reconstruction.

Decouples geometry from semantics (v1 conflated them): matching is label-agnostic
by default; label correctness is reported separately. Everything is computed in the
gravity-aligned Z-up world frame.

Metric summary (see docs/PHASE_SPECS.md Phase 0):
  * Hungarian matching on OBB IoU, threshold 0.25 (0.5 secondary).
  * Per matched pair: centroid L2, symmetry-aware rotation error, per-axis scale
    ratio error, Chamfer-L1, F-score@5cm/2cm.
  * Scan2CAD accuracy: GT fraction with a prediction within 20cm & 20deg & 20% scale.
  * Scene: precision / recall / F1 @ IoU 0.25, duplicate rate, count ratio.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..data.base import GTObject
from . import geometry as geo


@dataclass
class SceneObject:
    """A predicted (or GT-as-prediction) object in the world frame."""

    label: str
    T_world_obj: np.ndarray
    extents: np.ndarray
    mesh: object = None            # trimesh.Trimesh in world frame, optional
    confidence: float = 1.0
    provenance: dict = field(default_factory=dict)

    def box(self) -> geo.Box:
        return geo.Box.from_pose(self.T_world_obj, self.extents)


def gt_to_scene_object(g: GTObject) -> SceneObject:
    return SceneObject(label=g.label, T_world_obj=g.T_world_obj, extents=g.extents, mesh=g.mesh)


# Default Scan2CAD-style symmetry per class (about world up). Keyed by substring of
# the lowercase label; first match wins. Unknown -> "none". Refine as needed.
_SYMMETRY_BY_KEYWORD = [
    ("round table", "inf"), ("vase", "inf"), ("bottle", "inf"), ("cup", "inf"),
    ("bowl", "inf"), ("pot", "inf"), ("plant", "inf"), ("lamp", "inf"),
    ("bin", "inf"), ("basket", "inf"), ("clock", "inf"), ("plate", "inf"),
    ("stool", "c4"), ("box", "c4"),
    ("table", "c2"), ("desk", "c2"), ("bench", "c2"), ("sofa", "c2"),
    ("couch", "c2"), ("bed", "c2"), ("cabinet", "c2"), ("shelf", "c2"),
    ("monitor", "c2"), ("tv", "c2"), ("pillow", "c2"), ("book", "c2"),
]


def symmetry_for_label(label: str) -> str:
    s = (label or "").strip().lower()
    for key, sym in _SYMMETRY_BY_KEYWORD:
        if key in s:
            return sym
    return "none"


# ------------------------------------------------------------------- matching

def _iou_matrix(preds: List[SceneObject], gts: List[SceneObject]) -> np.ndarray:
    M = np.zeros((len(preds), len(gts)), dtype=np.float64)
    pboxes = [p.box() for p in preds]
    gboxes = [g.box() for g in gts]
    for i, pb in enumerate(pboxes):
        for j, gb in enumerate(gboxes):
            M[i, j] = geo.obb_iou(pb, gb)
    return M


def hungarian_match(iou: np.ndarray, threshold: float, require_label: bool = False,
                    labels_pred=None, labels_gt=None):
    """Return list of (pred_idx, gt_idx, iou) with IoU >= threshold, one-to-one.

    Uses Hungarian assignment maximizing total IoU, then drops sub-threshold pairs.
    """
    if iou.size == 0:
        return []
    cost = iou.copy()
    if require_label and labels_pred is not None and labels_gt is not None:
        for i, lp in enumerate(labels_pred):
            for j, lg in enumerate(labels_gt):
                if (lp or "").strip().lower() != (lg or "").strip().lower():
                    cost[i, j] = 0.0
    rows, cols = linear_sum_assignment(-cost)
    matches = []
    for r, c in zip(rows, cols):
        if iou[r, c] >= threshold and (not require_label or cost[r, c] > 0):
            matches.append((int(r), int(c), float(iou[r, c])))
    return matches


# ------------------------------------------------------------ scene-level agg

def _duplicate_and_count(iou: np.ndarray, threshold: float, n_gt: int):
    """Duplicate rate = extra predictions (beyond first) assigned to each GT / #GT.

    Each prediction is assigned to its best-overlapping GT (IoU >= threshold).
    """
    if n_gt == 0:
        return 0.0
    counts = np.zeros(n_gt, dtype=int)
    if iou.size:
        for i in range(iou.shape[0]):
            j = int(np.argmax(iou[i]))
            if iou[i, j] >= threshold:
                counts[j] += 1
    duplicates = int(np.sum(np.maximum(counts - 1, 0)))
    return duplicates / n_gt


def evaluate(preds: List[SceneObject], gts: List[SceneObject],
             iou_threshold: float = 0.25,
             compute_geometry: bool = True,
             surface_points: int = 10000) -> dict:
    """Compute the full Phase-0 metric bundle for one scene."""
    n_pred, n_gt = len(preds), len(gts)
    iou = _iou_matrix(preds, gts)
    matches = hungarian_match(iou, iou_threshold)

    # scene precision/recall/f1
    tp = len(matches)
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # secondary IoU 0.5
    matches_50 = hungarian_match(iou, 0.5)
    recall_50 = len(matches_50) / n_gt if n_gt else 0.0

    # per-pair errors
    centroid_err, rot_err, scale_err_max, label_correct = [], [], [], []
    chamfer, f5, f2 = [], [], []
    scan2cad_hits = 0
    for pi, gi, _ in matches:
        p, g = preds[pi], gts[gi]
        c_err = float(np.linalg.norm(p.T_world_obj[:3, 3] - g.T_world_obj[:3, 3]))
        sym = symmetry_for_label(g.label)
        r_err = geo.rotation_error_deg(p.T_world_obj[:3, :3], g.T_world_obj[:3, :3], sym)
        s_err = geo.scale_ratio_error(p.extents, g.extents)
        s_err_max = float(np.max(s_err))
        centroid_err.append(c_err)
        rot_err.append(r_err)
        scale_err_max.append(s_err_max)
        label_correct.append((p.label or "").strip().lower() == (g.label or "").strip().lower())
        if c_err <= 0.20 and r_err <= 20.0 and s_err_max <= 0.20:
            scan2cad_hits += 1
        if compute_geometry and p.mesh is not None and g.mesh is not None:
            pp = geo.sample_surface(p.mesh, surface_points, seed=pi)
            gg = geo.sample_surface(g.mesh, surface_points, seed=1000 + gi)
            cf = geo.chamfer_and_fscore(pp, gg, taus=(0.05, 0.02))
            chamfer.append(cf["chamfer_l1"])
            f5.append(cf["fscore@0.05"])
            f2.append(cf["fscore@0.02"])

    def _agg(x, fn=np.median):
        return float(fn(x)) if len(x) else float("nan")

    metrics = {
        "n_pred": n_pred,
        "n_gt": n_gt,
        "count_ratio": (n_pred / n_gt) if n_gt else float("nan"),
        "iou_threshold": iou_threshold,
        "tp": tp,
        "fp": n_pred - tp,
        "fn": n_gt - tp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "recall@0.5": recall_50,
        "duplicate_rate": _duplicate_and_count(iou, iou_threshold, n_gt),
        "scan2cad_accuracy": (scan2cad_hits / n_gt) if n_gt else float("nan"),
        "label_accuracy": _agg([1.0 if c else 0.0 for c in label_correct], np.mean),
        "centroid_err_median_m": _agg(centroid_err),
        "centroid_err_mean_m": _agg(centroid_err, np.mean),
        "rotation_err_median_deg": _agg(rot_err),
        "rotation_err_mean_deg": _agg(rot_err, np.mean),
        "scale_err_median": _agg(scale_err_max),
        "scale_err_mean": _agg(scale_err_max, np.mean),
        "chamfer_l1_median_m": _agg(chamfer),
        "fscore@0.05_mean": _agg(f5, np.mean),
        "fscore@0.02_mean": _agg(f2, np.mean),
    }
    return metrics
