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

Two matching protocols are reported side by side (docs/ACTION_ITEMS.md AI-7):
  * OUR protocol (headline ``f1``/``precision``/``recall`` + ``iou``/``micro``/``macro``):
    Hungarian on 3D OBB IoU >= threshold. Stricter; NOT what the coworker's table uses.
  * COWORKER-COMPARABLE (``cd_*``): GREEDY match on 2D top-down (X,Y) CENTROID distance
    <= tau (default 1 m, swept over ``centroid_f1_by_tau``/``centroid_recall_by_tau``) --
    confirmed vs their ``harness.py`` defaults (greedy, not Hungarian; 2D, not 3D; centroid,
    not IoU). ``cd_f1``/``cd_precision``/``cd_recall`` label-agnostic; ``cd_micro_f1``/
    ``cd_macro_f1``/``cd_per_class`` label-aware (their Object Micro/Macro F1).
    ``cd_micro_f1_many_to_one`` = their over-segmentation-tolerant any-overlap object F1
    (``micro_f1 - many_to_one`` = the over-segmentation penalty).

Other coworker-comparable pieces:
  * class_free_recall_1m: their Class-Free Geo Recall = fraction of GT with ANY predicted
    centroid within 1 m in X,Y (label ignored, NOT one-to-one).
  * scene_chamfer_mean_m: SCENE-LEVEL pooled symmetric Chamfer (their convention, confirmed).
  * surf_recall/surf_precision/surf_fscore@tau: SURFACE-reconstruction point coverage
    (Tanks-and-Temples heritage) -- a DIFFERENT metric from class_free_recall_1m; needs
    meshes on both sides (NaN/absent until GT meshes attach -- AI-8).
  * chamfer_symmetric_mean_m: per-matched-pair average Chamfer (= our chamfer_l1 / 2).
  * matched_per_scene / objects_per_scene / predictions_per_scene: named counts.
Association / fragmentation / merge (Option B) needs per-detection track provenance and is
wired separately. Still to bit-verify vs the coworker's source: macro averaging set and the
exact "many-to-one F1" definition (AI-7).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

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


def _f1(precision: float, recall: float) -> float:
    """Harmonic mean of precision and recall; 0 when both are 0."""
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def _norm_label(label: str) -> str:
    return (label or "").strip().lower()


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


def _micro_macro_from_matches(matches, labels_pred, labels_gt) -> dict:
    """Micro/macro/per-class F1 from a set of LABEL-AWARE matches (each pair shares a label).

    Micro pools true positives over all objects; macro averages per-class F1 over the union
    of GT and predicted classes (a class present on only one side contributes an F1 of 0,
    penalising both misses and hallucinated categories -- the more defensible choice; still
    to bit-verify against the coworker's ``evaluate_objects`` source, AI-7).
    """
    n_pred, n_gt = len(labels_pred), len(labels_gt)
    tp = len(matches)
    micro_precision = tp / n_pred if n_pred else 0.0
    micro_recall = tp / n_gt if n_gt else 0.0

    classes = sorted(set(labels_gt) | set(labels_pred))
    per_class: dict[str, dict] = {}
    f1s, precisions, recalls = [], [], []
    for c in classes:
        tp_c = sum(1 for _, gi, _ in matches if labels_gt[gi] == c)
        n_pred_c = sum(1 for l in labels_pred if l == c)
        n_gt_c = sum(1 for l in labels_gt if l == c)
        fp_c, fn_c = n_pred_c - tp_c, n_gt_c - tp_c
        p_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) else 0.0
        r_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) else 0.0
        f_c = _f1(p_c, r_c)
        per_class[c] = {"tp": tp_c, "fp": fp_c, "fn": fn_c, "n_gt": n_gt_c,
                        "n_pred": n_pred_c, "precision": p_c, "recall": r_c, "f1": f_c}
        f1s.append(f_c); precisions.append(p_c); recalls.append(r_c)

    return {
        "micro_precision": micro_precision, "micro_recall": micro_recall,
        "micro_f1": _f1(micro_precision, micro_recall),
        "macro_precision": float(np.mean(precisions)) if precisions else float("nan"),
        "macro_recall": float(np.mean(recalls)) if recalls else float("nan"),
        "macro_f1": float(np.mean(f1s)) if f1s else float("nan"),
        "matched": tp, "per_class": per_class,
    }


def _label_aware_metrics(preds: List[SceneObject], gts: List[SceneObject],
                         iou: np.ndarray, iou_threshold: float) -> dict:
    """Label-aware micro/macro/per-class F1 under OUR OBB-IoU matching (a match needs
    IoU >= threshold AND a label match). This is our stricter, non-coworker protocol;
    the coworker-comparable centroid-matched versions are in :func:`_centroid_metrics`."""
    labels_pred = [_norm_label(p.label) for p in preds]
    labels_gt = [_norm_label(g.label) for g in gts]
    matches = hungarian_match(iou, iou_threshold, require_label=True,
                              labels_pred=labels_pred, labels_gt=labels_gt)
    out = _micro_macro_from_matches(matches, labels_pred, labels_gt)
    out["label_aware_matched"] = out.pop("matched")
    return out


def greedy_match_centroid(dist: np.ndarray, tau: float, require_label: bool = False,
                          labels_pred=None, labels_gt=None):
    """GREEDY one-to-one matching on 2D-XY centroid distance; keep pairs with dist <= tau.

    Matches the coworker's shipped default (``harness.py`` -> ``ObjectMatchThresholds()``:
    ``use_hungarian=False``, 2D top-down centroid, ``default_distance_m=1.0``). Candidate
    pairs are taken in ascending distance and assigned first-come, each pred/gt used once.
    ``dist`` must already be the XY (top-down) distance matrix. Returns
    ``[(pred_idx, gt_idx, dist), ...]``.
    """
    n_pred, n_gt = dist.shape
    cand = []
    for i in range(n_pred):
        for j in range(n_gt):
            if dist[i, j] <= tau and (not require_label or labels_pred[i] == labels_gt[j]):
                cand.append((float(dist[i, j]), i, j))
    cand.sort()
    used_p, used_g, matches = set(), set(), []
    for d, i, j in cand:
        if i in used_p or j in used_g:
            continue
        used_p.add(i); used_g.add(j)
        matches.append((i, j, d))
    return matches


def _xy_dist_matrix(preds, gts):
    """Top-down (X,Y) Euclidean distance matrix between pred and gt object centroids."""
    if not preds or not gts:
        return np.zeros((len(preds), len(gts)), dtype=np.float64)
    pc = np.array([np.asarray(p.T_world_obj)[:2, 3] for p in preds], dtype=np.float64)
    gc = np.array([np.asarray(g.T_world_obj)[:2, 3] for g in gts], dtype=np.float64)
    return cdist(pc, gc)


def _centroid_metrics(preds: List[SceneObject], gts: List[SceneObject],
                      taus=(0.25, 0.5, 0.75, 1.0, 1.5), primary_tau: float = 1.0) -> dict:
    """Coworker-comparable object metrics (``cd_`` prefix), confirmed vs source (AI-7).

    Matching = GREEDY on 2D top-down centroid distance <= tau (default 1 m), swept over
    ``taus``. At ``primary_tau``: label-agnostic precision/recall/F1 and label-aware
    micro/macro F1 (their Object Micro/Macro F1; macro over GT union pred categories).
    ``cd_micro_f1_many_to_one`` is their over-segmentation-tolerant object F1 (any-overlap:
    a GT counts if it has >=1 valid same-category detection; a detection counts if it has
    >=1 valid same-category GT). ``class_free_recall_1m`` = fraction of GT with ANY predicted
    centroid within 1 m (label ignored). NB categories use our normalised (lowercased) labels;
    the coworker's published run buckets on raw case-sensitive strings -- a divergence only if
    the two sides disagree on casing (ours are internally consistent).
    """
    n_pred, n_gt = len(preds), len(gts)
    labels_pred = [_norm_label(p.label) for p in preds]
    labels_gt = [_norm_label(g.label) for g in gts]
    dist = _xy_dist_matrix(preds, gts)

    def _prf(matches):
        tp = len(matches)
        p = tp / n_pred if n_pred else 0.0
        r = tp / n_gt if n_gt else 0.0
        return tp, p, r, _f1(p, r)

    tp, prec, rec, f1 = _prf(greedy_match_centroid(dist, primary_tau))
    lbl_matches = greedy_match_centroid(dist, primary_tau, require_label=True,
                                        labels_pred=labels_pred, labels_gt=labels_gt)
    mm = _micro_macro_from_matches(lbl_matches, labels_pred, labels_gt)

    # many-to-one (any-overlap, within category): GT with >=1 valid same-cat detection,
    # detections with >=1 valid same-cat GT. micro_f1 - many_to_one = over-segmentation cost.
    gt_any, hy_any = set(), set()
    if n_pred and n_gt:
        for i in range(n_pred):
            for j in range(n_gt):
                if dist[i, j] <= primary_tau and labels_pred[i] == labels_gt[j]:
                    hy_any.add(i); gt_any.add(j)
    m2o_recall = len(gt_any) / n_gt if n_gt else 0.0
    m2o_precision = len(hy_any) / n_pred if n_pred else 0.0

    class_free_recall_1m = (float(np.mean(dist.min(axis=0) <= 1.0))
                            if (n_pred and n_gt) else (0.0 if n_gt else float("nan")))

    f1_by_tau, recall_by_tau = {}, {}
    for t in taus:
        _, _, rt, ft = _prf(greedy_match_centroid(dist, t))
        f1_by_tau[f"{t}"] = ft
        recall_by_tau[f"{t}"] = rt

    return {
        "cd_tau": primary_tau,
        "cd_precision": prec, "cd_recall": rec, "cd_f1": f1, "cd_matched": tp,
        "cd_micro_f1": mm["micro_f1"], "cd_micro_precision": mm["micro_precision"],
        "cd_micro_recall": mm["micro_recall"], "cd_macro_f1": mm["macro_f1"],
        "cd_macro_precision": mm["macro_precision"], "cd_macro_recall": mm["macro_recall"],
        "cd_per_class": mm["per_class"],
        "cd_micro_f1_many_to_one": _f1(m2o_precision, m2o_recall),
        "cd_micro_precision_many_to_one": m2o_precision,
        "cd_micro_recall_many_to_one": m2o_recall,
        "class_free_recall_1m": class_free_recall_1m,
        "centroid_f1_by_tau": f1_by_tau, "centroid_recall_by_tau": recall_by_tau,
    }


def _scene_geometry(preds: List[SceneObject], gts: List[SceneObject],
                    surface_points: int, taus=(0.05, 0.02)) -> dict:
    """Scene-level, class-free geometry: all predicted surface points vs all GT points.

    No matching and no labels -- the coworker-comparable whole-scene Chamfer and
    geometric coverage. Returns an empty dict when either side has no mesh (so the
    keys are simply absent and aggregate/table skip them) -- e.g. before GT meshes
    are attached (AI-8).
    """
    pool_n = min(surface_points, 5000)  # bound the pooled cloud across many objects
    pred_pts = [geo.sample_surface(p.mesh, pool_n, seed=i)
                for i, p in enumerate(preds) if p.mesh is not None]
    gt_pts = [geo.sample_surface(g.mesh, pool_n, seed=1000 + j)
              for j, g in enumerate(gts) if g.mesh is not None]
    pred_pts = [a for a in pred_pts if len(a)]
    gt_pts = [a for a in gt_pts if len(a)]
    if not pred_pts or not gt_pts:
        return {}
    cf = geo.chamfer_and_fscore(np.vstack(pred_pts), np.vstack(gt_pts), taus=taus)
    out = {"scene_chamfer_mean_m": cf["chamfer_mean"]}
    # surf_*@tau = SURFACE-reconstruction point-coverage (Tanks-and-Temples heritage), NOT the
    # coworker's Class-Free Geo Recall (that is 1m-centroid class_free_recall_1m). Kept distinct.
    for tau in taus:
        out[f"surf_recall@{tau}"] = cf[f"recall@{tau}"]
        out[f"surf_precision@{tau}"] = cf[f"precision@{tau}"]
        out[f"surf_fscore@{tau}"] = cf[f"fscore@{tau}"]
    return out


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
    chamfer, chamfer_mean, f5, f2 = [], [], [], []
    scan2cad_hits = 0
    for pi, gi, _ in matches:
        p, g = preds[pi], gts[gi]
        c_err = float(np.linalg.norm(p.T_world_obj[:3, 3] - g.T_world_obj[:3, 3]))
        sym = symmetry_for_label(g.label)
        # axis-labeling-invariant orientation + scale (min-volume OBB axes are
        # unordered; naive R-vs-R comparison overstates rotation error massively).
        r_err, s_err = geo.box_pose_error(
            p.T_world_obj[:3, :3], p.extents, g.T_world_obj[:3, :3], g.extents, sym)
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
            chamfer_mean.append(cf["chamfer_mean"])
            f5.append(cf["fscore@0.05"])
            f2.append(cf["fscore@0.02"])

    def _agg(x, fn=np.median):
        return float(fn(x)) if len(x) else float("nan")

    metrics = {
        "n_pred": n_pred,
        "n_gt": n_gt,
        # named per-scene counts (coworker table: matched / objects / predictions per scene)
        "matched_per_scene": tp,
        "objects_per_scene": n_gt,
        "predictions_per_scene": n_pred,
        "count_ratio": (n_pred / n_gt) if n_gt else float("nan"),
        "iou_threshold": iou_threshold,
        "tp": tp,
        "fp": n_pred - tp,
        "fn": n_gt - tp,
        # headline P/R/F1 are LABEL-AGNOSTIC (geometry only); label-aware micro/macro below
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
        "chamfer_symmetric_mean_m": _agg(chamfer_mean),  # coworker "average chamfer" (= l1/2)
        "fscore@0.05_mean": _agg(f5, np.mean),
        "fscore@0.02_mean": _agg(f2, np.mean),
    }
    # label-aware micro/macro under OUR OBB-IoU matching (stricter, non-coworker protocol)
    metrics.update(_label_aware_metrics(preds, gts, iou, iou_threshold))
    # coworker-comparable centroid-distance matching (tau=1m swept): cd_* + class_free_recall_1m
    metrics.update(_centroid_metrics(preds, gts))
    # scene-level, class-free surface geometry (coworker: whole-scene Chamfer). NOTE: surf_*@tau
    # is a SURFACE point-coverage F-score, NOT the coworker's Class-Free Geo Recall (which is the
    # 1m-centroid class_free_recall_1m above) -- kept separate on purpose (AI-7).
    if compute_geometry:
        metrics.update(_scene_geometry(preds, gts, surface_points))
    return metrics
