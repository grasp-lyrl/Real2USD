#!/usr/bin/env python3
"""
Single-run evaluation: load predictions and GT, match, compute metrics,
write by_run JSON with instances, diagnostics, and optional registration/retrieval summaries.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Allow imports from parent (evaluations/) when run as script
import sys
_EVAL_ROOT = Path(__file__).resolve().parent.parent
if str(_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_EVAL_ROOT))

from eval_common import (
    estimate_supervisely_global_yaw,
    load_predictions_clio_graphml,
    load_predictions_scene_graph,
    load_supervisely_gt,
)
from label_matching import load_label_configs, load_learned_aliases
from metrics_3d import greedy_match, pair_iou_3d, strict_relaxed_from_aabb, open_set_metrics, UNLABELED_CLASS

import numpy as np


def _read_json(path: Path) -> Dict:
    with open(path) as f:
        return json.load(f)


def _load_eval_config(config_path: str, eval_dir: str) -> Dict:
    p = Path(config_path)
    if not p.is_absolute():
        p = Path(eval_dir) / p
    with open(p) as f:
        cfg = json.load(f)
    # Resolve paths relative to config dir
    base = p.parent
    for key in ("canonical_labels", "label_aliases", "learned_label_aliases"):
        if key in cfg.get("paths", {}):
            rel = cfg["paths"][key]
            if not Path(rel).is_absolute():
                cfg["paths"][key] = str((base / rel).resolve())
    return cfg


def _filter_predictions_by_config(preds: List[Dict], config: Dict) -> Tuple[List[Dict], Dict]:
    """Filter predictions by prediction.ignore_raw_labels, ignore_canonical_labels, ignore_raw_contains."""
    pred_cfg = config.get("prediction", {}) or {}
    ignore_raw: List[str] = list(pred_cfg.get("ignore_raw_labels") or [])
    ignore_raw_contains: List[str] = list(pred_cfg.get("ignore_raw_contains") or [])
    ignore_canonical: List[str] = list(pred_cfg.get("ignore_canonical_labels") or [])

    def norm(s: str) -> str:
        return (s or "").strip().lower()

    ignore_raw_norm = {norm(x) for x in ignore_raw}
    ignore_canonical_norm = {norm(x) for x in ignore_canonical}

    kept = []
    for p in preds:
        raw = norm(p.get("label_raw") or p.get("slot_label_raw") or "")
        can = norm(p.get("label_canonical") or "")
        if raw in ignore_raw_norm:
            continue
        if can and can in ignore_canonical_norm:
            continue
        if ignore_raw_contains and any(sub in raw for sub in ignore_raw_contains):
            continue
        kept.append(p)

    return kept, {"num_predictions_ignored": len(preds) - len(kept)}


def _compute_fp_breakdown(
    preds: List[Dict],
    gts: List[Dict],
    matches: List[Dict],
    unmatched_pred_indices: List[int],
    iou_threshold: float,
    require_label_match: bool,
    iou_mode: str,
) -> Dict[str, int]:
    wrong_label, low_iou, no_overlap, unresolved = 0, 0, 0, 0
    matched_gt = {m["gt_idx"] for m in matches}
    for pi in unmatched_pred_indices:
        pred = preds[pi]
        lbl = pred.get("label_canonical")
        if not (lbl and str(lbl).strip()):
            unresolved += 1
            continue
        best_iou = 0.0
        best_gt_idx = None
        for gi, gt in enumerate(gts):
            iou = pair_iou_3d(pred, gt, iou_mode=iou_mode)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gi
        if best_iou < iou_threshold:
            if best_iou > 0:
                low_iou += 1
            else:
                no_overlap += 1
        else:
            if require_label_match and best_gt_idx is not None:
                if pred.get("label_canonical") != gts[best_gt_idx].get("label_canonical"):
                    wrong_label += 1
                else:
                    low_iou += 1  # same label but already taken by another pred
            else:
                no_overlap += 1
    return {
        "wrong_label": wrong_label,
        "low_iou": low_iou,
        "no_overlap": no_overlap,
        "unresolved_label": unresolved,
    }


def _compute_range_buckets(
    preds: List[Dict],
    gts: List[Dict],
    matches: List[Dict],
    unmatched_pred: List[int],
    unmatched_gt: List[int],
    bucket_limits_m: List[float],
    iou_threshold: float,
    require_label_match: bool,
    iou_mode: str,
) -> Dict:
    """bucket_limits_m e.g. [2, 4] -> near [0,2), mid [2,4), far [4, inf). Distance = L2 norm of centroid (XY or XYZ)."""
    if not bucket_limits_m:
        return {}
    names = ["near", "mid", "far"]
    limits = [0.0] + sorted(bucket_limits_m) + [1e9]
    assert len(names) == len(limits) - 1

    def bucket(d: Dict) -> int:
        c = d.get("center")
        if not c:
            return 0
        xy = np.asarray(c[:2], dtype=np.float64)
        dist = float(np.linalg.norm(xy))
        for i in range(len(limits) - 1):
            if limits[i] <= dist < limits[i + 1]:
                return i
        return len(names) - 1

    gt_buckets = [bucket(gt) for gt in gts]
    matched_gt_set = {m["gt_idx"] for m in matches}
    pred_buckets = [bucket(pred) for pred in preds]

    out = {n: {"tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "mean_iou_3d": 0.0} for n in names}
    iou_list = {n: [] for n in names}
    for m in matches:
        gi = m["gt_idx"]
        b = gt_buckets[gi]
        n = names[b]
        out[n]["tp"] += 1
        iou_list[n].append(m.get("iou", 0.0))
    for pi in unmatched_pred:
        b = pred_buckets[pi]
        out[names[b]]["fp"] += 1
    for gi in unmatched_gt:
        b = gt_buckets[gi]
        out[names[b]]["fn"] += 1
    for n in names:
        t, f, fn = out[n]["tp"], out[n]["fp"], out[n]["fn"]
        out[n]["precision"] = t / (t + f) if (t + f) > 0 else 0.0
        out[n]["recall"] = t / (t + fn) if (t + fn) > 0 else 0.0
        p, r = out[n]["precision"], out[n]["recall"]
        out[n]["f1"] = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        out[n]["mean_iou_3d"] = float(np.mean(iou_list[n])) if iou_list[n] else 0.0
    return {"buckets": out}


def _registration_summary_from_run_dir(run_dir: Path, preds: List[Dict], matches: List[Dict]) -> Dict:
    """Fill registration_summary from run_dir/diagnostics/registration_metrics.jsonl if present."""
    summary = {}
    jsonl_path = run_dir / "diagnostics" / "registration_metrics.jsonl"
    if not jsonl_path.exists():
        return summary
    deltas = []
    try:
        for line in jsonl_path.read_text().strip().splitlines():
            if not line:
                continue
            rec = json.loads(line)
            dbg = rec.get("debug") or {}
            t = dbg.get("translation_delta_m")
            if t is not None:
                deltas.append(float(t))
    except Exception:
        pass
    if deltas:
        summary["mean_centroid_error_delta_m"] = float(np.mean(deltas))
        summary["n_with_registration"] = len(deltas)
    # helped_rate / hurt_rate require before-vs-after vs GT; need pipeline vs sam3d_only comparison
    return summary


def _retrieval_summary(preds: List[Dict], matched_pred_indices: Set[int]) -> Dict:
    swapped = [i for i, p in enumerate(preds) if p.get("retrieval_swapped")]
    if not swapped:
        return {"swapped_rate": 0.0, "swapped_tp_rate": 0.0}
    swapped_tp = sum(1 for i in swapped if i in matched_pred_indices)
    return {
        "swapped_rate": len(swapped) / len(preds) if preds else 0.0,
        "swapped_tp_rate": swapped_tp / len(swapped) if swapped else 0.0,
    }


def evaluate(
    preds: List[Dict],
    gts: List[Dict],
    config: Dict,
    run_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    match_cfg = config.get("matching", {}) or {}
    iou_threshold = float(match_cfg.get("iou_threshold", 0.1))
    require_label_match = bool(match_cfg.get("require_label_match", False))
    iou_mode = str(match_cfg.get("iou_mode", "aabb")).strip().lower()
    range_buckets_m = list(match_cfg.get("range_buckets_m") or [2, 4])
    compute_open_set = bool(match_cfg.get("compute_open_set_metrics", True))
    os_similarity_threshold = float(match_cfg.get("os_similarity_threshold", 0.9))

    matches, unmatched_pred, unmatched_gt, _ = greedy_match(
        preds, gts, iou_threshold=iou_threshold, require_label_match=require_label_match, iou_mode=iou_mode
    )
    tp = len(matches)
    fp = len(unmatched_pred)
    fn = len(unmatched_gt)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    mean_iou_3d = float(np.mean([m.get("iou", 0.0) for m in matches])) if matches else 0.0
    mean_centroid_error_m = float(np.mean([m.get("centroid_error_m", 0.0) for m in matches])) if matches else 0.0

    # Strict/relaxed (over matched pairs)
    strict_acc = [m.get("strict", False) for m in matches]
    relaxed_acc = [m.get("relaxed", False) for m in matches]
    strict_accuracy = float(np.mean(strict_acc)) if strict_acc else 0.0
    relaxed_accuracy = float(np.mean(relaxed_acc)) if relaxed_acc else 0.0
    f1_relaxed = (
        2 * relaxed_accuracy * mean_iou_3d / (relaxed_accuracy + mean_iou_3d)
        if (relaxed_accuracy + mean_iou_3d) > 0
        else 0.0
    )

    # Per-class: use sentinel for missing labels; we exclude it from output and macro so "unknown" is not a class
    per_class: Dict[str, Dict] = {}
    for m in matches:
        gt = gts[m["gt_idx"]]
        lbl = (gt.get("label_canonical") or "").strip() or UNLABELED_CLASS
        if lbl not in per_class:
            per_class[lbl] = {
                "tp": 0, "fp": 0, "fn": 0,
                "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "mean_iou_3d": 0.0, "strict_accuracy": 0.0, "relaxed_accuracy": 0.0, "mean_centroid_error_m": 0.0,
                "ious": [], "strict_list": [], "relaxed_list": [], "centroid_errors": [],
            }
        per_class[lbl]["tp"] += 1
        per_class[lbl]["ious"].append(m.get("iou", 0.0))
        per_class[lbl]["strict_list"].append(m.get("strict", False))
        per_class[lbl]["relaxed_list"].append(m.get("relaxed", False))
        per_class[lbl]["centroid_errors"].append(m.get("centroid_error_m", 0.0))
    gt_labels = set((gt.get("label_canonical") or "").strip() or UNLABELED_CLASS for gt in gts)
    for lbl in gt_labels:
        if lbl not in per_class:
            per_class[lbl] = {
                "tp": 0, "fp": 0, "fn": 0,
                "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "mean_iou_3d": 0.0, "strict_accuracy": 0.0, "relaxed_accuracy": 0.0, "mean_centroid_error_m": 0.0,
                "ious": [], "strict_list": [], "relaxed_list": [], "centroid_errors": [],
            }
    for pi in unmatched_pred:
        lbl = (preds[pi].get("label_canonical") or "").strip() or UNLABELED_CLASS
        if lbl not in per_class:
            per_class[lbl] = {
                "tp": 0, "fp": 0, "fn": 0,
                "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "mean_iou_3d": 0.0, "strict_accuracy": 0.0, "relaxed_accuracy": 0.0, "mean_centroid_error_m": 0.0,
                "ious": [], "strict_list": [], "relaxed_list": [], "centroid_errors": [],
            }
        per_class[lbl]["fp"] += 1
    for gi in unmatched_gt:
        lbl = (gts[gi].get("label_canonical") or "").strip() or UNLABELED_CLASS
        per_class[lbl]["fn"] += 1
    for lbl, pc in per_class.items():
        t, f, fn = pc["tp"], pc["fp"], pc["fn"]
        pc["precision"] = t / (t + f) if (t + f) > 0 else 0.0
        pc["recall"] = t / (t + fn) if (t + fn) > 0 else 0.0
        p, r = pc["precision"], pc["recall"]
        pc["f1"] = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        pc["mean_iou_3d"] = float(np.mean(pc["ious"])) if pc["ious"] else 0.0
        pc["strict_accuracy"] = float(np.mean(pc["strict_list"])) if pc["strict_list"] else 0.0
        pc["relaxed_accuracy"] = float(np.mean(pc["relaxed_list"])) if pc["relaxed_list"] else 0.0
        pc["mean_centroid_error_m"] = float(np.mean(pc["centroid_errors"])) if pc["centroid_errors"] else 0.0
        for k in ("ious", "strict_list", "relaxed_list", "centroid_errors"):
            del pc[k]

    # Exclude unknown/unlabeled from per_class output so evals only report real classes
    for skip in (UNLABELED_CLASS, "unknown"):
        per_class.pop(skip, None)

    # Macro-average: only over known classes (exclude unknown/unlabeled)
    def _is_known_class(lbl: str) -> bool:
        s = (lbl or "").strip().lower()
        return bool(s) and s != "unknown" and lbl != UNLABELED_CLASS

    classes_with_matches = [lbl for lbl, pc in per_class.items() if pc.get("tp", 0) >= 1 and _is_known_class(lbl)]
    if classes_with_matches:
        mean_strict_accuracy_per_class = float(np.mean([per_class[lbl]["strict_accuracy"] for lbl in classes_with_matches]))
        mean_relaxed_accuracy_per_class = float(np.mean([per_class[lbl]["relaxed_accuracy"] for lbl in classes_with_matches]))
        mean_iou_3d_per_class = float(np.mean([per_class[lbl]["mean_iou_3d"] for lbl in classes_with_matches]))
        mean_centroid_error_m_per_class = float(np.mean([per_class[lbl]["mean_centroid_error_m"] for lbl in classes_with_matches]))
    else:
        mean_strict_accuracy_per_class = 0.0
        mean_relaxed_accuracy_per_class = 0.0
        mean_iou_3d_per_class = 0.0
        mean_centroid_error_m_per_class = 0.0

    fp_breakdown = _compute_fp_breakdown(
        preds, gts, matches, unmatched_pred, iou_threshold, require_label_match, iou_mode
    )
    range_metrics = _compute_range_buckets(
        preds, gts, matches, unmatched_pred, unmatched_gt, range_buckets_m,
        iou_threshold, require_label_match, iou_mode
    )
    matched_pred_set = {m["pred_idx"] for m in matches}
    retrieval_summary = _retrieval_summary(preds, matched_pred_set)
    registration_summary = _registration_summary_from_run_dir(run_dir or Path(), preds, matches) if run_dir else {}

    benchmark = {
        "strict_accuracy": strict_accuracy,
        "relaxed_accuracy": relaxed_accuracy,
        "strict_precision": sum(strict_acc) / len(preds) if preds else 0.0,
        "relaxed_precision": sum(relaxed_acc) / len(preds) if preds else 0.0,
        "f1_relaxed": f1_relaxed,
        "num_objects_in_map": len(preds),
        "mean_strict_accuracy_per_class": mean_strict_accuracy_per_class,
        "mean_relaxed_accuracy_per_class": mean_relaxed_accuracy_per_class,
        "mean_iou_3d_per_class": mean_iou_3d_per_class,
        "mean_centroid_error_m_per_class": mean_centroid_error_m_per_class,
    }
    if compute_open_set:
        os_metrics = open_set_metrics(
            preds, gts,
            iou_threshold=iou_threshold,
            iou_mode=iou_mode,
            similarity_threshold=os_similarity_threshold,
        )
        benchmark.update(os_metrics)

    return {
        "counts": {"tp": tp, "fp": fp, "fn": fn},
        "detection_metrics": {"precision": precision, "recall": recall, "f1": f1},
        "geometry_metrics": {"mean_iou_3d": mean_iou_3d, "mean_centroid_error_m": mean_centroid_error_m},
        "per_class": per_class,
        "fp_breakdown": fp_breakdown,
        "range_metrics": range_metrics,
        "benchmark_compat": benchmark,
        "matching": {"iou_threshold": iou_threshold, "require_label_match": require_label_match, "iou_mode": iou_mode},
        "instances": {"predictions": preds, "gts": gts},
        "matches": matches,
        "unmatched_prediction_indices": unmatched_pred,
        "unmatched_gt_indices": unmatched_gt,
        "diagnostics": {
            "registration_summary": registration_summary,
            "retrieval_summary": retrieval_summary,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation for one prediction set, write by_run JSON.")
    parser.add_argument("--prediction-type", default="scene_graph", choices=("scene_graph", "clio"))
    parser.add_argument("--prediction-path", default=None, help="scene_graph.json or CLIO graphml")
    parser.add_argument("--prediction-json", default=None, help="Alias for --prediction-path (for backward compat)")
    parser.add_argument("--gt-json", required=True)
    parser.add_argument("--run-dir", default=None, help="Run directory (for registration diagnostics; default: parent of prediction-path)")
    parser.add_argument("--scene", default="unknown_scene")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--method-tag", default="")
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--eval-config", default=None)
    parser.add_argument("--frame-normalization", default=None, choices=("none", "gt_global_yaw"),
                        help="Override eval_config: 'none' = no rotation, 'gt_global_yaw' = align to dominant GT yaw")
    parser.add_argument("--no-ignore-labels", action="store_true",
                        help="Include all predictions: clear ignore_raw_labels, ignore_raw_contains, ignore_canonical_labels")
    parser.add_argument("--sensor-source", default="unknown")
    args = parser.parse_args()
    pred_path = args.prediction_path or args.prediction_json
    if not pred_path:
        parser.error("Provide --prediction-path or --prediction-json")

    eval_dir = str(_EVAL_ROOT)
    config_path = args.eval_config or str(_EVAL_ROOT / "eval_config.json")
    config = _load_eval_config(config_path, eval_dir)
    if args.no_ignore_labels:
        config = copy.deepcopy(config)
        config.setdefault("prediction", {})
        config["prediction"]["ignore_raw_labels"] = []
        config["prediction"]["ignore_raw_contains"] = []
        config["prediction"]["ignore_canonical_labels"] = []
    run_dir = Path(args.run_dir) if args.run_dir else Path(pred_path).resolve().parent
    results_root = Path(args.results_root)
    run_id = args.run_id or run_dir.name

    canonical_path = config["paths"]["canonical_labels"]
    aliases_path = config["paths"]["label_aliases"]
    learned_aliases_path = config["paths"]["learned_label_aliases"]
    canonical, alias_map, contains_rules = load_label_configs(canonical_path, aliases_path)
    learned_aliases = load_learned_aliases(learned_aliases_path)
    label_cfg = config.get("label_resolution", {}) or {}
    use_embedding = bool(label_cfg.get("use_embedding_for_unresolved", False))
    learn_new = bool(label_cfg.get("learn_new_aliases", False))
    embedding_min = float(label_cfg.get("embedding_min_score", 0.5))

    frame_norm = args.frame_normalization
    if frame_norm is None:
        frame_norm = str(config.get("matching", {}).get("frame_normalization", "none")).strip().lower()
    gt_global_yaw_rad = estimate_supervisely_global_yaw(args.gt_json) if frame_norm == "gt_global_yaw" else 0.0

    if args.prediction_type == "scene_graph":
        preds_all = load_predictions_scene_graph(
            pred_path,
            canonical=canonical,
            alias_map=alias_map,
            contains_rules=contains_rules,
            learned_alias_map=learned_aliases,
            use_embedding_for_unresolved=use_embedding,
            learn_new_aliases=learn_new,
            learned_alias_path=learned_aliases_path,
            embedding_min_score=embedding_min,
            label_source=config.get("prediction", {}).get("label_source", "prefer_retrieved_else_label"),
            normalize_yaw_rad=gt_global_yaw_rad,
        )
    else:
        preds_all = load_predictions_clio_graphml(
            pred_path,
            canonical=canonical,
            alias_map=alias_map,
            contains_rules=contains_rules,
            learned_alias_map=learned_aliases,
            use_embedding_for_unresolved=use_embedding,
            learn_new_aliases=learn_new,
            learned_alias_path=learned_aliases_path,
            embedding_min_score=embedding_min,
            normalize_yaw_rad=gt_global_yaw_rad,
        )
    preds, pred_filter = _filter_predictions_by_config(preds_all, config)

    gts = load_supervisely_gt(
        args.gt_json,
        canonical=canonical,
        alias_map=alias_map,
        contains_rules=contains_rules,
        learned_alias_map=learned_aliases,
        use_embedding_for_unresolved=use_embedding,
        learn_new_aliases=learn_new,
        learned_alias_path=learned_aliases_path,
        embedding_min_score=embedding_min,
        normalize_yaw_rad=gt_global_yaw_rad,
    )

    result = evaluate(preds, gts, config, run_dir=run_dir)
    result["prediction_filter"] = pred_filter
    result["metadata"] = {
        "scene": args.scene,
        "run_id": run_id,
        "method_tag": args.method_tag or None,
        "sensor_source": args.sensor_source,
        "prediction_type": args.prediction_type,
        "prediction_path": str(Path(pred_path).resolve()),
        "gt_json": str(Path(args.gt_json).resolve()),
        "results_root": str(results_root.resolve()),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "frame_normalization": {"mode": frame_norm, "gt_global_yaw_deg": float(gt_global_yaw_rad * 180.0 / np.pi)},
    }
    h = hashlib.sha256(json.dumps({k: result[k] for k in ("counts", "detection_metrics", "metadata")}, sort_keys=True).encode()).hexdigest()[:12]
    result["metadata"]["eval_hash"] = h

    by_run_dir = results_root / "by_run"
    by_run_dir.mkdir(parents=True, exist_ok=True)
    scene_slug = args.scene.replace(" ", "_")
    ts = result["metadata"]["created_at"][:19].replace(":", "").replace("-", "").replace("T", "_")
    tag = (args.method_tag or "default").replace(" ", "_")
    out_name = f"{scene_slug}_{run_id}_{tag}_{ts}_{h}.json"
    out_path = by_run_dir / out_name
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
