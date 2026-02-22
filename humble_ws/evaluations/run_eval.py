import argparse
import csv
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

from eval_common import (
    estimate_supervisely_global_yaw,
    load_predictions_clio_graphml,
    load_predictions_scene_graph,
    load_supervisely_gt,
)
from label_matching import default_config_paths, load_label_configs, load_learned_aliases
from metrics_3d import greedy_match, pair_iou_3d


def _read_json(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def _load_eval_config(eval_config_path: str, eval_dir: str) -> Dict:
    cfg = _read_json(eval_config_path)
    paths = cfg.get("paths", {})
    c_default, a_default = default_config_paths(eval_dir)
    cfg["paths"] = {
        "canonical_labels": str(Path(paths.get("canonical_labels", c_default)).resolve()),
        "label_aliases": str(Path(paths.get("label_aliases", a_default)).resolve()),
        "learned_label_aliases": str(
            Path(paths.get("learned_label_aliases", str((Path(eval_dir) / "learned_label_aliases.json")))).resolve()
        ),
    }
    return cfg


def _hash_inputs(pred_path: str, gt_path: str, eval_config: Dict, canonical: Dict, aliases: Dict) -> str:
    h = hashlib.sha256()
    for p in (pred_path, gt_path):
        with open(p, "rb") as f:
            h.update(f.read())
    h.update(json.dumps(eval_config, sort_keys=True).encode("utf-8"))
    h.update(json.dumps(canonical, sort_keys=True).encode("utf-8"))
    h.update(json.dumps(aliases, sort_keys=True).encode("utf-8"))
    return h.hexdigest()[:12]


def _strict_relaxed_precision(matches: List[Dict], num_predictions: int) -> Dict:
    strict = sum(1 for m in matches if m.get("strict"))
    relaxed = sum(1 for m in matches if m.get("relaxed"))
    strict_precision = (strict / num_predictions) if num_predictions > 0 else 0.0
    relaxed_precision = (relaxed / num_predictions) if num_predictions > 0 else 0.0
    return {
        "strict_matches": strict,
        "relaxed_matches": relaxed,
        "strict_precision": strict_precision,
        "relaxed_precision": relaxed_precision,
    }


def _extract_performance(run_dir: Path) -> Dict:
    out = {
        "e2e_latency_ms": None,
        "e2e_rate_hz": None,
        "sam3d_inference_ms": None,
        "sam3d_inference_hz": None,
        "cpu_peak_mb": None,
        "gpu_peak_mb": None,
    }
    timing_summary = run_dir / "timing_summary.json"
    if not timing_summary.exists():
        return out
    try:
        data = _read_json(str(timing_summary))
    except Exception:
        return out

    # Handle both list and dict layouts.
    rows = []
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        if "summary" in data and isinstance(data["summary"], list):
            rows = data["summary"]
        else:
            for k, v in data.items():
                if isinstance(v, dict):
                    node_name = v.get("node_name")
                    step_name = v.get("step_name")
                    if not node_name and "|" in k:
                        node_name, step_name = k.split("|", 1)
                    if node_name and step_name:
                        row = dict(v)
                        row["node_name"] = node_name
                        row["step_name"] = step_name
                        rows.append(row)

    for r in rows:
        node = str(r.get("node_name", ""))
        step = str(r.get("step_name", ""))
        mean_ms = r.get("mean_ms")
        if mean_ms is None:
            continue
        mean_ms = float(mean_ms)
        if step == "frame_total" and ("lidar_cam_node" in node or "realsense_cam_node" in node):
            out["e2e_latency_ms"] = mean_ms
            out["e2e_rate_hz"] = (1000.0 / mean_ms) if mean_ms > 0 else None
        if "sam3d_worker" in node and step == "inference":
            out["sam3d_inference_ms"] = mean_ms
            out["sam3d_inference_hz"] = (1000.0 / mean_ms) if mean_ms > 0 else None
    return out


def _load_pose_json(p: Path) -> Dict:
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return {}


def _registration_retrieval_diagnostics(
    predictions: List[Dict],
    gts: List[Dict],
    matches: List[Dict],
    unmatched_pred: List[int],
    run_dir: Path,
) -> Dict:
    # Retrieval summary.
    swapped_idx = [i for i, p in enumerate(predictions) if bool(p.get("retrieval_swapped", False))]
    matched_pred_idx = {m["pred_idx"] for m in matches}
    unmatched_pred_idx = set(unmatched_pred)
    swapped_matched = sum(1 for i in swapped_idx if i in matched_pred_idx)
    swapped_unmatched = sum(1 for i in swapped_idx if i in unmatched_pred_idx)
    retrieval = {
        "swapped_count": int(len(swapped_idx)),
        "swapped_rate": float(len(swapped_idx) / len(predictions)) if predictions else 0.0,
        "swapped_matched_tp_count": int(swapped_matched),
        "swapped_unmatched_fp_count": int(swapped_unmatched),
        "swapped_tp_rate": float(swapped_matched / len(swapped_idx)) if swapped_idx else 0.0,
    }

    # Registration helped/hurt from initial_position -> final prediction center error.
    pose_cache: Dict[str, Dict] = {}

    def _initial_position_for_pred(pred: Dict):
        job_id = pred.get("job_id")
        pose_path = None
        if job_id:
            pose_path = run_dir / "output" / str(job_id) / "pose.json"
        if pose_path is None or not pose_path.exists():
            dp = pred.get("data_path")
            if dp:
                pose_path = Path(dp).parent / "pose.json"
        if pose_path is None or not pose_path.exists():
            return None
        key = str(pose_path.resolve())
        if key not in pose_cache:
            pose_cache[key] = _load_pose_json(pose_path)
        pos = pose_cache[key].get("initial_position")
        if not isinstance(pos, list) or len(pos) < 3:
            return None
        return np.asarray(pos[:3], dtype=np.float64)

    improved = 0
    worsened = 0
    unchanged = 0
    deltas = []
    evaluated = 0
    for m in matches:
        pred = predictions[m["pred_idx"]]
        gt = gts[m["gt_idx"]]
        init_pos = _initial_position_for_pred(pred)
        if init_pos is None:
            continue
        gt_center = np.asarray(gt["center"], dtype=np.float64)
        final_center = np.asarray(pred["center"], dtype=np.float64)
        init_err = float(np.linalg.norm(init_pos - gt_center))
        final_err = float(np.linalg.norm(final_center - gt_center))
        delta = init_err - final_err
        deltas.append(delta)
        evaluated += 1
        if delta > 1e-6:
            improved += 1
        elif delta < -1e-6:
            worsened += 1
        else:
            unchanged += 1
    registration = {
        "evaluated_count": int(evaluated),
        "helped_count": int(improved),
        "hurt_count": int(worsened),
        "unchanged_count": int(unchanged),
        "helped_rate": float(improved / evaluated) if evaluated > 0 else 0.0,
        "hurt_rate": float(worsened / evaluated) if evaluated > 0 else 0.0,
        "mean_centroid_error_delta_m": float(np.mean(deltas)) if deltas else 0.0,
    }

    return {"retrieval_summary": retrieval, "registration_summary": registration}


def evaluate(predictions: List[Dict], gts: List[Dict], cfg: Dict, run_dir: Path) -> Dict:
    mcfg = cfg.get("matching", {})
    iou_threshold = float(mcfg.get("iou_threshold", 0.25))
    require_label_match = bool(mcfg.get("require_label_match", True))
    iou_mode = str(mcfg.get("iou_mode", "aabb")).strip().lower()
    matches, unmatched_pred, unmatched_gt, _ = greedy_match(
        predictions,
        gts,
        iou_threshold=iou_threshold,
        require_label_match=require_label_match,
        iou_mode=iou_mode,
    )
    tp = len(matches)
    fp = len(unmatched_pred)
    fn = len(unmatched_gt)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    mean_iou = float(np.mean([m["iou"] for m in matches])) if matches else 0.0
    mean_centroid_error = float(np.mean([m["centroid_error_m"] for m in matches])) if matches else 0.0

    strict_relaxed = _strict_relaxed_precision(matches, len(predictions))
    strict_accuracy = strict_relaxed["strict_matches"] / len(gts) if gts else 0.0
    relaxed_accuracy = strict_relaxed["relaxed_matches"] / len(gts) if gts else 0.0
    relaxed_f1 = (
        2 * strict_relaxed["relaxed_precision"] * relaxed_accuracy
        / (strict_relaxed["relaxed_precision"] + relaxed_accuracy)
        if (strict_relaxed["relaxed_precision"] + relaxed_accuracy) > 0
        else 0.0
    )

    # FP reason breakdown.
    fp_breakdown = {"wrong_label": 0, "low_iou": 0, "no_overlap": 0, "unresolved_label": 0}
    for pi in unmatched_pred:
        p = predictions[pi]
        if p.get("label_canonical") is None:
            fp_breakdown["unresolved_label"] += 1
            continue
        best_iou, best_gt_label = 0.0, None
        for gt in gts:
            iou = pair_iou_3d(p, gt, iou_mode=iou_mode)
            if iou > best_iou:
                best_iou = iou
                best_gt_label = gt.get("label_canonical")
        if best_iou <= 0.0:
            fp_breakdown["no_overlap"] += 1
        elif best_gt_label != p.get("label_canonical"):
            fp_breakdown["wrong_label"] += 1
        else:
            fp_breakdown["low_iou"] += 1

    # Per-class TP/FP/FN and P/R/F1.
    class_labels = sorted(
        {
            x.get("label_canonical")
            for x in (predictions + gts)
            if x.get("label_canonical") is not None
        }
    )
    per_class = {}
    for c in class_labels:
        tp_c = sum(1 for m in matches if predictions[m["pred_idx"]].get("label_canonical") == c)
        fp_c = sum(1 for i in unmatched_pred if predictions[i].get("label_canonical") == c)
        fn_c = sum(1 for i in unmatched_gt if gts[i].get("label_canonical") == c)
        p_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
        r_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
        f1_c = (2 * p_c * r_c / (p_c + r_c)) if (p_c + r_c) > 0 else 0.0
        ious_c = [m["iou"] for m in matches if predictions[m["pred_idx"]].get("label_canonical") == c]
        per_class[c] = {
            "tp": tp_c,
            "fp": fp_c,
            "fn": fn_c,
            "precision": p_c,
            "recall": r_c,
            "f1": f1_c,
            "mean_iou_3d": float(np.mean(ious_c)) if ious_c else 0.0,
        }

    diagnostics = _registration_retrieval_diagnostics(
        predictions=predictions,
        gts=gts,
        matches=matches,
        unmatched_pred=unmatched_pred,
        run_dir=run_dir,
    )

    return {
        "counts": {"tp": tp, "fp": fp, "fn": fn, "num_predictions": len(predictions), "num_gt": len(gts)},
        "detection_metrics": {"precision": precision, "recall": recall, "f1": f1},
        "geometry_metrics": {"mean_iou_3d": mean_iou, "mean_centroid_error_m": mean_centroid_error},
        "matching": {"iou_mode": iou_mode, "iou_threshold": iou_threshold, "require_label_match": require_label_match},
        "benchmark_compat": {
            "iou": mean_iou,
            "strict_accuracy": strict_accuracy,
            "relaxed_accuracy": relaxed_accuracy,
            "strict_precision": strict_relaxed["strict_precision"],
            "relaxed_precision": strict_relaxed["relaxed_precision"],
            "f1_relaxed": relaxed_f1,
            "num_objects_in_map": len(predictions),
        },
        "fp_breakdown": fp_breakdown,
        "per_class": per_class,
        "matches": matches,
        "unmatched_prediction_indices": unmatched_pred,
        "unmatched_gt_indices": unmatched_gt,
        "unmatched_prediction_ids": [predictions[i].get("id") for i in unmatched_pred],
        "unmatched_gt_ids": [gts[i].get("id") for i in unmatched_gt],
        "diagnostics": diagnostics,
    }


def _write_csv_row(csv_path: Path, row: Dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _write_csv_rows(csv_path: Path, rows: List[Dict]):
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _normalize_label_text(text: str) -> str:
    s = (text or "").strip().lower()
    s = s.replace("/", " ").replace("_", " ").replace("-", " ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _filter_predictions_by_config(preds: List[Dict], cfg: Dict) -> (List[Dict], Dict):
    pcfg = cfg.get("prediction", {}) if isinstance(cfg, dict) else {}
    ignore_raw = {_normalize_label_text(x) for x in pcfg.get("ignore_raw_labels", [])}
    ignore_contains = {_normalize_label_text(x) for x in pcfg.get("ignore_raw_contains", [])}
    ignore_canonical = {_normalize_label_text(x) for x in pcfg.get("ignore_canonical_labels", [])}

    filtered = []
    ignored_rows = []
    for p in preds:
        raw_norm = _normalize_label_text(p.get("label_raw", ""))
        canon_norm = _normalize_label_text(p.get("label_canonical", ""))
        reason = None
        if raw_norm and raw_norm in ignore_raw:
            reason = "ignore_raw_label"
        elif raw_norm and any(k and k in raw_norm for k in ignore_contains):
            reason = "ignore_raw_contains"
        elif canon_norm and canon_norm in ignore_canonical:
            reason = "ignore_canonical_label"
        if reason:
            ignored_rows.append(
                {
                    "pred_id": p.get("id"),
                    "job_id": p.get("job_id"),
                    "label_raw": p.get("label_raw"),
                    "label_canonical": p.get("label_canonical"),
                    "reason": reason,
                }
            )
        else:
            filtered.append(p)
    info = {
        "num_predictions_before_filter": len(preds),
        "num_predictions_after_filter": len(filtered),
        "num_predictions_ignored": len(ignored_rows),
        "ignored_predictions": ignored_rows,
        "ignore_config": {
            "ignore_raw_labels": sorted(ignore_raw),
            "ignore_raw_contains": sorted(ignore_contains),
            "ignore_canonical_labels": sorted(ignore_canonical),
        },
    }
    return filtered, info


def _instance_view(x: Dict) -> Dict:
    return {
        "id": x.get("id"),
        "job_id": x.get("job_id"),
        "label_raw": x.get("label_raw"),
        "label_canonical": x.get("label_canonical"),
        "slot_label_raw": x.get("slot_label_raw"),
        "slot_label_canonical": x.get("slot_label_canonical"),
        "retrieved_label_raw": x.get("retrieved_label_raw"),
        "retrieved_label_canonical": x.get("retrieved_label_canonical"),
        "aabb_min": x.get("aabb_min"),
        "aabb_max": x.get("aabb_max"),
        "center": x.get("center"),
        "dimensions": x.get("dimensions"),
        "obb_center": x.get("obb_center"),
        "obb_dimensions": x.get("obb_dimensions"),
        "obb_rotation_matrix": x.get("obb_rotation_matrix"),
    }


def _build_match_rows(
    predictions: List[Dict],
    gts: List[Dict],
    matches: List[Dict],
    unmatched_pred: List[int],
    unmatched_gt: List[int],
    iou_mode: str,
) -> List[Dict]:
    rows = []
    matched_pred = {m["pred_idx"] for m in matches}
    matched_gt = {m["gt_idx"] for m in matches}
    for m in matches:
        p = predictions[m["pred_idx"]]
        g = gts[m["gt_idx"]]
        rows.append(
            {
                "status": "matched",
                "pred_idx": m["pred_idx"],
                "pred_id": p.get("id"),
                "pred_label_raw": p.get("label_raw"),
                "pred_label_canonical": p.get("label_canonical"),
                "gt_idx": m["gt_idx"],
                "gt_id": g.get("id"),
                "gt_label_raw": g.get("label_raw"),
                "gt_label_canonical": g.get("label_canonical"),
                "iou": m.get("iou"),
                "centroid_error_m": m.get("centroid_error_m"),
                "strict": m.get("strict"),
                "relaxed": m.get("relaxed"),
                "label_match": p.get("label_canonical") == g.get("label_canonical"),
            }
        )
    for pi in unmatched_pred:
        p = predictions[pi]
        best_iou, best_gi = 0.0, None
        for gi, g in enumerate(gts):
            iou = pair_iou_3d(p, g, iou_mode=iou_mode)
            if iou > best_iou:
                best_iou, best_gi = iou, gi
        bg = gts[best_gi] if best_gi is not None else {}
        rows.append(
            {
                "status": "unmatched_pred",
                "pred_idx": pi,
                "pred_id": p.get("id"),
                "pred_label_raw": p.get("label_raw"),
                "pred_label_canonical": p.get("label_canonical"),
                "gt_idx": best_gi,
                "gt_id": bg.get("id"),
                "gt_label_raw": bg.get("label_raw"),
                "gt_label_canonical": bg.get("label_canonical"),
                "iou": best_iou,
                "centroid_error_m": (
                    float(
                        np.linalg.norm(
                            np.asarray(p.get("center", [0, 0, 0]), dtype=np.float64)
                            - np.asarray(bg.get("center", [0, 0, 0]), dtype=np.float64)
                        )
                    )
                    if best_gi is not None
                    else None
                ),
                "strict": None,
                "relaxed": None,
                "label_match": (
                    p.get("label_canonical") == bg.get("label_canonical") if best_gi is not None else None
                ),
            }
        )
    for gi in unmatched_gt:
        if gi in matched_gt:
            continue
        g = gts[gi]
        rows.append(
            {
                "status": "unmatched_gt",
                "pred_idx": None,
                "pred_id": None,
                "pred_label_raw": None,
                "pred_label_canonical": None,
                "gt_idx": gi,
                "gt_id": g.get("id"),
                "gt_label_raw": g.get("label_raw"),
                "gt_label_canonical": g.get("label_canonical"),
                "iou": None,
                "centroid_error_m": None,
                "strict": None,
                "relaxed": None,
                "label_match": None,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Run rerun-safe evaluation from saved artifacts.")
    parser.add_argument("--prediction-json", default=None, help="Path to scene_graph.json or scene_graph_sam3d_only.json (legacy alias for --prediction-path)")
    parser.add_argument("--prediction-path", default=None, help="Path to prediction artifact (scene graph JSON or CLIO GraphML)")
    parser.add_argument("--prediction-type", default="scene_graph", choices=["scene_graph", "clio"], help="Prediction source type")
    parser.add_argument("--gt-json", required=True, help="Path to Supervisely cuboid_3d JSON")
    parser.add_argument("--eval-config", default=None, help="Path to eval_config.json")
    parser.add_argument("--run-dir", default=None, help="Optional run directory for timing/perf extraction")
    parser.add_argument("--scene", default="unknown_scene")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--method-tag", default=None, help="Optional method tag (e.g., pipeline_full, sam3d_only, clio)")
    parser.add_argument("--results-root", default=None, help="Results output root. Default: <run_dir>/results when run_dir is set.")
    args = parser.parse_args()

    eval_dir = str(Path(__file__).resolve().parent)
    eval_cfg_path = args.eval_config or str((Path(eval_dir) / "eval_config.json").resolve())
    cfg = _load_eval_config(eval_cfg_path, eval_dir)
    canonical_path = cfg["paths"]["canonical_labels"]
    aliases_path = cfg["paths"]["label_aliases"]
    learned_aliases_path = cfg["paths"]["learned_label_aliases"]
    label_cfg = cfg.get("label_resolution", {})
    use_embedding_for_unresolved = bool(label_cfg.get("use_embedding_for_unresolved", False))
    learn_new_aliases = bool(label_cfg.get("learn_new_aliases", False))
    embedding_min_score = float(label_cfg.get("embedding_min_score", 0.55))
    canonical, alias_map, contains_rules = load_label_configs(canonical_path, aliases_path)
    learned_aliases = load_learned_aliases(learned_aliases_path)
    canonical_raw = _read_json(canonical_path)
    aliases_raw = _read_json(aliases_path)
    matching_cfg = cfg.get("matching", {})
    frame_normalization = str(matching_cfg.get("frame_normalization", "none")).strip().lower()
    gt_global_yaw_rad = 0.0
    if frame_normalization == "gt_global_yaw":
        gt_global_yaw_rad = estimate_supervisely_global_yaw(args.gt_json)
    prediction_path = args.prediction_path or args.prediction_json
    if not prediction_path:
        raise SystemExit("[ERR] Provide --prediction-path (or --prediction-json).")
    prediction_type = str(args.prediction_type).strip().lower()

    if prediction_type == "scene_graph":
        preds = load_predictions_scene_graph(
            prediction_path,
            canonical=canonical,
            alias_map=alias_map,
            contains_rules=contains_rules,
            learned_alias_map=learned_aliases,
            use_embedding_for_unresolved=use_embedding_for_unresolved,
            learn_new_aliases=learn_new_aliases,
            learned_alias_path=learned_aliases_path,
            embedding_min_score=embedding_min_score,
            label_source=cfg.get("prediction", {}).get("label_source", "prefer_retrieved_else_label"),
            normalize_yaw_rad=gt_global_yaw_rad,
        )
    else:
        preds = load_predictions_clio_graphml(
            prediction_path,
            canonical=canonical,
            alias_map=alias_map,
            contains_rules=contains_rules,
            learned_alias_map=learned_aliases,
            use_embedding_for_unresolved=use_embedding_for_unresolved,
            learn_new_aliases=learn_new_aliases,
            learned_alias_path=learned_aliases_path,
            embedding_min_score=embedding_min_score,
            normalize_yaw_rad=gt_global_yaw_rad,
        )
    gts = load_supervisely_gt(
        args.gt_json,
        canonical=canonical,
        alias_map=alias_map,
        contains_rules=contains_rules,
        learned_alias_map=learned_aliases,
        use_embedding_for_unresolved=use_embedding_for_unresolved,
        learn_new_aliases=learn_new_aliases,
        learned_alias_path=learned_aliases_path,
        embedding_min_score=embedding_min_score,
        normalize_yaw_rad=gt_global_yaw_rad,
    )

    eval_hash = _hash_inputs(prediction_path, args.gt_json, cfg, canonical_raw, aliases_raw)
    run_id = args.run_id or (Path(args.run_dir).name if args.run_dir else Path(prediction_path).parent.name)
    run_dir_for_eval = Path(args.run_dir) if args.run_dir else Path(prediction_path).parent
    if args.results_root:
        results_root = Path(args.results_root)
    elif args.run_dir:
        results_root = run_dir_for_eval / "results"
    else:
        results_root = Path(eval_dir) / "results"
    by_run_dir = results_root / "by_run"
    by_run_dir.mkdir(parents=True, exist_ok=True)
    out_file = by_run_dir / f"{args.scene}_{run_id}_{eval_hash}.json"

    preds_filtered, filter_info = _filter_predictions_by_config(preds, cfg)
    metrics = evaluate(preds_filtered, gts, cfg, run_dir=run_dir_for_eval)
    perf = _extract_performance(run_dir_for_eval)
    metrics["performance"] = perf
    metrics["metadata"] = {
        "scene": args.scene,
        "run_id": run_id,
        "method_tag": args.method_tag,
        "prediction_path": str(Path(prediction_path).resolve()),
        "prediction_type": prediction_type,
        "gt_json": str(Path(args.gt_json).resolve()),
        "eval_config": str(Path(eval_cfg_path).resolve()),
        "canonical_labels": str(Path(canonical_path).resolve()),
        "label_aliases": str(Path(aliases_path).resolve()),
        "learned_label_aliases": str(Path(learned_aliases_path).resolve()),
        "label_resolution": {
            "use_embedding_for_unresolved": use_embedding_for_unresolved,
            "learn_new_aliases": learn_new_aliases,
            "embedding_min_score": embedding_min_score,
        },
        "frame_normalization": {
            "mode": frame_normalization,
            "gt_global_yaw_rad": gt_global_yaw_rad,
            "gt_global_yaw_deg": float(np.degrees(gt_global_yaw_rad)),
        },
        "eval_hash": eval_hash,
        "results_root": str(results_root.resolve()),
        "created_at": datetime.now().isoformat(),
    }
    metrics["prediction_filter"] = filter_info
    metrics["instances"] = {
        "predictions": [_instance_view(x) for x in preds_filtered],
        "gts": [_instance_view(x) for x in gts],
    }
    metrics["diagnostics"]["match_rows"] = _build_match_rows(
        predictions=preds_filtered,
        gts=gts,
        matches=metrics.get("matches", []),
        unmatched_pred=metrics.get("unmatched_prediction_indices", []),
        unmatched_gt=metrics.get("unmatched_gt_indices", []),
        iou_mode=metrics.get("matching", {}).get("iou_mode", "aabb"),
    )
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[OK] wrote by-run metrics: {out_file}")
    match_csv = by_run_dir / f"{args.scene}_{run_id}_{eval_hash}_match_details.csv"
    _write_csv_rows(match_csv, metrics["diagnostics"]["match_rows"])
    print(f"[OK] wrote match details: {match_csv}")

    main_row = {
        "scene": args.scene,
        "run_id": run_id,
        "method_tag": args.method_tag or prediction_type,
        "prediction_type": prediction_type,
        "eval_hash": eval_hash,
        "precision": metrics["detection_metrics"]["precision"],
        "recall": metrics["detection_metrics"]["recall"],
        "f1": metrics["detection_metrics"]["f1"],
        "mean_iou_3d": metrics["geometry_metrics"]["mean_iou_3d"],
        "strict_accuracy": metrics["benchmark_compat"]["strict_accuracy"],
        "relaxed_accuracy": metrics["benchmark_compat"]["relaxed_accuracy"],
        "strict_precision": metrics["benchmark_compat"]["strict_precision"],
        "relaxed_precision": metrics["benchmark_compat"]["relaxed_precision"],
        "f1_relaxed": metrics["benchmark_compat"]["f1_relaxed"],
        "num_objects_in_map": metrics["benchmark_compat"]["num_objects_in_map"],
        "e2e_latency_ms": metrics["performance"]["e2e_latency_ms"],
        "e2e_rate_hz": metrics["performance"]["e2e_rate_hz"],
        "sam3d_inference_ms": metrics["performance"]["sam3d_inference_ms"],
        "sam3d_inference_hz": metrics["performance"]["sam3d_inference_hz"],
        "cpu_peak_mb": metrics["performance"]["cpu_peak_mb"],
        "gpu_peak_mb": metrics["performance"]["gpu_peak_mb"],
        "iou_mode": metrics.get("matching", {}).get("iou_mode"),
        "registration_helped_rate": metrics.get("diagnostics", {}).get("registration_summary", {}).get("helped_rate"),
        "registration_hurt_rate": metrics.get("diagnostics", {}).get("registration_summary", {}).get("hurt_rate"),
        "retrieval_swapped_rate": metrics.get("diagnostics", {}).get("retrieval_summary", {}).get("swapped_rate"),
        "retrieval_swapped_tp_rate": metrics.get("diagnostics", {}).get("retrieval_summary", {}).get("swapped_tp_rate"),
        "predictions_ignored": metrics.get("prediction_filter", {}).get("num_predictions_ignored"),
        "frame_normalization_mode": frame_normalization,
        "frame_normalization_yaw_deg": float(np.degrees(gt_global_yaw_rad)),
    }
    _write_csv_row(results_root / "tables" / "benchmark_compat_latest.csv", main_row)
    print(f"[OK] appended table row: {results_root / 'tables' / 'benchmark_compat_latest.csv'}")


if __name__ == "__main__":
    main()
