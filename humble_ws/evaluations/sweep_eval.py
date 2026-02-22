import argparse
import copy
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from eval_common import estimate_supervisely_global_yaw, load_predictions_scene_graph, load_supervisely_gt
from label_matching import load_label_configs, load_learned_aliases
from run_eval import _filter_predictions_by_config, _load_eval_config, _read_json, evaluate


def _parse_thresholds(raw: str) -> List[float]:
    vals = []
    for x in (raw or "").split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(float(x))
    if not vals:
        vals = [0.05, 0.10, 0.15, 0.20, 0.25]
    return sorted(set(vals))


def _parse_bools(raw: str) -> List[bool]:
    out = []
    for x in (raw or "").split(","):
        s = x.strip().lower()
        if s in ("true", "1", "yes", "y"):
            out.append(True)
        elif s in ("false", "0", "no", "n"):
            out.append(False)
    if not out:
        out = [False, True]
    return sorted(set(out))


def _write_csv(path: Path, rows: List[Dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Sweep eval across IoU thresholds and require_label_match options.")
    parser.add_argument("--prediction-json", required=True)
    parser.add_argument("--gt-json", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--scene", default="unknown_scene")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--eval-config", default=None)
    parser.add_argument("--results-root", default=None, help="Default: <run_dir>/results if run_dir provided")
    parser.add_argument("--thresholds", default="0.05,0.10,0.15,0.20,0.25")
    parser.add_argument("--label-match-options", default="false,true")
    args = parser.parse_args()

    eval_dir = str(Path(__file__).resolve().parent)
    eval_cfg_path = args.eval_config or str((Path(eval_dir) / "eval_config.json").resolve())
    base_cfg = _load_eval_config(eval_cfg_path, eval_dir)
    thresholds = _parse_thresholds(args.thresholds)
    label_flags = _parse_bools(args.label_match_options)

    canonical_path = base_cfg["paths"]["canonical_labels"]
    aliases_path = base_cfg["paths"]["label_aliases"]
    learned_aliases_path = base_cfg["paths"]["learned_label_aliases"]
    canonical, alias_map, contains_rules = load_label_configs(canonical_path, aliases_path)
    learned_aliases = load_learned_aliases(learned_aliases_path)
    label_cfg = base_cfg.get("label_resolution", {})
    use_embedding_for_unresolved = bool(label_cfg.get("use_embedding_for_unresolved", False))
    learn_new_aliases = bool(label_cfg.get("learn_new_aliases", False))
    embedding_min_score = float(label_cfg.get("embedding_min_score", 0.55))

    frame_norm = str(base_cfg.get("matching", {}).get("frame_normalization", "none")).strip().lower()
    gt_global_yaw_rad = estimate_supervisely_global_yaw(args.gt_json) if frame_norm == "gt_global_yaw" else 0.0

    preds_all = load_predictions_scene_graph(
        args.prediction_json,
        canonical=canonical,
        alias_map=alias_map,
        contains_rules=contains_rules,
        learned_alias_map=learned_aliases,
        use_embedding_for_unresolved=use_embedding_for_unresolved,
        learn_new_aliases=learn_new_aliases,
        learned_alias_path=learned_aliases_path,
        embedding_min_score=embedding_min_score,
        label_source=base_cfg.get("prediction", {}).get("label_source", "prefer_retrieved_else_label"),
        normalize_yaw_rad=gt_global_yaw_rad,
    )
    preds, pred_filter = _filter_predictions_by_config(preds_all, base_cfg)
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

    run_dir_for_eval = Path(args.run_dir) if args.run_dir else Path(args.prediction_json).parent
    run_id = args.run_id or (Path(args.run_dir).name if args.run_dir else Path(args.prediction_json).parent.name)
    if args.results_root:
        results_root = Path(args.results_root)
    elif args.run_dir:
        results_root = run_dir_for_eval / "results"
    else:
        results_root = Path(eval_dir) / "results"

    sweep_dir = results_root / "sweeps"
    by_setting_dir = sweep_dir / "by_setting"
    by_setting_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for thr in thresholds:
        for req in label_flags:
            cfg = copy.deepcopy(base_cfg)
            cfg.setdefault("matching", {})
            cfg["matching"]["iou_threshold"] = float(thr)
            cfg["matching"]["require_label_match"] = bool(req)
            m = evaluate(preds, gts, cfg, run_dir=run_dir_for_eval)
            rec = {
                "scene": args.scene,
                "run_id": run_id,
                "iou_mode": m.get("matching", {}).get("iou_mode"),
                "frame_normalization_mode": frame_norm,
                "frame_normalization_yaw_deg": float(gt_global_yaw_rad * 180.0 / 3.141592653589793),
                "iou_threshold": float(thr),
                "require_label_match": bool(req),
                "tp": m["counts"]["tp"],
                "fp": m["counts"]["fp"],
                "fn": m["counts"]["fn"],
                "precision": m["detection_metrics"]["precision"],
                "recall": m["detection_metrics"]["recall"],
                "f1": m["detection_metrics"]["f1"],
                "mean_iou_3d": m["geometry_metrics"]["mean_iou_3d"],
                "predictions_ignored": pred_filter.get("num_predictions_ignored", 0),
            }
            rows.append(rec)
            setting_json = by_setting_dir / (
                f"{args.scene}_{run_id}_thr{thr:.2f}_label{str(req).lower()}_{stamp}.json"
            )
            payload = {
                "metrics": m,
                "setting": rec,
                "prediction_filter": pred_filter,
                "metadata": {
                    "prediction_json": str(Path(args.prediction_json).resolve()),
                    "gt_json": str(Path(args.gt_json).resolve()),
                    "eval_config": str(Path(eval_cfg_path).resolve()),
                    "created_at": datetime.now().isoformat(),
                },
            }
            with open(setting_json, "w") as f:
                json.dump(payload, f, indent=2)

    # Build threshold-wise geometry ceiling + label penalty summary.
    summary_rows = []
    for thr in thresholds:
        a = next((r for r in rows if r["iou_threshold"] == thr and r["require_label_match"] is False), None)
        b = next((r for r in rows if r["iou_threshold"] == thr and r["require_label_match"] is True), None)
        if not a or not b:
            continue
        summary_rows.append(
            {
                "scene": args.scene,
                "run_id": run_id,
                "iou_threshold": thr,
                "geometry_ceiling_recall_no_label": a["recall"],
                "geometry_ceiling_f1_no_label": a["f1"],
                "recall_with_label": b["recall"],
                "f1_with_label": b["f1"],
                "label_penalty_recall_delta": a["recall"] - b["recall"],
                "label_penalty_f1_delta": a["f1"] - b["f1"],
                "predictions_ignored": a["predictions_ignored"],
            }
        )

    rows_csv = sweep_dir / f"sweep_raw_{args.scene}_{run_id}_{stamp}.csv"
    summary_csv = sweep_dir / f"sweep_summary_{args.scene}_{run_id}_{stamp}.csv"
    _write_csv(rows_csv, rows)
    _write_csv(summary_csv, summary_rows)
    print(f"[OK] wrote {rows_csv}")
    print(f"[OK] wrote {summary_csv}")


if __name__ == "__main__":
    main()
