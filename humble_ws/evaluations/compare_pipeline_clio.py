import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Optional, Tuple


def _load_json(path: Path) -> Dict:
    with open(path) as f:
        return json.load(f)


def _pick_by_type(results_root: Path, scene: str, run_id: str, prediction_type: str) -> Optional[Path]:
    by_run = results_root / "by_run"
    if not by_run.exists():
        return None
    best = None
    best_ts = ""
    for p in by_run.glob("*.json"):
        try:
            d = _load_json(p)
            m = d.get("metadata", {})
            if m.get("scene") != scene or m.get("run_id") != run_id:
                continue
            if m.get("prediction_type", "scene_graph") != prediction_type:
                continue
            ts = str(m.get("created_at", ""))
            if ts >= best_ts:
                best = p
                best_ts = ts
        except Exception:
            continue
    return best


def _extract_summary(d: Dict) -> Dict:
    counts = d.get("counts", {})
    det = d.get("detection_metrics", {})
    geo = d.get("geometry_metrics", {})
    bench = d.get("benchmark_compat", {})
    fp = d.get("fp_breakdown", {})
    return {
        "tp": counts.get("tp"),
        "fp": counts.get("fp"),
        "fn": counts.get("fn"),
        "precision": det.get("precision"),
        "recall": det.get("recall"),
        "f1": det.get("f1"),
        "mean_iou_3d": geo.get("mean_iou_3d"),
        "strict_accuracy": bench.get("strict_accuracy"),
        "relaxed_accuracy": bench.get("relaxed_accuracy"),
        "strict_precision": bench.get("strict_precision"),
        "relaxed_precision": bench.get("relaxed_precision"),
        "f1_relaxed": bench.get("f1_relaxed"),
        "num_objects_in_map": bench.get("num_objects_in_map"),
        "fp_wrong_label": fp.get("wrong_label"),
        "fp_low_iou": fp.get("low_iou"),
        "fp_no_overlap": fp.get("no_overlap"),
        "fp_unresolved_label": fp.get("unresolved_label"),
    }


def _safe_delta(a, b):
    try:
        return float(a) - float(b)
    except Exception:
        return None


def _write_csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _build_summary_rows(pipeline: Dict, clio: Dict):
    p = _extract_summary(pipeline)
    c = _extract_summary(clio)
    rows = []
    for k in p.keys():
        rows.append(
            {
                "metric": k,
                "pipeline": p.get(k),
                "clio": c.get(k),
                "delta_pipeline_minus_clio": _safe_delta(p.get(k), c.get(k)),
            }
        )
    return rows


def _build_per_class_rows(pipeline: Dict, clio: Dict):
    pp = pipeline.get("per_class", {})
    cp = clio.get("per_class", {})
    labels = sorted(set(pp.keys()) | set(cp.keys()))
    out = []
    for cls in labels:
        a = pp.get(cls, {})
        b = cp.get(cls, {})
        out.append(
            {
                "class": cls,
                "pipeline_tp": a.get("tp", 0),
                "clio_tp": b.get("tp", 0),
                "delta_tp": _safe_delta(a.get("tp", 0), b.get("tp", 0)),
                "pipeline_fp": a.get("fp", 0),
                "clio_fp": b.get("fp", 0),
                "delta_fp": _safe_delta(a.get("fp", 0), b.get("fp", 0)),
                "pipeline_fn": a.get("fn", 0),
                "clio_fn": b.get("fn", 0),
                "delta_fn": _safe_delta(a.get("fn", 0), b.get("fn", 0)),
                "pipeline_precision": a.get("precision"),
                "clio_precision": b.get("precision"),
                "delta_precision": _safe_delta(a.get("precision"), b.get("precision")),
                "pipeline_recall": a.get("recall"),
                "clio_recall": b.get("recall"),
                "delta_recall": _safe_delta(a.get("recall"), b.get("recall")),
                "pipeline_f1": a.get("f1"),
                "clio_f1": b.get("f1"),
                "delta_f1": _safe_delta(a.get("f1"), b.get("f1")),
            }
        )
    return out


def main():
    parser = argparse.ArgumentParser(description="Build side-by-side pipeline vs CLIO comparison from by_run JSONs.")
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--pipeline-json", default=None, help="Optional explicit pipeline by_run json path")
    parser.add_argument("--clio-json", default=None, help="Optional explicit CLIO by_run json path")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    pipeline_json = Path(args.pipeline_json) if args.pipeline_json else _pick_by_type(
        results_root, args.scene, args.run_id, "scene_graph"
    )
    clio_json = Path(args.clio_json) if args.clio_json else _pick_by_type(
        results_root, args.scene, args.run_id, "clio"
    )
    if pipeline_json is None or not pipeline_json.exists():
        raise SystemExit("[ERR] pipeline by_run json not found (use --pipeline-json or ensure scene/run_id/prediction_type match).")
    if clio_json is None or not clio_json.exists():
        raise SystemExit("[ERR] clio by_run json not found (use --clio-json or ensure scene/run_id/prediction_type match).")

    p = _load_json(pipeline_json)
    c = _load_json(clio_json)
    comp_dir = results_root / "comparisons"
    stem = f"{args.scene}_{args.run_id}_pipeline_vs_clio"
    summary_csv = comp_dir / f"{stem}_summary.csv"
    per_class_csv = comp_dir / f"{stem}_per_class.csv"
    meta_json = comp_dir / f"{stem}_meta.json"

    _write_csv(summary_csv, _build_summary_rows(p, c))
    _write_csv(per_class_csv, _build_per_class_rows(p, c))
    payload = {
        "scene": args.scene,
        "run_id": args.run_id,
        "pipeline_json": str(pipeline_json.resolve()),
        "clio_json": str(clio_json.resolve()),
    }
    comp_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[OK] wrote {summary_csv}")
    print(f"[OK] wrote {per_class_csv}")
    print(f"[OK] wrote {meta_json}")


if __name__ == "__main__":
    main()
