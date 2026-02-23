"""
Compare pipeline_full, sam3d_only, and CLIO from by_run JSONs.

Outputs: summary CSV (incl. strict/relaxed accuracy, mean_iou_3d, mean_centroid_error_m,
and macro-averaged per-class localization), localization plots, main metrics plot, FP breakdown, per-class CSV/plot.

Localization metrics:
- Over matched pairs: strict_accuracy, relaxed_accuracy, mean_iou_3d, mean_centroid_error_m, f1_relaxed.
- Macro over classes: mean_strict_accuracy_per_class, mean_relaxed_accuracy_per_class, mean_iou_3d_per_class,
  mean_centroid_error_m_per_class (each class weighted once). See EVAL_METRICS.md.

Per-class (in per_class CSV and per_class_f1 plot): f1, precision, recall, mean_iou_3d,
strict_accuracy, relaxed_accuracy, mean_centroid_error_m per class.
"""
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


def _load_json(path: Path) -> Dict:
    with open(path) as f:
        return json.load(f)


def _pick_by_method(results_root: Path, scene: str, run_id: str, method_tag: str) -> Optional[Path]:
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
            if (m.get("method_tag") or "") != method_tag:
                continue
            ts = str(m.get("created_at", ""))
            if ts >= best_ts:
                best = p
                best_ts = ts
        except Exception:
            continue
    return best


def _extract_row(method: str, d: Dict) -> Dict:
    counts = d.get("counts", {})
    det = d.get("detection_metrics", {})
    geo = d.get("geometry_metrics", {})
    fp = d.get("fp_breakdown", {})
    bench = d.get("benchmark_compat", {})
    return {
        "method": method,
        "sensor_source": d.get("metadata", {}).get("sensor_source", "unknown"),
        "tp": counts.get("tp", 0),
        "fp": counts.get("fp", 0),
        "fn": counts.get("fn", 0),
        "precision": det.get("precision", 0.0),
        "recall": det.get("recall", 0.0),
        "f1": det.get("f1", 0.0),
        "mean_iou_3d": geo.get("mean_iou_3d", 0.0),
        "mean_centroid_error_m": geo.get("mean_centroid_error_m", 0.0),
        "strict_accuracy": bench.get("strict_accuracy", 0.0),
        "relaxed_accuracy": bench.get("relaxed_accuracy", 0.0),
        "strict_precision": bench.get("strict_precision", 0.0),
        "relaxed_precision": bench.get("relaxed_precision", 0.0),
        "f1_relaxed": bench.get("f1_relaxed", 0.0),
        "mean_strict_accuracy_per_class": bench.get("mean_strict_accuracy_per_class", 0.0),
        "mean_relaxed_accuracy_per_class": bench.get("mean_relaxed_accuracy_per_class", 0.0),
        "mean_iou_3d_per_class": bench.get("mean_iou_3d_per_class", 0.0),
        "mean_centroid_error_m_per_class": bench.get("mean_centroid_error_m_per_class", 0.0),
        "osR_strict": bench.get("osR_strict", 0.0),
        "osR_relaxed": bench.get("osR_relaxed", 0.0),
        "osP_strict": bench.get("osP_strict", 0.0),
        "osP_relaxed": bench.get("osP_relaxed", 0.0),
        "F1_os": bench.get("F1_os", 0.0),
        "fp_wrong_label": fp.get("wrong_label", 0),
        "fp_low_iou": fp.get("low_iou", 0),
        "fp_no_overlap": fp.get("no_overlap", 0),
        "fp_unresolved_label": fp.get("unresolved_label", 0),
        "predictions_ignored": d.get("prediction_filter", {}).get("num_predictions_ignored", 0),
        "near_f1": d.get("range_metrics", {}).get("buckets", {}).get("near", {}).get("f1", 0.0),
        "mid_f1": d.get("range_metrics", {}).get("buckets", {}).get("mid", {}).get("f1", 0.0),
        "far_f1": d.get("range_metrics", {}).get("buckets", {}).get("far", {}).get("f1", 0.0),
    }


def _write_csv(path: Path, rows: List[Dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _plot_main_metrics(rows: List[Dict], out_png: Path):
    methods = [r["method"] for r in rows]
    f1 = [float(r["f1"]) for r in rows]
    rec = [float(r["recall"]) for r in rows]
    prec = [float(r["precision"]) for r in rows]
    x = list(range(len(methods)))
    w = 0.25
    plt.figure(figsize=(9, 5))
    ax = plt.gca()
    ax.bar([i - w for i in x], f1, width=w, label="f1")
    ax.bar(x, rec, width=w, label="recall")
    ax.bar([i + w for i in x], prec, width=w, label="precision")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Pipeline vs SAM3D-only vs CLIO")
    ax.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=220)
    plt.close()


def _plot_localization_metrics(rows: List[Dict], out_png: Path):
    """Strict/relaxed accuracy and IoU/centroid error (localization) across methods. All are general (over matched pairs)."""
    methods = [r["method"] for r in rows]
    strict_acc = [float(r.get("strict_accuracy", 0.0)) for r in rows]
    relaxed_acc = [float(r.get("relaxed_accuracy", 0.0)) for r in rows]
    mean_iou = [float(r.get("mean_iou_3d", 0.0)) for r in rows]
    f1_relaxed = [float(r.get("f1_relaxed", 0.0)) for r in rows]
    x = list(range(len(methods)))
    w = 0.2
    plt.figure(figsize=(9, 5))
    ax = plt.gca()
    ax.bar([i - 1.5 * w for i in x], strict_acc, width=w, label="strict_acc")
    ax.bar([i - 0.5 * w for i in x], relaxed_acc, width=w, label="relaxed_acc")
    ax.bar([i + 0.5 * w for i in x], mean_iou, width=w, label="mean_iou_3d")
    ax.bar([i + 1.5 * w for i in x], f1_relaxed, width=w, label="f1_relaxed")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Localization: strict/relaxed accuracy, IoU, f1_relaxed (over matched pairs)")
    ax.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=220)
    plt.close()


def _plot_localization_per_class(rows: List[Dict], out_png: Path):
    """Macro-average localization over classes (each class weighted equally)."""
    methods = [r["method"] for r in rows]
    strict_pc = [float(r.get("mean_strict_accuracy_per_class", 0.0)) for r in rows]
    relaxed_pc = [float(r.get("mean_relaxed_accuracy_per_class", 0.0)) for r in rows]
    mean_iou_pc = [float(r.get("mean_iou_3d_per_class", 0.0)) for r in rows]
    x = list(range(len(methods)))
    w = 0.25
    plt.figure(figsize=(9, 5))
    ax = plt.gca()
    ax.bar([i - w for i in x], strict_pc, width=w, label="strict_acc (avg/class)")
    ax.bar([i for i in x], relaxed_pc, width=w, label="relaxed_acc (avg/class)")
    ax.bar([i + w for i in x], mean_iou_pc, width=w, label="mean_iou_3d (avg/class)")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Localization: macro-average over classes (each class counts once)")
    ax.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=220)
    plt.close()


def _plot_fp_breakdown(rows: List[Dict], out_png: Path):
    methods = [r["method"] for r in rows]
    wrong = [int(r["fp_wrong_label"]) for r in rows]
    low_iou = [int(r["fp_low_iou"]) for r in rows]
    no_ov = [int(r["fp_no_overlap"]) for r in rows]
    unr = [int(r["fp_unresolved_label"]) for r in rows]
    x = list(range(len(methods)))
    plt.figure(figsize=(9, 5))
    ax = plt.gca()
    ax.bar(x, wrong, label="wrong_label")
    ax.bar(x, low_iou, bottom=wrong, label="low_iou")
    b2 = [wrong[i] + low_iou[i] for i in range(len(x))]
    ax.bar(x, no_ov, bottom=b2, label="no_overlap")
    b3 = [b2[i] + no_ov[i] for i in range(len(x))]
    ax.bar(x, unr, bottom=b3, label="unresolved_label")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylabel("FP count")
    ax.set_title("FP Reason Breakdown by Method")
    ax.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=220)
    plt.close()


def _plot_per_class(rows_per_method: Dict[str, Dict], out_png: Path):
    labels = sorted(set(k for d in rows_per_method.values() for k in d.keys()))
    methods = list(rows_per_method.keys())
    x = list(range(len(labels)))
    w = 0.8 / max(1, len(methods))
    plt.figure(figsize=(11, 5))
    ax = plt.gca()
    for mi, method in enumerate(methods):
        vals = [float(rows_per_method[method].get(lbl, {}).get("f1", 0.0)) for lbl in labels]
        offs = [i - 0.4 + (mi + 0.5) * w for i in x]
        ax.bar(offs, vals, width=w, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("F1")
    ax.set_title("Per-class F1 by Method")
    ax.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=220)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare 3 methods side-by-side from by_run JSON outputs.")
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--method-tags", default="pipeline_full,sam3d_only,clio")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    methods = [x.strip() for x in args.method_tags.split(",") if x.strip()]
    by_method = {}
    for m in methods:
        p = _pick_by_method(results_root, args.scene, args.run_id, m)
        if p is None:
            raise SystemExit(f"[ERR] Missing by_run JSON for method_tag={m}, scene={args.scene}, run_id={args.run_id}")
        by_method[m] = _load_json(p)

    comp_dir = results_root / "comparisons"
    stem = f"{args.scene}_{args.run_id}_three_methods"
    summary_csv = comp_dir / f"{stem}_summary.csv"
    per_class_csv = comp_dir / f"{stem}_per_class.csv"
    meta_json = comp_dir / f"{stem}_meta.json"
    plot_main = comp_dir / f"{stem}_main_metrics.png"
    plot_localization = comp_dir / f"{stem}_localization_metrics.png"
    plot_localization_per_class = comp_dir / f"{stem}_localization_per_class.png"
    plot_fp = comp_dir / f"{stem}_fp_breakdown.png"
    plot_class = comp_dir / f"{stem}_per_class_f1.png"

    summary_rows = [_extract_row(method, d) for method, d in by_method.items()]
    _write_csv(summary_csv, summary_rows)

    per_class_rows = []
    rows_per_method = {}
    for method, d in by_method.items():
        pc = d.get("per_class", {})
        rows_per_method[method] = pc
        for cls, vals in pc.items():
            per_class_rows.append(
                {
                    "method": method,
                    "class": cls,
                    "tp": vals.get("tp", 0),
                    "fp": vals.get("fp", 0),
                    "fn": vals.get("fn", 0),
                    "precision": vals.get("precision", 0.0),
                    "recall": vals.get("recall", 0.0),
                    "f1": vals.get("f1", 0.0),
                    "mean_iou_3d": vals.get("mean_iou_3d", 0.0),
                    "strict_accuracy": vals.get("strict_accuracy", 0.0),
                    "relaxed_accuracy": vals.get("relaxed_accuracy", 0.0),
                    "mean_centroid_error_m": vals.get("mean_centroid_error_m", 0.0),
                }
            )
    _write_csv(per_class_csv, per_class_rows)

    _plot_main_metrics(summary_rows, plot_main)
    _plot_localization_metrics(summary_rows, plot_localization)
    _plot_localization_per_class(summary_rows, plot_localization_per_class)
    _plot_fp_breakdown(summary_rows, plot_fp)
    _plot_per_class(rows_per_method, plot_class)

    comp_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_json, "w") as f:
        json.dump(
            {
                "scene": args.scene,
                "run_id": args.run_id,
                "method_tags": methods,
                "by_run_sources": {k: v.get("metadata", {}).get("eval_hash") for k, v in by_method.items()},
            },
            f,
            indent=2,
        )

    print(f"[OK] wrote {summary_csv}")
    print(f"[OK] wrote {per_class_csv}")
    print(f"[OK] wrote {plot_main}")
    print(f"[OK] wrote {plot_localization}")
    print(f"[OK] wrote {plot_localization_per_class}")
    print(f"[OK] wrote {plot_fp}")
    print(f"[OK] wrote {plot_class}")
    print(f"[OK] wrote {meta_json}")


if __name__ == "__main__":
    main()
