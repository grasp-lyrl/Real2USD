import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


def _load_by_run_files(results_root: Path):
    by_run = results_root / "by_run"
    if not by_run.exists():
        return []
    rows = []
    for p in sorted(by_run.glob("*.json")):
        try:
            with open(p) as f:
                m = json.load(f)
            rows.append(
                {
                    "scene": m["metadata"].get("scene"),
                    "run_id": m["metadata"].get("run_id"),
                    "prediction_type": m["metadata"].get("prediction_type", "scene_graph"),
                    "eval_hash": m["metadata"].get("eval_hash"),
                    "precision": m["detection_metrics"].get("precision"),
                    "recall": m["detection_metrics"].get("recall"),
                    "f1": m["detection_metrics"].get("f1"),
                    "mean_iou_3d": m["geometry_metrics"].get("mean_iou_3d"),
                    "strict_accuracy": m["benchmark_compat"].get("strict_accuracy"),
                    "relaxed_accuracy": m["benchmark_compat"].get("relaxed_accuracy"),
                    "strict_precision": m["benchmark_compat"].get("strict_precision"),
                    "relaxed_precision": m["benchmark_compat"].get("relaxed_precision"),
                    "f1_relaxed": m["benchmark_compat"].get("f1_relaxed"),
                    "num_objects_in_map": m["benchmark_compat"].get("num_objects_in_map"),
                    "e2e_latency_ms": m.get("performance", {}).get("e2e_latency_ms"),
                    "e2e_rate_hz": m.get("performance", {}).get("e2e_rate_hz"),
                    "sam3d_inference_ms": m.get("performance", {}).get("sam3d_inference_ms"),
                    "sam3d_inference_hz": m.get("performance", {}).get("sam3d_inference_hz"),
                    "cpu_peak_mb": m.get("performance", {}).get("cpu_peak_mb"),
                    "gpu_peak_mb": m.get("performance", {}).get("gpu_peak_mb"),
                    "iou_mode": m.get("matching", {}).get("iou_mode"),
                    "registration_helped_rate": m.get("diagnostics", {}).get("registration_summary", {}).get("helped_rate"),
                    "registration_hurt_rate": m.get("diagnostics", {}).get("registration_summary", {}).get("hurt_rate"),
                    "registration_delta_centroid_m": m.get("diagnostics", {}).get("registration_summary", {}).get("mean_centroid_error_delta_m"),
                    "retrieval_swapped_rate": m.get("diagnostics", {}).get("retrieval_summary", {}).get("swapped_rate"),
                    "retrieval_swapped_tp_rate": m.get("diagnostics", {}).get("retrieval_summary", {}).get("swapped_tp_rate"),
                    "predictions_ignored": m.get("prediction_filter", {}).get("num_predictions_ignored"),
                    "frame_normalization_mode": m.get("metadata", {}).get("frame_normalization", {}).get("mode"),
                    "frame_normalization_yaw_deg": m.get("metadata", {}).get("frame_normalization", {}).get("gt_global_yaw_deg"),
                    "per_class_json": json.dumps(m.get("per_class", {}), sort_keys=True),
                    "by_run_json": str(p),
                }
            )
        except Exception:
            continue
    return rows


def _explode_per_class(rows):
    out = []
    for r in rows:
        raw = r.get("per_class_json", "{}")
        try:
            pc = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except json.JSONDecodeError:
            pc = {}
        for cls, m in pc.items():
            out.append(
                {
                    "scene": r.get("scene"),
                    "run_id": r.get("run_id"),
                    "eval_hash": r.get("eval_hash"),
                    "class": cls,
                    "tp": m.get("tp"),
                    "fp": m.get("fp"),
                    "fn": m.get("fn"),
                    "precision": m.get("precision"),
                    "recall": m.get("recall"),
                    "f1": m.get("f1"),
                    "mean_iou_3d": m.get("mean_iou_3d"),
                    "by_run_json": r.get("by_run_json"),
                }
            )
    return out


def _write_csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(k, "")) for k in keys) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _write_latex(path: Path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    cols = "l" * len(keys)
    lines = [
        "\\begin{tabular}{" + cols + "}",
        "\\hline",
        " & ".join(keys) + " \\\\",
        "\\hline",
    ]
    for r in rows:
        lines.append(" & ".join(str(r.get(k, "")) for k in keys) + " \\\\")
    lines += ["\\hline", "\\end{tabular}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Collect by-run JSON metrics into paper-friendly tables.")
    default_results = str((Path(__file__).resolve().parent / "results").resolve())
    parser.add_argument("--results-root", default=default_results)
    args = parser.parse_args()

    results_root = Path(args.results_root)
    rows = _load_by_run_files(results_root)
    if not rows:
        print("[WARN] no by_run metrics found")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tables = results_root / "tables"
    csv_path = tables / f"ablation_main_{stamp}.csv"
    md_path = tables / f"ablation_main_{stamp}.md"
    tex_path = tables / f"ablation_main_{stamp}.tex"
    perf_csv = tables / f"benchmark_compat_{stamp}.csv"
    per_class_csv = tables / f"per_class_{stamp}.csv"

    _write_csv(csv_path, rows)
    _write_csv(perf_csv, rows)
    _write_markdown(md_path, rows)
    _write_latex(tex_path, rows)
    _write_csv(per_class_csv, _explode_per_class(rows))

    print(f"[OK] wrote {csv_path}")
    print(f"[OK] wrote {md_path}")
    print(f"[OK] wrote {tex_path}")
    print(f"[OK] wrote {perf_csv}")
    print(f"[OK] wrote {per_class_csv}")


if __name__ == "__main__":
    main()
