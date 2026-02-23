import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


def _latest_csv(results_root: Path, pattern: str) -> Path:
    files = sorted((results_root / "tables").glob(pattern))
    if not files:
        raise SystemExit(f"[ERR] No files matching {pattern} under {results_root / 'tables'}")
    return files[-1]


def _load_pose_sensitivity(results_root: Path) -> pd.DataFrame:
    diag_dir = results_root / "diagnostics"
    rows: List[Dict] = []
    if not diag_dir.exists():
        return pd.DataFrame(rows)
    for p in diag_dir.glob("*_pose_sensitivity.json"):
        try:
            d = json.loads(p.read_text())
            s = d.get("summary", {})
            rows.append(
                {
                    "sensor_source": s.get("sensor_source", "unknown"),
                    "scene": s.get("scene"),
                    "run_id": s.get("run_id"),
                    "method_tag": s.get("method_tag"),
                    "pose_sensitivity_score": s.get("pose_sensitivity_score", 0.0),
                    "baseline_f1": s.get("baseline_f1", 0.0),
                    "std_f1": s.get("std_f1", 0.0),
                }
            )
        except Exception:
            continue
    return pd.DataFrame(rows)


def _write_sensor_summary(df: pd.DataFrame, pose_df: pd.DataFrame, out_csv: Path, out_json: Path):
    keep = ["lidar", "realsense"]
    d = df[df["sensor_source"].isin(keep)].copy()
    if d.empty:
        raise SystemExit("[ERR] No lidar/realsense rows found in ablation table.")
    cols = [
        "f1",
        "recall",
        "precision",
        "mean_iou_3d",
        "near_f1",
        "mid_f1",
        "far_f1",
        "registration_helped_rate",
        "registration_hurt_rate",
        "near_recall",
        "mid_recall",
        "far_recall",
    ]
    for c in cols:
        if c not in d.columns:
            d[c] = pd.NA
    agg = d.groupby("sensor_source", as_index=False)[cols].mean(numeric_only=True)
    if not pose_df.empty:
        p = pose_df[pose_df["sensor_source"].isin(keep)].groupby("sensor_source", as_index=False)[
            ["pose_sensitivity_score", "baseline_f1", "std_f1"]
        ].mean(numeric_only=True)
        agg = agg.merge(p, on="sensor_source", how="left")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_csv, index=False)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"rows": agg.to_dict(orient="records")}, indent=2))


def _plot_sensor_decision(summary_csv: Path, out_png_main: Path, out_png_range: Path):
    d = pd.read_csv(summary_csv)
    if d.empty:
        return
    sensors = d["sensor_source"].tolist()
    xs = list(range(len(sensors)))

    plt.figure(figsize=(8, 5))
    width = 0.25
    for i, m in enumerate(["f1", "recall", "mean_iou_3d"]):
        vals = d[m].fillna(0.0).tolist()
        plt.bar([x + (i - 1) * width for x in xs], vals, width=width, label=m)
    plt.xticks(xs, sensors)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Mean score")
    plt.title("Sensor Decision: Main Geometry Metrics")
    plt.legend()
    plt.tight_layout()
    out_png_main.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png_main, dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    width = 0.25
    for i, m in enumerate(["near_f1", "mid_f1", "far_f1"]):
        vals = d[m].fillna(0.0).tolist()
        plt.bar([x + (i - 1) * width for x in xs], vals, width=width, label=m)
    plt.xticks(xs, sensors)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Mean F1")
    plt.title("Sensor Decision: Range-Bucket F1")
    plt.legend()
    plt.tight_layout()
    out_png_range.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png_range, dpi=220)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Build lidar-vs-realsense decision report from results artifacts.")
    default_results = str((Path(__file__).resolve().parent / "results").resolve())
    parser.add_argument("--results-root", default=default_results)
    parser.add_argument("--main-csv", default=None, help="Optional explicit ablation_main CSV.")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    main_csv = Path(args.main_csv) if args.main_csv else _latest_csv(results_root, "ablation_main_*.csv")
    df = pd.read_csv(main_csv)
    pose_df = _load_pose_sensitivity(results_root)

    comp_dir = results_root / "comparisons"
    out_csv = comp_dir / "sensor_decision_summary.csv"
    out_json = comp_dir / "sensor_decision_summary.json"
    out_png_main = comp_dir / "sensor_decision_main_metrics.png"
    out_png_range = comp_dir / "sensor_decision_range_f1.png"

    _write_sensor_summary(df, pose_df, out_csv, out_json)
    _plot_sensor_decision(out_csv, out_png_main, out_png_range)

    print(f"[OK] wrote {out_csv}")
    print(f"[OK] wrote {out_json}")
    print(f"[OK] wrote {out_png_main}")
    print(f"[OK] wrote {out_png_range}")


if __name__ == "__main__":
    main()
