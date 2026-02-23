import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from metrics_3d import greedy_match


def _load_json(path: Path) -> Dict:
    with open(path) as f:
        return json.load(f)


def _pick_by_method(results_root: Path, scene: str, run_id: str, method_tag: Optional[str]) -> Optional[Path]:
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
            if method_tag and (m.get("method_tag") or "") != method_tag:
                continue
            ts = str(m.get("created_at", ""))
            if ts >= best_ts:
                best = p
                best_ts = ts
        except Exception:
            continue
    return best


def _yaw_matrix_deg(yaw_deg: float) -> np.ndarray:
    y = np.radians(yaw_deg)
    c = float(np.cos(y))
    s = float(np.sin(y))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _transform_xyz(v: List[float], Rz: np.ndarray, t_xyz: np.ndarray) -> List[float]:
    a = np.asarray(v, dtype=np.float64).reshape(3)
    out = Rz @ a + t_xyz
    return out.tolist()


def _transform_prediction(p: Dict, yaw_deg: float, dx: float, dy: float, dz: float) -> Dict:
    q = dict(p)
    Rz = _yaw_matrix_deg(yaw_deg)
    t_xyz = np.array([dx, dy, dz], dtype=np.float64)
    if q.get("center") is not None:
        q["center"] = _transform_xyz(q["center"], Rz, t_xyz)
    if q.get("aabb_min") is not None and q.get("aabb_max") is not None:
        mn = np.asarray(q["aabb_min"], dtype=np.float64)
        mx = np.asarray(q["aabb_max"], dtype=np.float64)
        corners = np.array(
            [
                [mn[0], mn[1], mn[2]],
                [mn[0], mn[1], mx[2]],
                [mn[0], mx[1], mn[2]],
                [mn[0], mx[1], mx[2]],
                [mx[0], mn[1], mn[2]],
                [mx[0], mn[1], mx[2]],
                [mx[0], mx[1], mn[2]],
                [mx[0], mx[1], mx[2]],
            ],
            dtype=np.float64,
        )
        tc = (Rz @ corners.T).T + t_xyz
        q["aabb_min"] = tc.min(axis=0).tolist()
        q["aabb_max"] = tc.max(axis=0).tolist()
    if q.get("obb_center") is not None:
        q["obb_center"] = _transform_xyz(q["obb_center"], Rz, t_xyz)
    if q.get("obb_rotation_matrix") is not None:
        try:
            R0 = np.asarray(q["obb_rotation_matrix"], dtype=np.float64).reshape(3, 3)
            q["obb_rotation_matrix"] = (Rz @ R0).tolist()
        except Exception:
            pass
    return q


def _compute_metrics(preds: List[Dict], gts: List[Dict], iou_threshold: float, require_label_match: bool, iou_mode: str) -> Dict:
    matches, unmatched_pred, unmatched_gt, _ = greedy_match(
        preds,
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
    mean_iou = float(np.mean([m.get("iou", 0.0) for m in matches])) if matches else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1, "mean_iou_3d": mean_iou}


def _write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Offline pose sensitivity diagnostic from by-run JSON.")
    parser.add_argument("--by-run-json", default=None, help="Explicit by_run JSON path")
    parser.add_argument("--results-root", default=None, help="Results root containing by_run/")
    parser.add_argument("--scene", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--method-tag", default=None)
    parser.add_argument("--dx-grid", default="-0.10,0.0,0.10")
    parser.add_argument("--dy-grid", default="-0.10,0.0,0.10")
    parser.add_argument("--dz-grid", default="0.0")
    parser.add_argument("--yaw-grid-deg", default="-5.0,0.0,5.0")
    args = parser.parse_args()

    by_run_json = Path(args.by_run_json) if args.by_run_json else None
    if by_run_json is None:
        if not args.results_root or not args.scene or not args.run_id:
            raise SystemExit("[ERR] Provide --by-run-json OR (--results-root, --scene, --run-id).")
        p = _pick_by_method(Path(args.results_root), args.scene, args.run_id, args.method_tag)
        if p is None:
            raise SystemExit("[ERR] Could not find matching by_run JSON.")
        by_run_json = p

    data = _load_json(by_run_json)
    preds = data.get("instances", {}).get("predictions", [])
    gts = data.get("instances", {}).get("gts", [])
    if not preds or not gts:
        raise SystemExit("[ERR] by_run JSON missing instances.predictions or instances.gts.")

    iou_threshold = float(data.get("matching", {}).get("iou_threshold", 0.1))
    require_label_match = bool(data.get("matching", {}).get("require_label_match", False))
    iou_mode = str(data.get("matching", {}).get("iou_mode", "aabb")).strip().lower()

    dxs = [float(x) for x in args.dx_grid.split(",")]
    dys = [float(x) for x in args.dy_grid.split(",")]
    dzs = [float(x) for x in args.dz_grid.split(",")]
    yaws = [float(x) for x in args.yaw_grid_deg.split(",")]

    rows = []
    for yaw in yaws:
        for dx in dxs:
            for dy in dys:
                for dz in dzs:
                    p2 = [_transform_prediction(p, yaw_deg=yaw, dx=dx, dy=dy, dz=dz) for p in preds]
                    m = _compute_metrics(
                        p2,
                        gts,
                        iou_threshold=iou_threshold,
                        require_label_match=require_label_match,
                        iou_mode=iou_mode,
                    )
                    rows.append({"yaw_deg": yaw, "dx_m": dx, "dy_m": dy, "dz_m": dz, **m})

    baseline = next((r for r in rows if abs(r["yaw_deg"]) < 1e-9 and abs(r["dx_m"]) < 1e-9 and abs(r["dy_m"]) < 1e-9 and abs(r["dz_m"]) < 1e-9), None)
    baseline_f1 = float(baseline["f1"]) if baseline else 0.0
    f1_vals = np.asarray([float(r["f1"]) for r in rows], dtype=np.float64) if rows else np.zeros((0,), dtype=np.float64)
    summary = {
        "by_run_json": str(by_run_json.resolve()),
        "scene": data.get("metadata", {}).get("scene"),
        "run_id": data.get("metadata", {}).get("run_id"),
        "method_tag": data.get("metadata", {}).get("method_tag"),
        "sensor_source": data.get("metadata", {}).get("sensor_source", "unknown"),
        "num_samples": int(len(rows)),
        "baseline_f1": baseline_f1,
        "min_f1": float(np.min(f1_vals)) if f1_vals.size else 0.0,
        "max_f1": float(np.max(f1_vals)) if f1_vals.size else 0.0,
        "mean_f1": float(np.mean(f1_vals)) if f1_vals.size else 0.0,
        "std_f1": float(np.std(f1_vals)) if f1_vals.size else 0.0,
        "pose_sensitivity_score": float(baseline_f1 - np.min(f1_vals)) if f1_vals.size else 0.0,
    }

    results_root = Path(data.get("metadata", {}).get("results_root", by_run_json.parents[1]))
    diag_dir = results_root / "diagnostics"
    stem = f"{summary.get('scene','unknown')}_{summary.get('run_id','unknown')}_{summary.get('method_tag') or 'method'}_pose_sensitivity"
    csv_path = diag_dir / f"{stem}.csv"
    json_path = diag_dir / f"{stem}.json"
    _write_csv(csv_path, rows)
    diag_dir.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump({"summary": summary, "grid": rows}, f, indent=2)

    print(f"[OK] wrote {csv_path}")
    print(f"[OK] wrote {json_path}")


if __name__ == "__main__":
    main()
