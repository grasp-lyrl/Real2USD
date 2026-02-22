import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R


def _read_json(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def _load_points(path: str) -> np.ndarray:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".npy":
        pts = np.load(str(p))
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError(f"{path}: expected Nx3 npy")
        return np.asarray(pts[:, :3], dtype=np.float64)
    if suffix in (".ply", ".pcd"):
        try:
            import open3d as o3d
        except Exception as e:
            raise RuntimeError(f"open3d required for {suffix} input: {e}")
        pc = o3d.io.read_point_cloud(str(p))
        pts = np.asarray(pc.points, dtype=np.float64)
        if pts.size == 0:
            raise ValueError(f"{path}: empty point cloud")
        return pts
    raise ValueError(f"Unsupported point cloud format: {path}")


def _apply_T(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    ph = np.hstack([pts, ones])
    return (T @ ph.T).T[:, :3]


def _voxel_downsample(pts: np.ndarray, voxel: float) -> np.ndarray:
    if voxel <= 0:
        return pts
    q = np.floor(pts / voxel).astype(np.int64)
    _, idx = np.unique(q, axis=0, return_index=True)
    return pts[np.sort(idx)]


def _cap_points(pts: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or len(pts) <= max_points:
        return pts
    idx = np.linspace(0, len(pts) - 1, max_points, dtype=np.int64)
    return pts[idx]


def _nn_stats(a: np.ndarray, b: np.ndarray, cap_m: float) -> Dict:
    if len(a) == 0 or len(b) == 0:
        return {"mean": None, "median": None, "p95": None}
    tree = cKDTree(b)
    d, _ = tree.query(a, k=1)
    d = np.asarray(d, dtype=np.float64)
    d = np.minimum(d, cap_m)
    return {
        "mean": float(np.mean(d)),
        "median": float(np.median(d)),
        "p95": float(np.percentile(d, 95)),
    }


def _overlap_metrics(src: np.ndarray, tgt: np.ndarray, cap_m: float) -> Dict:
    s2t = _nn_stats(src, tgt, cap_m)
    t2s = _nn_stats(tgt, src, cap_m)
    chamfer = None
    if s2t["mean"] is not None and t2s["mean"] is not None:
        chamfer = float(0.5 * (s2t["mean"] + t2s["mean"]))
    return {"src_to_tgt": s2t, "tgt_to_src": t2s, "chamfer_mean": chamfer}


def _best_fit_transform(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # Solve R,t minimizing ||R*A+t - B||_2
    cA = A.mean(axis=0)
    cB = B.mean(axis=0)
    AA = A - cA
    BB = B - cB
    H = AA.T @ BB
    U, _, Vt = np.linalg.svd(H)
    Rm = Vt.T @ U.T
    if np.linalg.det(Rm) < 0:
        Vt[-1, :] *= -1
        Rm = Vt.T @ U.T
    t = cB - Rm @ cA
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rm
    T[:3, 3] = t
    return T


def _icp_point_to_point(
    src: np.ndarray,
    tgt: np.ndarray,
    max_iterations: int,
    max_corr: float,
    min_corr: int,
    conv_trans_m: float,
    conv_rot_deg: float,
) -> Tuple[np.ndarray, Dict]:
    T = np.eye(4, dtype=np.float64)
    tree = cKDTree(tgt)
    last_fit = {"fitness": 0.0, "residual_m": None, "num_corr": 0}
    for _ in range(max_iterations):
        src_tf = _apply_T(src, T)
        d, idx = tree.query(src_tf, k=1)
        d = np.asarray(d, dtype=np.float64)
        idx = np.asarray(idx, dtype=np.int64)
        mask = d <= max_corr
        if int(np.count_nonzero(mask)) < min_corr:
            break
        A = src_tf[mask]
        B = tgt[idx[mask]]
        dT = _best_fit_transform(A, B)
        T_new = dT @ T
        delta_t = float(np.linalg.norm(dT[:3, 3]))
        delta_r = float(np.degrees(np.linalg.norm(R.from_matrix(dT[:3, :3]).as_rotvec())))
        T = T_new
        last_fit = {
            "fitness": float(np.count_nonzero(mask)) / float(len(src)),
            "residual_m": float(np.mean(d[mask])) if np.any(mask) else None,
            "num_corr": int(np.count_nonzero(mask)),
        }
        if delta_t < conv_trans_m and delta_r < conv_rot_deg:
            break
    return T, last_fit


def _acceptance(before: Dict, after: Dict, acc_cfg: Dict) -> Dict:
    p95_before = before["src_to_tgt"]["p95"]
    p95_after = after["src_to_tgt"]["p95"]
    if p95_before is None or p95_after is None:
        return {"accepted": False, "reason": "missing_metrics"}
    improve = float(p95_before - p95_after)
    if acc_cfg.get("require_improvement", True) and improve < float(acc_cfg.get("min_p95_improvement_m", 0.0)):
        return {"accepted": False, "reason": "insufficient_improvement", "improvement_m": improve}
    if p95_after > float(acc_cfg.get("max_p95_after_m", 1e9)):
        return {"accepted": False, "reason": "p95_after_too_high", "improvement_m": improve}
    return {"accepted": True, "reason": "ok", "improvement_m": improve}


def _write_csv_row(path: Path, row: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Geometry-first sensor alignment evaluator and optional correction estimator.")
    parser.add_argument("--realsense-points", required=True, help="Path to realsense points (.npy/.ply/.pcd), in odom frame")
    parser.add_argument("--lidar-points", required=True, help="Path to lidar points (.npy/.ply/.pcd), in odom frame")
    parser.add_argument("--config", default="/home/hsu/repos/Real2USD/humble_ws/evaluations/sensor_alignment_config.json")
    parser.add_argument("--results-root", default="/home/hsu/repos/Real2USD/humble_ws/evaluations/results")
    parser.add_argument("--scene", default="unknown_scene")
    parser.add_argument("--run-id", default="unknown_run")
    args = parser.parse_args()

    cfg = _read_json(args.config)
    pre = cfg.get("preprocess", {})
    ov = cfg.get("overlap_metrics", {})
    icp = cfg.get("icp", {})
    acc = cfg.get("acceptance", {})

    rs = _load_points(args.realsense_points)
    ld = _load_points(args.lidar_points)
    rs = _cap_points(_voxel_downsample(rs, float(pre.get("voxel_size_m", 0.0))), int(pre.get("max_points", 0)))
    ld = _cap_points(_voxel_downsample(ld, float(pre.get("voxel_size_m", 0.0))), int(pre.get("max_points", 0)))
    cap_m = float(ov.get("nn_distance_cap_m", 2.0))

    before = _overlap_metrics(rs, ld, cap_m)
    T = np.eye(4, dtype=np.float64)
    icp_fit = {"fitness": None, "residual_m": None, "num_corr": 0}
    after = before
    if bool(icp.get("enabled", True)):
        T, icp_fit = _icp_point_to_point(
            src=rs,
            tgt=ld,
            max_iterations=int(icp.get("max_iterations", 30)),
            max_corr=float(icp.get("max_correspondence_m", 0.5)),
            min_corr=int(icp.get("min_correspondences", 200)),
            conv_trans_m=float(icp.get("convergence_translation_m", 1e-3)),
            conv_rot_deg=float(icp.get("convergence_rotation_deg", 0.1)),
        )
        rs_corr = _apply_T(rs, T)
        after = _overlap_metrics(rs_corr, ld, cap_m)

    decision = _acceptance(before, after, acc)
    euler = R.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)
    tx = T[:3, 3]

    out = {
        "metadata": {
            "scene": args.scene,
            "run_id": args.run_id,
            "realsense_points": str(Path(args.realsense_points).resolve()),
            "lidar_points": str(Path(args.lidar_points).resolve()),
            "config": str(Path(args.config).resolve()),
            "created_at": datetime.now().isoformat(),
        },
        "before": before,
        "after": after,
        "icp_fit": icp_fit,
        "transform_lidar_from_realsense": {
            "matrix_4x4": T.tolist(),
            "translation_xyz_m": tx.tolist(),
            "translation_norm_m": float(np.linalg.norm(tx)),
            "rotation_euler_xyz_deg": euler.tolist(),
        },
        "decision": decision,
    }

    results_root = Path(args.results_root)
    diag = results_root / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)
    out_json = diag / f"sensor_alignment_{args.scene}_{args.run_id}.json"
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[OK] wrote {out_json}")

    row = {
        "scene": args.scene,
        "run_id": args.run_id,
        "nn_median_m_before": before["src_to_tgt"]["median"],
        "nn_p95_m_before": before["src_to_tgt"]["p95"],
        "chamfer_mean_before": before["chamfer_mean"],
        "nn_median_m_after": after["src_to_tgt"]["median"],
        "nn_p95_m_after": after["src_to_tgt"]["p95"],
        "chamfer_mean_after": after["chamfer_mean"],
        "icp_fitness": icp_fit.get("fitness"),
        "icp_residual_m": icp_fit.get("residual_m"),
        "used_alignment_correction": bool(icp.get("enabled", True)),
        "accepted": decision.get("accepted"),
        "decision_reason": decision.get("reason"),
        "translation_norm_m": float(np.linalg.norm(tx)),
        "yaw_deg": float(euler[2]),
    }
    csv_path = results_root / "tables" / "sensor_alignment_latest.csv"
    _write_csv_row(csv_path, row)
    print(f"[OK] appended {csv_path}")


if __name__ == "__main__":
    main()
