"""
Plot bounding boxes overlaid with GT boxes and (optionally) with point cloud.
Boxes are drawn as rotated OBBs (oriented bounding boxes) using obb_center,
obb_dimensions, obb_rotation_matrix when available (no AABB).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.patches import Rectangle

# Allow imports from parent
import sys
_EVAL_ROOT = Path(__file__).resolve().parent.parent
if str(_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_EVAL_ROOT))


def _load_json(path: Path) -> Dict:
    with open(path) as f:
        return json.load(f)


def _yaw_rotation_matrix_rad(yaw_rad: float) -> np.ndarray:
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


# Minimum size for drawn boxes when OBB dimensions are zero (e.g. pipeline fallback)
_MIN_BOX_SIZE_M = 0.4


def _obb_xy_from_box(box: Dict) -> Optional[Tuple[float, float, float, float, float]]:
    """Get (center_x, center_y, width, height, yaw_rad) for the XY footprint of a 3D OBB.
    Uses the axis most aligned with world Z as vertical (ignored for footprint); the other
    two axes define the 2D rectangle and yaw. Returns None if no usable data."""
    center = box.get("obb_center") or box.get("center")
    if not center or len(center) < 3:
        return None
    cx, cy = float(center[0]), float(center[1])
    dims = box.get("obb_dimensions") or box.get("dimensions") or [0.0, 0.0, 0.0]
    dims = [float(dims[i]) if i < len(dims) else 0.0 for i in range(3)]
    R = box.get("obb_rotation_matrix")
    if R is None or not isinstance(R, (list, tuple)) or len(R) < 3:
        # No rotation: assume dims are world XYZ, use dim[0], dim[1] for XY
        dx = dims[0] if dims[0] >= 1e-6 else _MIN_BOX_SIZE_M
        dy = dims[1] if dims[1] >= 1e-6 else _MIN_BOX_SIZE_M
        return cx, cy, dx, dy, 0.0
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    # Which OBB axis is most aligned with world Z (vertical)?
    z_component = np.abs(R[2, :])
    k_z = int(np.argmax(z_component))
    # The other two axes are horizontal; use them for the XY footprint
    i_h, j_h = [i for i in range(3) if i != k_z]
    dx = dims[i_h] if dims[i_h] >= 1e-6 else _MIN_BOX_SIZE_M
    dy = dims[j_h] if dims[j_h] >= 1e-6 else _MIN_BOX_SIZE_M
    # Yaw = angle in XY plane of the first horizontal axis (column i_h)
    yaw = float(np.arctan2(R[1, i_h], R[0, i_h]))
    return cx, cy, dx, dy, yaw


def _obb_corners_xy(cx: float, cy: float, dx: float, dy: float, yaw: float) -> np.ndarray:
    """Four XY corners of the OBB (for axis limits)."""
    hx, hy = dx * 0.5, dy * 0.5
    c = np.cos(yaw)
    s = np.sin(yaw)
    corners = np.array([
        [-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]
    ], dtype=np.float64)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return (rot @ corners.T).T + np.array([cx, cy], dtype=np.float64)


def _load_pointcloud_xy(run_dir: Path, yaw_rad: float) -> Optional[np.ndarray]:
    """Load latest or concatenated lidar PC from run_dir/diagnostics/pointclouds/lidar/*.npy, apply -yaw to align with eval frame."""
    pc_dir = run_dir / "diagnostics" / "pointclouds" / "lidar"
    if not pc_dir.exists():
        pc_dir = run_dir / "diagnostics" / "pointclouds" / "realsense"
    if not pc_dir.exists():
        return None
    npy_files = sorted(pc_dir.glob("*.npy"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not npy_files:
        return None
    try:
        pts = np.load(npy_files[0])
        if pts.ndim == 2 and pts.shape[1] >= 2:
            xy = np.asarray(pts[:, :2], dtype=np.float64)
        else:
            return None
        if yaw_rad != 0:
            R = _yaw_rotation_matrix_rad(-yaw_rad)
            xy = (R @ xy.T).T
        return xy
    except Exception:
        return None


def _draw_bboxes_ax(
    ax,
    preds: List[Dict],
    gts: List[Dict],
    matches: List[Dict],
    unmatched_pred: set,
    unmatched_gt: set,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Draw GT and pred boxes as rotated OBBs; return (xmin, xmax, ymin, ymax) for axis limits."""
    xs, ys = [], []

    def draw_obb(box: Dict, color: str, linewidth: float, linestyle: str, label: str) -> None:
        obb = _obb_xy_from_box(box)
        if obb is None:
            return
        cx, cy, dx, dy, yaw = obb
        corners = _obb_corners_xy(cx, cy, dx, dy, yaw)
        xs.extend(corners[:, 0].tolist())
        ys.extend(corners[:, 1].tolist())
        rect = Rectangle(
            (cx - dx / 2, cy - dy / 2),
            dx,
            dy,
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
            linestyle=linestyle,
        )
        t = transforms.Affine2D().rotate_around(cx, cy, yaw) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)
        ax.text(cx, cy, label, color=color, fontsize=7, ha="center", va="center")

    for gi, g in enumerate(gts):
        color = "orange" if gi in unmatched_gt else "black"
        lw = 2.5 if gi in unmatched_gt else 2.0
        lbl = g.get("label_canonical")
        gt_label = lbl if (lbl and str(lbl).strip()) else "(unlabeled)"
        draw_obb(g, color, lw, "-", f"GT:{gt_label}")

    matched_pred = {m["pred_idx"] for m in matches}
    for pi, p in enumerate(preds):
        is_matched = pi in matched_pred
        color = "green" if is_matched else "red"
        ls = "-" if is_matched else "--"
        lbl = p.get("label_canonical")
        pred_label = lbl if (lbl and str(lbl).strip()) else "(unlabeled)"
        draw_obb(p, color, 2.0, ls, f"P:{pred_label}")

    for m in matches:
        pi, gi = int(m["pred_idx"]), int(m["gt_idx"])
        if pi >= len(preds) or gi >= len(gts):
            continue
        pc = preds[pi].get("center") or preds[pi].get("obb_center")
        gc = gts[gi].get("center") or gts[gi].get("obb_center")
        if pc is None or gc is None or len(pc) < 2 or len(gc) < 2:
            continue
        ax.plot([pc[0], gc[0]], [pc[1], gc[1]], color="blue", alpha=0.35, linewidth=1.0)
        mx, my = (pc[0] + gc[0]) * 0.5, (pc[1] + gc[1]) * 0.5
        ax.text(mx, my, f"iou={m.get('iou', 0):.2f}", color="blue", fontsize=7)

    if not xs or not ys:
        return None, None, None, None
    pad = max((max(xs) - min(xs)), (max(ys) - min(ys)), 1.0) * 0.05
    return min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot bbox overlays: pred+GT and pred+GT on point cloud (OBB).")
    parser.add_argument("--by-run-json", required=True, help="Path to by_run *.json from run_eval")
    parser.add_argument("--run-dir", default=None, help="Run dir for point cloud (default: from metadata or parent of prediction_path)")
    parser.add_argument("--out-dir", default=None, help="Output directory for PNGs (default: results_root/plots)")
    parser.add_argument("--out-png", default=None, help="Optional path for bbox-vs-GT figure (legacy compat)")
    parser.add_argument("--no-pc", action="store_true", help="Skip point-cloud overlay even if PC exists")
    args = parser.parse_args()

    by_run_path = Path(args.by_run_json)
    data = _load_json(by_run_path)
    instances = data.get("instances", {})
    preds = instances.get("predictions", [])
    gts = instances.get("gts", [])
    matches = data.get("matches", [])
    unmatched_pred = set(data.get("unmatched_prediction_indices", []))
    unmatched_gt = set(data.get("unmatched_gt_indices", []))

    # Log counts so user can see why overlay may show fewer than scene_graph objects
    n_pred = len(preds)
    n_gt = len(gts)
    n_drawn_pred = sum(1 for p in preds if _obb_xy_from_box(p) is not None)
    n_drawn_gt = sum(1 for g in gts if _obb_xy_from_box(g) is not None)
    if n_drawn_pred < n_pred or n_drawn_gt < n_gt:
        print(f"[INFO] Predictions: {n_drawn_pred}/{n_pred} drawn (missing center/dims for {n_pred - n_drawn_pred})")
        print(f"[INFO] GT: {n_drawn_gt}/{n_gt} drawn (missing center/dims for {n_gt - n_drawn_gt})")
    else:
        print(f"[INFO] Drawing {n_pred} predictions, {n_gt} GT boxes")

    meta = data.get("metadata", {})
    scene = meta.get("scene", "unknown_scene")
    run_id = meta.get("run_id", "unknown_run")
    method_tag = meta.get("method_tag") or "default"
    frame_norm = meta.get("frame_normalization", {}) or {}
    yaw_deg = float(frame_norm.get("gt_global_yaw_deg", 0.0))
    yaw_rad = np.radians(yaw_deg)
    results_root = meta.get("results_root")
    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif results_root:
        out_dir = Path(results_root) / "plots"
    else:
        out_dir = by_run_path.parent.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{by_run_path.stem}_overlay"

    # 1) Bbox + GT overlay (OBB)
    fig1, ax1 = plt.subplots(figsize=(9, 8))
    xmin, xmax, ymin, ymax = _draw_bboxes_ax(ax1, preds, gts, matches, unmatched_pred, unmatched_gt)
    ax1.set_title(f"BBox vs GT (OBB): {scene} / {run_id} / {method_tag}")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    if xmin is not None and xmax is not None and ymin is not None and ymax is not None:
        ax1.set_xlim(xmin, xmax)
        ax1.set_ylim(ymin, ymax)
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.2)
    plt.tight_layout()
    png_bbox_gt = Path(args.out_png) if args.out_png else (out_dir / f"{stem}_bbox_gt.png")
    png_bbox_gt.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_bbox_gt, dpi=220)
    plt.close()
    print(f"[OK] wrote {png_bbox_gt}")

    # 2) Bbox + GT over point cloud (if PC available)
    run_dir = Path(args.run_dir) if args.run_dir else None
    if not run_dir and meta.get("prediction_path"):
        run_dir = Path(meta["prediction_path"]).resolve().parent
    pc_xy = None if args.no_pc else ( _load_pointcloud_xy(run_dir, yaw_rad) if run_dir else None )

    if pc_xy is not None and len(pc_xy) > 0:
        fig2, ax2 = plt.subplots(figsize=(10, 9))
        ax2.scatter(pc_xy[:, 0], pc_xy[:, 1], s=0.1, c="gray", alpha=0.4, rasterized=True)
        xmin, xmax, ymin, ymax = _draw_bboxes_ax(ax2, preds, gts, matches, unmatched_pred, unmatched_gt)
        ax2.set_title(f"BBox + GT on point cloud (OBB): {scene} / {run_id} / {method_tag}")
        ax2.set_xlabel("X (m)")
        ax2.set_ylabel("Y (m)")
        if xmin is not None and xmax is not None and ymin is not None and ymax is not None:
            ax2.set_xlim(xmin, xmax)
            ax2.set_ylim(ymin, ymax)
        ax2.set_aspect("equal")
        ax2.grid(True, alpha=0.2)
        plt.tight_layout()
        png_pc = out_dir / f"{stem}_bbox_on_pc.png"
        plt.savefig(png_pc, dpi=220)
        plt.close()
        print(f"[OK] wrote {png_pc}")
    else:
        if not args.no_pc:
            print("[WARN] No point cloud found; run_dir/diagnostics/pointclouds/lidar/*.npy or realsense/*.npy missing or --run-dir not set.")


if __name__ == "__main__":
    main()
