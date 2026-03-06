"""
Plot bounding boxes overlaid with GT boxes and (optionally) with point cloud.
Boxes are drawn as rotated OBBs (oriented bounding boxes) using obb_center,
obb_dimensions, obb_rotation_matrix when available (no AABB).

Supports two inputs:
  - --by-run-json: by_run JSON from run_eval (predictions + GT + matches).
  - --usda: USDA file path; parses via parse_usda_data and plots OBBs (boundingBoxMin/Max
    + orient). Optionally --gt-json (Supervisely) to overlay GT and run matching (match lines).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# Allow imports from parent
_EVAL_ROOT = Path(__file__).resolve().parent.parent
if str(_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_EVAL_ROOT))

# Seaborn style: theme + ticks grid, no title
sns.set_theme()
sns.set_style("ticks", rc={"axes.grid": True})

# Standard semantic colors (colorblind-friendly)
COLOR_GT_MATCHED = "0.15"       # dark gray – ground truth, matched
COLOR_GT_UNMATCHED = "C1"      # orange – GT with no prediction (FN)
COLOR_PRED_MATCHED = "C2"      # green – prediction matched to GT (TP)
COLOR_PRED_UNMATCHED = "C3"    # red – prediction not matched (FP)
COLOR_MATCH_LINE = "C0"        # blue – match link


def _load_json(path: Path) -> Dict:
    with open(path) as f:
        return json.load(f)


def _usda_to_overlay_boxes(usda_data: Dict) -> List[Dict]:
    """Convert USDA parse_usda_data output to overlay box list. Uses shared usda_to_prediction_boxes (no frame normalization)."""
    from usda_to_prediction_boxes import usda_data_to_prediction_boxes
    return usda_data_to_prediction_boxes(usda_data, normalize_yaw_rad=0.0)


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


def _has_label(box: Dict) -> bool:
    """True if box has a non-empty label (label_canonical or label)."""
    lbl = box.get("label_canonical") or box.get("label")
    return bool(lbl) and bool(str(lbl).strip())


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


def _hide_labels_outside_bounds(ax, xmin: float, xmax: float, ymin: float, ymax: float) -> None:
    """Hide any text labels whose position is outside the given plot bounds."""
    for text in ax.texts:
        pos = text.get_position()
        if len(pos) >= 2:
            x, y = float(pos[0]), float(pos[1])
            if x < xmin or x > xmax or y < ymin or y > ymax:
                text.set_visible(False)


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
    skip_unlabeled: bool = False,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Draw GT and pred boxes as rotated OBBs; return (xmin, xmax, ymin, ymax) for axis limits.
    Limits from GT extent or pred extent when USDA-only. If skip_unlabeled, boxes without label are not drawn."""
    gt_xs, gt_ys = [], []
    pred_xs, pred_ys = [], []

    def draw_obb(box: Dict, color: str, linewidth: float, linestyle: str, label: str, collect_for_limits: bool = False, collect_pred_limits: bool = False) -> None:
        obb = _obb_xy_from_box(box)
        if obb is None:
            return
        cx, cy, dx, dy, yaw = obb
        corners = _obb_corners_xy(cx, cy, dx, dy, yaw)
        if collect_for_limits:
            gt_xs.extend(corners[:, 0].tolist())
            gt_ys.extend(corners[:, 1].tolist())
        if collect_pred_limits:
            pred_xs.extend(corners[:, 0].tolist())
            pred_ys.extend(corners[:, 1].tolist())
        if not (np.isfinite(dx) and np.isfinite(dy) and dx > 0 and dy > 0):
            return
        rect = Rectangle(
            (cx - dx / 2, cy - dy / 2),
            dx,
            dy,
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=5,
        )
        t = transforms.Affine2D().rotate_around(cx, cy, yaw) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)
        ax.text(cx, cy, label, color=color, fontsize=11, ha="center", va="center", zorder=6)

    usda_only = len(gts) == 0

    for gi, g in enumerate(gts):
        if skip_unlabeled and not _has_label(g):
            continue
        color = COLOR_GT_UNMATCHED if gi in unmatched_gt else COLOR_GT_MATCHED
        lw = 2.5 if gi in unmatched_gt else 2.0
        lbl = g.get("label_canonical")
        gt_label = lbl if (lbl and str(lbl).strip()) else "(unlabeled)"
        draw_obb(g, color, lw, "-", gt_label, collect_for_limits=True)

    matched_pred = {m["pred_idx"] for m in matches}
    for pi, p in enumerate(preds):
        if skip_unlabeled and not _has_label(p):
            continue
        if usda_only:
            color = COLOR_PRED_MATCHED
            ls = "-"
        else:
            is_matched = pi in matched_pred
            color = COLOR_PRED_MATCHED if is_matched else COLOR_PRED_UNMATCHED
            ls = "-" if is_matched else "--"
        lbl = p.get("label_canonical") or p.get("label")
        pred_label = lbl if (lbl and str(lbl).strip()) else "(unlabeled)"
        draw_obb(p, color, 2.0, ls, pred_label, collect_for_limits=False, collect_pred_limits=usda_only)

    if not usda_only:
        for m in matches:
            pi, gi = int(m["pred_idx"]), int(m["gt_idx"])
            if pi >= len(preds) or gi >= len(gts):
                continue
            if skip_unlabeled and (not _has_label(preds[pi]) or not _has_label(gts[gi])):
                continue
            pc = preds[pi].get("center") or preds[pi].get("obb_center")
            gc = gts[gi].get("center") or gts[gi].get("obb_center")
            if pc is None or gc is None or len(pc) < 2 or len(gc) < 2:
                continue
            ax.plot([pc[0], gc[0]], [pc[1], gc[1]], color=COLOR_MATCH_LINE, alpha=0.7, linewidth=2.0)

    # Legend
    if usda_only:
        legend_handles = [Line2D([0], [0], color=COLOR_PRED_MATCHED, linewidth=2, linestyle="-", label="USDA")]
    else:
        legend_handles = [
            Line2D([0], [0], color=COLOR_GT_MATCHED, linewidth=2, linestyle="-", label="GT (matched)"),
            Line2D([0], [0], color=COLOR_GT_UNMATCHED, linewidth=2.5, linestyle="-", label="GT (unmatched)"),
            Line2D([0], [0], color=COLOR_PRED_MATCHED, linewidth=2, linestyle="-", label="Pred (matched)"),
            Line2D([0], [0], color=COLOR_PRED_UNMATCHED, linewidth=2, linestyle="--", label="Pred (unmatched)"),
            Line2D([0], [0], color=COLOR_MATCH_LINE, linewidth=2, linestyle="-", alpha=0.7, label="Match"),
        ]
    ax.legend(handles=legend_handles, loc="best", frameon=True, fontsize=12)

    # Axis limits from GT or (when USDA only) from preds
    xs, ys = (gt_xs, gt_ys) if (gt_xs and gt_ys) else (pred_xs, pred_ys)
    if xs and ys:
        span_x = max(xs) - min(xs)
        span_y = max(ys) - min(ys)
        pad = max(span_x, span_y, 1.0) * 0.05
        pad = max(pad, 0.5)
        return min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad
    return None, None, None, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot bbox overlays: pred+GT and pred+GT on point cloud (OBB). Supports by_run JSON or USDA file.")
    parser.add_argument("--by-run-json", default=None, help="Path to by_run *.json from run_eval (required unless --usda)")
    parser.add_argument("--usda", default=None, help="Path to USDA file; plot boxes from parse_usda (same style as overlay)")
    parser.add_argument("--gt-json", default=None, help="Supervisely GT JSON (with --usda); load via eval_common and run matching for overlay")
    parser.add_argument("--eval-config", default=None, help="Eval config for label resolution when using --gt-json (default: eval_root/eval_config.json)")
    parser.add_argument("--run-dir", default=None, help="Run dir for point cloud (default: from metadata or parent of prediction_path)")
    parser.add_argument("--out-dir", default=None, help="Output directory for PNGs (default: results_root/plots or ./plots)")
    parser.add_argument("--out-png", default=None, help="Optional path for bbox-vs-GT figure (legacy compat)")
    parser.add_argument("--no-pc", action="store_true", help="Skip point-cloud overlay even if PC exists")
    parser.add_argument("--no-unlabeled", action="store_true", help="Do not plot objects that have no label (label_canonical/label empty or missing)")
    args = parser.parse_args([a for a in sys.argv[1:] if a.strip()])

    if args.usda:
        if args.by_run_json:
            parser.error("Use either --by-run-json or --usda, not both.")
        from parse_usda import parse_usda_data
        usda_path = Path(args.usda)
        if not usda_path.exists():
            raise SystemExit(f"USDA file not found: {usda_path}")
        usda_data = parse_usda_data(str(usda_path))
        preds = _usda_to_overlay_boxes(usda_data)
        gts = []
        matches = []
        unmatched_pred = set(range(len(preds)))
        unmatched_gt = set()
        yaw_rad = 0.0
        meta = {}

        if args.gt_json:
            gt_path = Path(args.gt_json)
            if not gt_path.exists():
                raise SystemExit(f"GT JSON not found: {gt_path}")
            eval_root = _EVAL_ROOT
            config_path = args.eval_config or str(eval_root / "eval_config.json")
            p_cfg = Path(config_path)
            if not p_cfg.is_absolute():
                p_cfg = (Path(os.getcwd()) / p_cfg).resolve()
                if not p_cfg.exists():
                    p_cfg = (Path(eval_root) / config_path).resolve()
            with open(p_cfg) as f:
                config = json.load(f)
            base = p_cfg.parent
            for key in ("canonical_labels", "label_aliases", "learned_label_aliases"):
                if key in config.get("paths", {}):
                    rel = config["paths"][key]
                    if not Path(rel).is_absolute():
                        config["paths"][key] = str((base / rel).resolve())
            paths = config.get("paths", {})
            from label_matching import load_label_configs, load_learned_aliases, resolve_label_with_learning
            canonical, alias_map, contains_rules = load_label_configs(
                paths.get("canonical_labels", ""), paths.get("label_aliases", "")
            )
            learned = load_learned_aliases(paths.get("learned_label_aliases", ""))
            label_cfg = config.get("label_resolution", {}) or {}
            use_emb = bool(label_cfg.get("use_embedding_for_unresolved", False))
            embed_min = float(label_cfg.get("embedding_min_score", 0.5))
            # Resolve USDA labels via label_aliases / canonical so matching can use label_canonical
            for p in preds:
                raw = p.get("label") or p.get("label_canonical") or ""
                resolved, _ = resolve_label_with_learning(
                    raw,
                    canonical=canonical,
                    alias_map=alias_map,
                    contains_rules=contains_rules,
                    learned_alias_map=learned,
                    use_embedding_for_unresolved=use_emb,
                    learn_new_aliases=False,
                    learned_alias_path=paths.get("learned_label_aliases", ""),
                    embedding_min_score=embed_min,
                )
                p["label_canonical"] = resolved if resolved else raw
            from eval_common import load_supervisely_gt
            gts = load_supervisely_gt(
                str(gt_path),
                canonical=canonical,
                alias_map=alias_map,
                contains_rules=contains_rules,
                learned_alias_map=learned,
                use_embedding_for_unresolved=use_emb,
                learn_new_aliases=False,
                learned_alias_path=paths.get("learned_label_aliases", ""),
                embedding_min_score=embed_min,
                normalize_yaw_rad=0.0,
            )
            from metrics_3d import greedy_match
            match_cfg = config.get("matching", {}) or {}
            iou_threshold = float(match_cfg.get("iou_threshold", 0.01))
            require_label = bool(match_cfg.get("require_label_match", True))
            iou_mode = str(match_cfg.get("iou_mode", "obb"))
            matches, unmatched_pred_list, unmatched_gt_list, _ = greedy_match(
                preds, gts, iou_threshold=iou_threshold, require_label_match=require_label, iou_mode=iou_mode
            )
            unmatched_pred = set(unmatched_pred_list)
            unmatched_gt = set(unmatched_gt_list)
            print(f"[INFO] USDA + GT: {len(preds)} preds, {len(gts)} GT, {len(matches)} matches")
        else:
            print(f"[INFO] USDA: drawing {len(preds)} boxes")

        out_dir = Path(args.out_dir) if args.out_dir else (usda_path.parent / "plots")
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{usda_path.stem}_overlay"
        n_pred = len(preds)
        n_gt = len(gts)
        skip_unlabeled = getattr(args, "no_unlabeled", False)
    else:
        if not args.by_run_json:
            parser.error("Provide --by-run-json or --usda")
        by_run_path = Path(args.by_run_json)
        data = _load_json(by_run_path)
        instances = data.get("instances", {})
        preds = instances.get("predictions", [])
        gts = instances.get("gts", [])
        matches = data.get("matches", [])
        unmatched_pred = set(data.get("unmatched_prediction_indices", []))
        unmatched_gt = set(data.get("unmatched_gt_indices", []))

        n_pred = len(preds)
        n_gt = len(gts)
        n_drawn_pred = sum(1 for p in preds if _obb_xy_from_box(p) is not None)
        n_drawn_gt = sum(1 for g in gts if _obb_xy_from_box(g) is not None)
        skip_unlabeled = getattr(args, "no_unlabeled", False)
        if skip_unlabeled:
            preds_draw = [p for p in preds if _has_label(p)]
            gts_draw = [g for g in gts if _has_label(g)]
            print(f"[INFO] --no-unlabeled: drawing {len(preds_draw)}/{n_pred} predictions, {len(gts_draw)}/{n_gt} GT boxes (labeled only)")
        if n_drawn_pred < n_pred or n_drawn_gt < n_gt:
            print(f"[INFO] Predictions: {n_drawn_pred}/{n_pred} drawn (missing center/dims for {n_pred - n_drawn_pred})")
            print(f"[INFO] GT: {n_drawn_gt}/{n_gt} drawn (missing center/dims for {n_gt - n_drawn_gt})")
        else:
            print(f"[INFO] Drawing {n_pred} predictions, {n_gt} GT boxes")

        meta = data.get("metadata", {})
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
    xmin, xmax, ymin, ymax = _draw_bboxes_ax(ax1, preds, gts, matches, unmatched_pred, unmatched_gt, skip_unlabeled=skip_unlabeled)
    ax1.set_xlabel("X (m)", fontsize=14)
    ax1.set_ylabel("Y (m)", fontsize=14)
    ax1.tick_params(axis="both", labelsize=12)
    if xmin is not None and xmax is not None and ymin is not None and ymax is not None:
        ax1.set_xlim(xmin, xmax)
        ax1.set_ylim(ymin, ymax)
        _hide_labels_outside_bounds(ax1, xmin, xmax, ymin, ymax)
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
        ax2.scatter(pc_xy[:, 0], pc_xy[:, 1], s=0.1, c="0.6", alpha=0.4, rasterized=True)
        xmin, xmax, ymin, ymax = _draw_bboxes_ax(ax2, preds, gts, matches, unmatched_pred, unmatched_gt, skip_unlabeled=skip_unlabeled)
        ax2.set_xlabel("X (m)", fontsize=14)
        ax2.set_ylabel("Y (m)", fontsize=14)
        ax2.tick_params(axis="both", labelsize=12)
        if xmin is not None and xmax is not None and ymin is not None and ymax is not None:
            ax2.set_xlim(xmin, xmax)
            ax2.set_ylim(ymin, ymax)
            _hide_labels_outside_bounds(ax2, xmin, xmax, ymin, ymax)
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
