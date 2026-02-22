import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def _load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def _rect_from_aabb_xy(aabb_min, aabb_max):
    x0, y0 = float(aabb_min[0]), float(aabb_min[1])
    x1, y1 = float(aabb_max[0]), float(aabb_max[1])
    return x0, y0, (x1 - x0), (y1 - y0)


def main():
    parser = argparse.ArgumentParser(description="Overlay GT and predicted bbox footprints from by-run eval JSON.")
    parser.add_argument("--by-run-json", required=True, help="Path to one by_run *.json produced by run_eval.py")
    parser.add_argument("--out-png", default=None, help="Optional output png path")
    args = parser.parse_args()

    by_run_path = Path(args.by_run_json)
    data = _load_json(by_run_path)
    instances = data.get("instances", {})
    preds = instances.get("predictions", [])
    gts = instances.get("gts", [])
    matches = data.get("matches", [])
    unmatched_pred = set(data.get("unmatched_prediction_indices", []))
    unmatched_gt = set(data.get("unmatched_gt_indices", []))

    fig, ax = plt.subplots(figsize=(9, 8))

    # Draw GT as black outlines; highlight unmatched GT in orange.
    for gi, g in enumerate(gts):
        if not g.get("aabb_min") or not g.get("aabb_max"):
            continue
        x, y, w, h = _rect_from_aabb_xy(g["aabb_min"], g["aabb_max"])
        color = "orange" if gi in unmatched_gt else "black"
        lw = 2.0 if gi in unmatched_gt else 1.2
        rect = Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=lw, linestyle="-")
        ax.add_patch(rect)
        cx, cy = float(g["center"][0]), float(g["center"][1])
        ax.text(cx, cy, f"GT:{g.get('label_canonical')}", color=color, fontsize=7)

    # Draw predictions: matched green, unmatched red dashed.
    matched_pred = {m["pred_idx"] for m in matches}
    for pi, p in enumerate(preds):
        if not p.get("aabb_min") or not p.get("aabb_max"):
            continue
        x, y, w, h = _rect_from_aabb_xy(p["aabb_min"], p["aabb_max"])
        is_matched = pi in matched_pred
        color = "green" if is_matched else "red"
        ls = "-" if is_matched else "--"
        rect = Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=1.2, linestyle=ls)
        ax.add_patch(rect)
        cx, cy = float(p["center"][0]), float(p["center"][1])
        ax.text(cx, cy, f"P:{p.get('label_canonical')}", color=color, fontsize=7)

    # Draw match center lines and labels.
    for m in matches:
        pi, gi = int(m["pred_idx"]), int(m["gt_idx"])
        if pi >= len(preds) or gi >= len(gts):
            continue
        pc = preds[pi].get("center")
        gc = gts[gi].get("center")
        if pc is None or gc is None:
            continue
        ax.plot([pc[0], gc[0]], [pc[1], gc[1]], color="blue", alpha=0.35, linewidth=1.0)
        mx, my = (pc[0] + gc[0]) * 0.5, (pc[1] + gc[1]) * 0.5
        ax.text(mx, my, f"iou={m.get('iou', 0):.2f}", color="blue", fontsize=7)

    scene = data.get("metadata", {}).get("scene", "unknown_scene")
    run_id = data.get("metadata", {}).get("run_id", "unknown_run")
    ax.set_title(f"BBox Overlay XY: {scene} / {run_id}")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    if args.out_png:
        out = Path(args.out_png)
    else:
        out = by_run_path.parent.parent / "plots" / f"{by_run_path.stem}_bbox_overlay_xy.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=220)
    plt.close()
    print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()
