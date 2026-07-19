"""Score the OLD v1 (real2sam3d) SAM3D outputs against Supervisely GT with the SAME v2
metrics, for an apples-to-apples v1-vs-v2 baseline on the real Go2 scenes.

v1 outputs (/data/sam3d/<v1scene>/): scene_graph.json keyed by step; each object has an
open-vocab label + a posed mesh in glbs_world/<step>/<label>.glb. v1's world frame =
absolute-odom orientation & Z, but XY shifted by the first robot pose (demo_go2.py subtracts
init_odom["t"][:2]). So map to the absolute GT frame by a PURE XY translation read from
step_init/odom_rs.json (validated: ADD gives matches, SUB/raw give none).

Boxes: v1 scene_graph only stores an axis-aligned aabb_bounds (inflated for rotated objects),
so for a fair OBB-IoU vs v2 we recompute the oriented bbox from the posed glbs_world mesh.

Run:  uv run python scripts/score_v1_baseline.py <v1scene> <gt-scene> [--dedup M]
  e.g. uv run python scripts/score_v1_baseline.py hallway-01-rs hallway-1
"""
import sys, json, argparse
from pathlib import Path

import numpy as np
import trimesh

from r2s3d_core.data.realsense import resolve_rs_scene
from r2s3d_core.data.supervisely import load_supervisely_gt
from r2s3d_core.eval.metrics import SceneObject, gt_to_scene_object, evaluate

ap = argparse.ArgumentParser()
ap.add_argument("v1scene")                       # dir name under /data/sam3d
ap.add_argument("gt_scene")                       # r2s3d scene id for GT + offset
ap.add_argument("--root", default="/data/sam3d")
ap.add_argument("--dedup", type=float, default=0.0,
                help="if >0, greedily suppress same-XY duplicates within this many metres "
                     "(v1 had no cross-frame consolidation)")
ap.add_argument("--iou", type=float, default=0.25)
args = ap.parse_args()

scene_dir = Path(args.root) / args.v1scene
sg = json.load(open(scene_dir / "scene_graph.json"))
off = np.array(json.load(open(scene_dir / "step_init" / "odom_rs.json"))["t"], float)
off[2] = 0.0                                       # XY-only translation to absolute odom
print(f"v1 scene={args.v1scene}  XY offset to absolute odom = ({off[0]:.3f}, {off[1]:.3f})")

preds = []
skipped = 0
for step, objs in sg.items():
    for label, info in objs.items():
        glb = scene_dir / "glbs_world" / step / f"{label}.glb"
        if not glb.exists():
            skipped += 1
            continue
        m = trimesh.load(glb, force="mesh")
        if not isinstance(m, trimesh.Trimesh) or m.vertices.shape[0] < 4:
            skipped += 1
            continue
        m.apply_translation(off)                   # v1-world -> absolute odom (pure XY shift)
        try:
            obb = m.bounding_box_oriented
            T = np.asarray(obb.primitive.transform, float)
            ext = np.asarray(obb.primitive.extents, float)
        except Exception:
            skipped += 1
            continue
        preds.append(SceneObject(label=label, T_world_obj=T, extents=ext, mesh=m,
                                 confidence=float(info.get("confidence", 1.0))))
print(f"loaded {len(preds)} v1 objects ({skipped} skipped/missing mesh)")

if args.dedup > 0:
    kept = []
    for p in sorted(preds, key=lambda o: -o.confidence):
        c = p.T_world_obj[:3, 3]
        if all(np.linalg.norm(c[:2] - k.T_world_obj[:2, 3]) > args.dedup for k in kept):
            kept.append(p)
    print(f"dedup @{args.dedup}m: {len(preds)} -> {len(kept)}")
    preds = kept

_, gtjson = resolve_rs_scene(args.gt_scene)
gts = [gt_to_scene_object(g) for g in load_supervisely_gt(gtjson)]
m = evaluate(preds, gts, iou_threshold=args.iou, compute_geometry=False)
a = m["aggregate"] if "aggregate" in m else m

def g(k):
    return a.get(k, float("nan"))

print(f"\n=== v1 baseline: {args.v1scene} vs GT {args.gt_scene} (n_pred={len(preds)} n_gt={len(gts)}) ===")
for lbl, k in [("iou_f1@%.2f" % args.iou, "f1"), ("iou_precision", "precision"),
               ("iou_recall", "recall"), ("recall@0.5", "recall@0.5"),
               ("scan2cad", "scan2cad_accuracy"), ("centroid_med_m", "centroid_err_median_m"),
               ("rot_med_deg", "rotation_err_median_deg"), ("scale_err_med", "scale_err_median"),
               ("cd_f1@1m", "cd_f1"), ("cd_prec@1m", "cd_precision"), ("cd_recall@1m", "cd_recall"),
               ("class_free_recall_1m", "class_free_recall_1m"), ("label_acc", "label_accuracy"),
               ("duplicate_rate", "duplicate_rate")]:
    v = g(k)
    print(f"  {lbl:22s} {v:.3f}" if isinstance(v, (int, float)) else f"  {lbl:22s} {v}")
