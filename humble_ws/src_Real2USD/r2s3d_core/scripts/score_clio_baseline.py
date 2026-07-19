"""Score EXISTING Clio real-data scene graphs against the Supervisely GT with v2 metrics.

Clio (Maggio et al., RA-L'24) was run on the 4 Go2 scenes on an old machine; the processed
open-set object graphs live at /data/Clio/<scene>.graphml (node_type=object nodes carry an
open-set `name`, `bbox_pos` = world OBB center, `bbox_dim` = full extents, `bbox_orientation`
= 3x3 world rotation). Coordinates are already in the ABSOLUTE-odom Z-up frame = the same
frame as the Supervisely GT (verified: per-scene X/Y ranges coincide), so NO frame
reconciliation is needed (unlike the v1 baseline).

This gives a published-method baseline on OUR real-robot data, directly comparable to the v2
table (docs/STATUS.md "FULL 4-SCENE v2 REAL-ROBOT TABLE") and the v1 baseline
(scripts/score_v1_baseline.py). Box GT (cuboids) -> no mesh Chamfer; --label-map clip snaps
Clio's open-set labels to the GT vocab (fair, same protocol as our open-vocab runs).

Usage:
  uv run --extra registration [--extra detector] python scripts/score_clio_baseline.py \
      [--scene lounge-0 ...] [--iou 0.25] [--label-map clip] [--dedup 0.5]
"""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

import networkx as nx
import numpy as np

from r2s3d_core.data.supervisely import load_supervisely_gt
from r2s3d_core.eval.metrics import SceneObject, evaluate, gt_to_scene_object

CLIO_DIR = Path("/data/Clio")
GT_DIR = Path("/home/chris.hsu/repos/Real2USD/humble_ws/evaluations/supervisely")

# clio graphml stem -> (supervisely gt scene id)
SCENE_MAP = {
    "lounge-0": "lounge_0",
    "hallway-1": "hallway_01",
    "smalloffice-0": "smalloffice_00",
    "smalloffice-1": "smalloffice_01",
}

_LABEL_PREFIX = re.compile(r"^(an?\s+image\s+of|a\s+photo\s+of|an?\s+image\s+of\s+an?|an?)\s+", re.I)


def _clean_label(name: str) -> str:
    """'an image of chair' -> 'chair'."""
    return _LABEL_PREFIX.sub("", (name or "").strip()).strip() or (name or "").strip()


def load_clio_objects(graphml: Path) -> list[SceneObject]:
    G = nx.read_graphml(str(graphml))
    preds: list[SceneObject] = []
    for _, d in G.nodes(data=True):
        if d.get("node_type") != "object":
            continue
        center = np.asarray(ast.literal_eval(d["bbox_pos"]), float)
        extents = np.asarray(ast.literal_eval(d["bbox_dim"]), float)
        R = np.asarray(ast.literal_eval(d["bbox_orientation"]), float).reshape(3, 3)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = center
        preds.append(SceneObject(label=_clean_label(d.get("name", "")),
                                 T_world_obj=T, extents=extents))
    return preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", nargs="+", default=list(SCENE_MAP), choices=list(SCENE_MAP))
    ap.add_argument("--iou", type=float, default=0.25)
    ap.add_argument("--label-map", choices=["none", "clip"], default="none")
    ap.add_argument("--dedup", type=float, default=0.0,
                    help="optional same-XY dedup radius (m); Clio should not need it")
    args = ap.parse_args()

    keys = ["f1", "precision", "recall", "recall@0.5", "scan2cad_accuracy",
            "centroid_err_median_m", "rotation_err_median_deg", "scale_err_median",
            "cd_f1", "class_free_recall_1m", "label_accuracy", "duplicate_rate"]
    agg: dict[str, list[float]] = {k: [] for k in keys}

    for scene in args.scene:
        gml = CLIO_DIR / f"{SCENE_MAP[scene]}.graphml"
        gtf = GT_DIR / f"{scene}_voxel_pointcloud.pcd.json"
        if not gml.exists() or not gtf.exists():
            print(f"[{scene}] MISSING clio={gml.exists()} gt={gtf.exists()}; skip")
            continue
        preds = load_clio_objects(gml)
        if args.dedup > 0:
            kept: list[SceneObject] = []
            for p in preds:
                c = p.T_world_obj[:3, 3]
                if all(np.linalg.norm(c[:2] - k.T_world_obj[:2, 3]) > args.dedup for k in kept):
                    kept.append(p)
            preds = kept
        if args.label_map == "clip":
            from r2s3d_core.eval.label_map import remap_pred_labels
            vocab = sorted({g.label for g in load_supervisely_gt(str(gtf))})
            remap_pred_labels(preds, vocab)

        gts = [gt_to_scene_object(g) for g in load_supervisely_gt(str(gtf))]
        m = evaluate(preds, gts, iou_threshold=args.iou, compute_geometry=False)
        a = m.get("aggregate", m)
        print(f"\n=== Clio: {scene} (n_pred={len(preds)} n_gt={len(gts)}) ===")
        for k in keys:
            v = a.get(k, float("nan"))
            print(f"  {k:24s} {v:.3f}" if isinstance(v, (int, float)) else f"  {k:24s} {v}")
            if isinstance(v, (int, float)) and np.isfinite(v):
                agg[k].append(v)

    print("\n=== Clio 4-scene mean ===")
    for k in keys:
        vs = agg[k]
        print(f"  {k:24s} {np.mean(vs):.3f}" if vs else f"  {k:24s} n/a")


if __name__ == "__main__":
    main()
