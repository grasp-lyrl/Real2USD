"""Paired per-object rotation diagnostic for the real-robot table.

The real table's per-scene rotation *median* is taken over the IoU>=0.25 matched
set, which changes with each variant (1-6 matches/scene) -- so it does NOT compare
the same objects across layout / +ICP / +scale+ICP. This pairs each track to a GT
ONCE (nearest centroid, greedy, <=1 m) and reports the rotation error of the SAME
objects under each variant, with a Wilcoxon signed-rank test (as done for scale).

  uv run --extra rosbag --extra registration --extra mesh \
    python scripts/figs/paired_rotation_real.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import wilcoxon

from r2s3d_core.data.registry import make_source
from r2s3d_core.eval import geometry as geo
from r2s3d_core.eval.metrics import symmetry_for_label

ROOT = Path(__file__).resolve().parents[2]
SCENES = {"lounge-0": "lounge0", "hallway-1": "hallway1",
          "smalloffice-0": "smalloffice0", "smalloffice-1": "smalloffice1"}
VARIANTS = ["layout", "icp", "scaleicp"]
TAU = 1.0  # association radius (m), matches cd-F1


def load_preds(scene_key, variant):
    """{job_id: (R(3x3), extents(3), label, centroid(3))} from a run's scene_graph."""
    d = ROOT / f"results/phase0_{scene_key}_rs_{variant}"
    sg = next(d.glob("*/scene_graph.json"), None)
    if sg is None:
        return {}
    objs = json.loads(sg.read_text())["objects"]
    out = {}
    for o in objs:
        T = np.asarray(o["T_world_obj"], float)
        out[o["job_id"]] = (T[:3, :3], np.asarray(o["extents"], float),
                            o.get("label", ""), T[:3, 3])
    return out


def gt_boxes(scene):
    src = make_source("realsense", scene, stride=2)
    gts = []
    for g in (src.gt() or []):
        T = np.asarray(g.T_world_obj, float)
        gts.append((T[:3, :3], np.asarray(g.extents, float), g.label, T[:3, 3]))
    return gts


def main():
    # per-object rotation error under each variant, paired by (scene, job_id->gt)
    rot = {v: [] for v in VARIANTS}
    n_pairs = 0
    for scene, key in SCENES.items():
        gts = gt_boxes(scene)
        if not gts:
            print(f"[{scene}] no GT; skip"); continue
        preds = {v: load_preds(key, v) for v in VARIANTS}
        if not preds["layout"] or not preds["icp"]:
            print(f"[{scene}] missing layout/icp run; skip"); continue
        gt_cent = np.array([g[3] for g in gts])
        tree = cKDTree(gt_cent)
        used_gt = set()
        # associate each track to a GT ONCE, by its layout centroid (greedy, nearest first)
        cand = []
        for jid, (_, _, _, c) in preds["layout"].items():
            dist, gi = tree.query(c)
            if dist <= TAU:
                cand.append((float(dist), jid, int(gi)))
        cand.sort()
        for dist, jid, gi in cand:
            if gi in used_gt:
                continue
            # require the track to exist in all variants we compare
            if not all(jid in preds[v] for v in VARIANTS):
                continue
            used_gt.add(gi)
            Rg, eg, lg, _ = gts[gi]
            sym = symmetry_for_label(lg)
            for v in VARIANTS:
                Rp, ep, _, _ = preds[v][jid]
                r_err, _ = geo.box_pose_error(Rp, ep, Rg, eg, sym)
                rot[v].append(float(r_err))
            n_pairs += 1
        print(f"[{scene}] {sum(1 for _,j,g in cand)} candidates, paired so far {n_pairs}")

    print(f"\n=== paired rotation over {n_pairs} objects (4 real scenes) ===")
    for v in VARIANTS:
        a = np.array(rot[v])
        print(f"  {v:9} median={np.median(a):5.1f} deg   mean={a.mean():5.1f}")
    lay, icp = np.array(rot["layout"]), np.array(rot["icp"])
    d = icp - lay
    print(f"\n  layout -> +ICP paired delta: median {np.median(d):+.2f} deg")
    print(f"  improved (icp<layout): {int((d < 0).sum())}/{n_pairs}   "
          f"worsened: {int((d > 0).sum())}   tie: {int((d == 0).sum())}")
    if n_pairs >= 5 and np.any(d != 0):
        try:
            w = wilcoxon(icp, lay)
            print(f"  Wilcoxon (icp vs layout): p={w.pvalue:.4f}")
        except Exception as e:
            print(f"  Wilcoxon failed: {e}")


if __name__ == "__main__":
    main()
