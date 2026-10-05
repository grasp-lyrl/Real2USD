"""Is the real-robot placement error a single fixable extrinsic, or per-frame drift?

For each real scene, match predicted object centroids to GT and fit ONE global
correction (translation-only, then full SE(3) via Umeyama). If the residual
centroid error collapses under a single transform, the ~0.5 m offset is a constant
extrinsic miscalibration (correctable once); if it stays high, it is drift.

  uv run --extra rosbag python scripts/figs/calib_diag.py
"""
from __future__ import annotations
import json, glob
import numpy as np
from scipy.spatial import cKDTree
from r2s3d_core.data.registry import make_source

SCENES = {"lounge-0": "lounge0", "hallway-1": "hallway1",
          "smalloffice-0": "smalloffice0", "smalloffice-1": "smalloffice1"}
MATCH_R = 1.5  # generous match radius (m) to catch offset objects


def umeyama(src, dst):
    """Least-squares rigid SE(3) mapping src->dst (no scale)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    H = S.T @ D
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1; R = Vt.T @ U.T
    t = mu_d - R @ mu_s
    return R, t


def main():
    all_before, all_tr, all_se3 = [], [], []
    for scene, key in SCENES.items():
        run = None
        for pat in (f"results/phase0_{key}_rs_scaleicp", f"results/phase0_{key}_rs_icp"):
            g = glob.glob(pat + "/*/scene_graph.json")
            if g:
                run = g[0]; break
        if not run:
            print(f"[{scene}] no run"); continue
        preds = np.array([np.asarray(o["T_world_obj"], float)[:3, 3]
                          for o in json.loads(open(run).read())["objects"]])
        gts = np.array([np.asarray(g.T_world_obj, float)[:3, 3]
                        for g in (make_source("realsense", scene, stride=2).gt() or [])])
        if len(preds) == 0 or len(gts) == 0:
            print(f"[{scene}] empty"); continue
        # match pred->nearest GT within radius, one GT each (greedy)
        tree = cKDTree(gts); d, idx = tree.query(preds)
        used, P, G = set(), [], []
        for i in np.argsort(d):
            if d[i] <= MATCH_R and idx[i] not in used:
                used.add(idx[i]); P.append(preds[i]); G.append(gts[idx[i]])
        P, G = np.array(P), np.array(G)
        if len(P) < 3:
            print(f"[{scene}] only {len(P)} matches"); continue
        before = np.linalg.norm(P - G, axis=1)
        t_only = np.linalg.norm((P + (G - P).mean(0)) - G, axis=1)
        R, t = umeyama(P, G)
        se3 = np.linalg.norm((P @ R.T + t) - G, axis=1)
        all_before += before.tolist(); all_tr += t_only.tolist(); all_se3 += se3.tolist()
        ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        print(f"[{scene}] n={len(P):2d}  centroid median  before={np.median(before):.3f}  "
              f"+trans={np.median(t_only):.3f}  +SE3={np.median(se3):.3f}  "
              f"(fit |t|={np.linalg.norm(t):.2f}m, R={ang:.1f}deg)")
    b, tr, s = np.array(all_before), np.array(all_tr), np.array(all_se3)
    print(f"\nPOOLED (n={len(b)}): median centroid  before={np.median(b):.3f}  "
          f"+trans={np.median(tr):.3f}  +SE3={np.median(s):.3f}")
    print("=> if +trans/+SE3 collapses the error, it's a constant extrinsic (fixable);"
          " if it stays high, it's per-frame drift.")


if __name__ == "__main__":
    main()
