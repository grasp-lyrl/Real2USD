"""CLEAN shape-completion experiment (isolates generation's shape value from placement).

The end-to-end coverage diagnostic (gen_coverage_diag.py) was confounded: at low coverage the
asset is mis-registered (placement fails with few points) and the fused cloud is contaminated by
supporting surfaces. This script removes those confounds to answer the pure question:

  "Given the object's true pose (ORACLE), does the generated shape reconstruct the surface the
   robot never saw, better than the observed cluster can?"

Per matched LARGE GT object (real THOR GT mesh):
  * ORACLE-place the asset: map its OBB-local shape into the GT OBB (correct pose+scale), choosing
    the best of the 24 cube rotations by fit to GT — so only the asset's intrinsic SHAPE is used,
    not its (coverage-limited) registration.
  * CLEAN the cluster: keep only observed points inside the GT OBB (inflated) — a fair "clustering
    method associates points to the object", removing edge-bleed from countertops/floors.
  * coverage c = frac GT surface within tau of the clean cluster.
  * asset_unobs / cluster_unobs = mean NN distance from the UNOBSERVED GT surface to the
    oracle-placed asset / to the clean cluster. The money metric: how well each fills the unseen part.

Usage:
  uv run --extra procthor --extra mesh --extra registration --extra detector \
    python scripts/gen_shape_completion.py --scene 200 \
    --asset-run results/procthor_procthor_object_track_icp_s200_val --tau 0.05 --min-extent 0.5
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from r2s3d_core.baselines.sam3d_layout import _fast_obb
from r2s3d_core.data.registry import make_source
from r2s3d_core.detect.cache import DetectionSet
from r2s3d_core.eval import geometry as geo
from r2s3d_core.tracks import TrackState, run_tracker

N = 4000


def _cube_rotations():
    """The 24 proper rotations mapping a box's axes to a box's axes (signed permutations, det +1)."""
    Rs = []
    for perm in itertools.permutations(range(3)):
        P = np.eye(3)[list(perm)]
        for s in itertools.product((1, -1), repeat=3):
            R = P * np.array(s)
            if abs(np.linalg.det(R) - 1.0) < 1e-6:
                Rs.append(R)
    return Rs


_ROT24 = _cube_rotations()


def _to_obb_local(pts, R, c, ext):
    """World points -> OBB-local unit-box coords (axes = OBB axes, scaled by extent)."""
    return ((pts - c) @ R) / np.maximum(ext, 1e-6)


def oracle_place_asset(asset_mesh, gt_R, gt_c, gt_ext):
    """Sample the asset, express its shape in its own OBB unit box, then map into the GT OBB
    (oracle pose+scale) with the cube rotation that best fits GT. Returns asset points in world."""
    Ta, ea = _fast_obb(asset_mesh)
    ap = geo.sample_surface(asset_mesh, N, seed=7)
    if len(ap) == 0:
        return None
    a_unit = _to_obb_local(ap, Ta[:3, :3], Ta[:3, 3], ea)          # asset shape in unit box
    return a_unit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="200")
    ap.add_argument("--split", default="val")
    ap.add_argument("--asset-run", required=True)
    ap.add_argument("--detections", default="results/detections/procthor/gt")
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--min-extent", type=float, default=0.5, help="keep GT objs with max extent >= this (m)")
    ap.add_argument("--obb-inflate", type=float, default=1.15, help="GT OBB inflation for cluster cleaning")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    src = make_source("procthor", args.scene, split=args.split, gt_mesh="asset", stride=1)
    frames = list(src)
    gts = [g for g in src.gt() if getattr(g, "mesh", None) is not None]
    ds = DetectionSet.load(args.detections, scene=args.scene)
    tracks = run_tracker(frames, ds.by_frame(), {"reid": True, "late_merge": True})
    mature = [t for t in tracks if t.state == TrackState.MATURE and len(t.fused_cloud) >= 4]

    # asset meshes by track id
    import trimesh
    sg = json.load(open(next(Path(args.asset_run).glob("*/scene_graph.json"))))
    queue = Path(sg.get("sam3d_queue") or (Path(args.asset_run) / "sam3d_queue"))
    assets = {}
    for o in sg["objects"]:
        mp = queue / (o.get("mesh") or "")
        if o.get("mesh") and mp.exists():
            m = trimesh.load(str(mp), force="mesh")
            m.apply_transform(np.asarray(o["T_world_mesh"], float))
            assets[int(o["id"])] = m

    gt_c = np.array([g.T_world_obj[:3, 3] for g in gts])
    gt_pts = {i: geo.sample_surface(g.mesh, N, seed=1000 + i) for i, g in enumerate(gts)}
    used, rows = set(), []
    for t in sorted(mature, key=lambda t: -len(t.fused_cloud)):
        if t.track_id not in assets:
            continue
        d = np.linalg.norm(gt_c[:, :2] - t.centroid[:2], axis=1)
        gi = next((int(j) for j in np.argsort(d) if j not in used and d[j] <= 1.0), None)
        if gi is None:
            continue
        g = gts[gi]
        gR, gc, gext = g.T_world_obj[:3, :3], g.T_world_obj[:3, 3], np.asarray(g.extents, float)
        if float(np.max(gext)) < args.min_extent:
            continue  # large objects only
        used.add(gi)

        gpts = gt_pts[gi]
        obs = np.asarray(t.fused_cloud, float)
        # clean cluster: keep observed points inside the (inflated) GT OBB
        loc = _to_obb_local(obs, gR, gc, gext)
        obs_clip = obs[np.all(np.abs(loc) <= 0.5 * args.obb_inflate, axis=1)]
        if len(obs_clip) < 4:
            continue

        # oracle-place the asset shape into the GT OBB, best of 24 rotations vs GT (in unit box)
        a_unit = oracle_place_asset(assets[t.track_id], gR, gc, gext)
        if a_unit is None:
            continue
        g_unit = _to_obb_local(gpts, gR, gc, gext)
        best_R, best_cd = None, np.inf
        for R in _ROT24:
            cd = geo.chamfer_and_fscore(a_unit @ R.T, g_unit)["chamfer_mean"]
            if cd < best_cd:
                best_cd, best_R = cd, R
        asset_world = (a_unit @ best_R.T) * gext @ gR.T + gc   # oracle-placed asset in world

        # coverage from the CLEAN cluster
        d_gt_obs, _ = cKDTree(obs_clip).query(gpts)
        observed = d_gt_obs <= args.tau
        coverage = float(observed.mean())

        unobs = gpts[~observed]
        if len(unobs) == 0:
            continue
        asset_unobs = float(cKDTree(asset_world).query(unobs)[0].mean())
        cluster_unobs = float(d_gt_obs[~observed].mean())
        rows.append({
            "track_id": int(t.track_id), "gt_label": g.label,
            "max_extent": round(float(np.max(gext)), 3),
            "coverage": round(coverage, 4), "unobs_frac": round(1 - coverage, 4),
            "asset_shape_cd": round(float(best_cd), 4),   # unit-box shape match to GT
            "asset_unobs_recon": round(asset_unobs, 4),
            "cluster_unobs_recon": round(cluster_unobs, 4),
            "asset_better": asset_unobs < cluster_unobs,
            "n_clip": len(obs_clip),
        })

    if not rows:
        raise SystemExit("no large matched objects — lower --min-extent?")
    out = Path(args.out or f"results/paper/_tables/shape_completion_s{args.scene}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    cov = np.array([r["coverage"] for r in rows])
    print(f"\nmatched {len(rows)} LARGE objects (max_extent >= {args.min_extent} m) | "
          f"coverage med {np.median(cov):.2f}")
    print("\n=== UNOBSERVED-surface reconstruction (m): oracle-placed asset vs clean cluster ===")
    print(f"  {'bin':14s}{'n':>4s}{'unobs_frac':>12s}{'asset_recon':>13s}{'cluster_recon':>15s}{'asset_wins':>12s}")
    for lo, hi, name in [(0, .3, "low <0.3"), (.3, .6, "mid .3-.6"), (.6, 1.01, "high >0.6")]:
        sel = [r for r in rows if lo <= r["coverage"] < hi]
        if not sel:
            print(f"  {name:14s}{0:>4d}"); continue
        uf = np.mean([r["unobs_frac"] for r in sel])
        a = np.mean([r["asset_unobs_recon"] for r in sel])
        c = np.mean([r["cluster_unobs_recon"] for r in sel])
        wins = sum(r["asset_better"] for r in sel)
        print(f"  {name:14s}{len(sel):>4d}{uf:>12.3f}{a:>13.3f}{c:>15.3f}{f'{wins}/{len(sel)}':>12s}")
    aw = sum(r["asset_better"] for r in rows)
    print(f"\nasset reconstructs unobserved surface better in {aw}/{len(rows)} objects")
    # most-occluded large object -> qualitative figure candidate
    occ = min(rows, key=lambda r: r["coverage"])
    print(f"qualitative panel candidate (most occluded large obj): track {occ['track_id']} "
          f"'{occ['gt_label']}' coverage {occ['coverage']:.2f} asset {occ['asset_unobs_recon']} "
          f"cluster {occ['cluster_unobs_recon']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
