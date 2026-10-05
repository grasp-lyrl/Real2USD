"""Generation-value experiment: geometry accuracy vs OBSERVATION COVERAGE (partial-view argument).

The claim: generation's worth is concentrated where the robot observes little of an object (chairs
under tables, occlusion, grazing views) — the regime real deployment lives in. On well-observed
objects the observed cluster is already fine (why the average delta is modest); on partially-observed
objects the cluster is fundamentally incomplete while the SAM3D asset's shape prior completes it.

Per matched GT object (ProcTHOR val, real THOR GT meshes) we compute:
  coverage c   = fraction of GT surface within `tau` of the track's fused observed cloud
  cluster_cd   = symmetric Chamfer(observed cloud, GT)         [the clustering payload]
  asset_cd     = symmetric Chamfer(posed SAM3D mesh, GT)       [the generation payload]
  unobs_frac   = 1 - c
  asset_unobs  = mean NN dist from the UNOBSERVED GT surface to the asset mesh  (the money metric:
                 how well generation reconstructs what was never seen; the cluster has ~nothing there)

Outputs per-object CSV + coverage-binned means for BOTH stories (Chamfer-vs-coverage and
unobserved-region reconstruction) so we can pick the stronger one.

Usage:
  uv run --extra procthor --extra mesh --extra registration --extra detector \
      python scripts/gen_coverage_diag.py --scene 200 \
      --asset-run results/procthor_procthor_object_track_icp_s200_val --tau 0.05
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from r2s3d_core.data.registry import make_source
from r2s3d_core.detect.cache import DetectionSet
from r2s3d_core.eval import geometry as geo
from r2s3d_core.tracks import TrackState, run_tracker

N = 3000  # surface samples per object


def _posed_assets(asset_run: Path) -> dict[int, object]:
    """{track_id: world-posed SAM3D trimesh} from the asset run's scene_graph.json."""
    import trimesh
    sgs = list(asset_run.glob("*/scene_graph.json"))
    if not sgs:
        raise SystemExit(f"no scene_graph.json under {asset_run}")
    sg = json.load(open(sgs[0]))
    queue = Path(sg.get("sam3d_queue") or (asset_run / "sam3d_queue"))
    out: dict[int, object] = {}
    for o in sg["objects"]:
        if not o.get("mesh"):
            continue
        mp = queue / o["mesh"]
        if not mp.exists():
            continue
        m = trimesh.load(str(mp), force="mesh")
        m.apply_transform(np.asarray(o["T_world_mesh"], float))
        out[int(o["id"])] = m
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="200")
    ap.add_argument("--split", default="val")
    ap.add_argument("--asset-run", required=True)
    ap.add_argument("--detections", default="results/detections/procthor/gt")
    ap.add_argument("--tau", type=float, default=0.05, help="coverage/observed threshold (m)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    src = make_source("procthor", args.scene, split=args.split, gt_mesh="asset", stride=1)
    frames = list(src)
    gts = [g for g in src.gt() if getattr(g, "mesh", None) is not None]
    ds = DetectionSet.load(args.detections, scene=args.scene)
    tracks = run_tracker(frames, ds.by_frame(), {"reid": True, "late_merge": True})
    mature = [t for t in tracks if t.state == TrackState.MATURE and len(t.fused_cloud) >= 4]
    assets = _posed_assets(Path(args.asset_run))
    print(f"scene {args.scene}: {len(gts)} GT, {len(mature)} mature tracks, {len(assets)} asset meshes")

    # greedy match track -> GT by 2D-centroid distance <= 1 m
    gt_pts = {i: geo.sample_surface(g.mesh, N, seed=1000 + i) for i, g in enumerate(gts)}
    gt_c = np.array([g.T_world_obj[:3, 3] for g in gts])
    used = set()
    rows = []
    for t in sorted(mature, key=lambda t: -len(t.fused_cloud)):
        if t.track_id not in assets:
            continue  # need the asset mesh to compare
        d = np.linalg.norm(gt_c[:, :2] - t.centroid[:2], axis=1)
        order = np.argsort(d)
        gi = next((int(j) for j in order if j not in used and d[j] <= 1.0), None)
        if gi is None:
            continue
        used.add(gi)

        g = gt_pts[gi]
        obs = np.asarray(t.fused_cloud, float)
        asset = geo.sample_surface(assets[t.track_id], N, seed=t.track_id)
        if len(asset) == 0:
            continue

        tree_obs = cKDTree(obs)
        d_gt_obs, _ = tree_obs.query(g)          # each GT pt -> nearest observed pt
        observed = d_gt_obs <= args.tau
        coverage = float(observed.mean())

        cluster_cd = geo.chamfer_and_fscore(obs, g)["chamfer_mean"]
        asset_cd = geo.chamfer_and_fscore(asset, g)["chamfer_mean"]

        unobs = g[~observed]
        if len(unobs):
            asset_unobs = float(cKDTree(asset).query(unobs)[0].mean())
            cluster_unobs = float(d_gt_obs[~observed].mean())  # >= tau by construction
        else:
            asset_unobs = cluster_unobs = float("nan")

        rows.append({
            "track_id": int(t.track_id), "gt_label": gts[gi].label,
            "coverage": round(coverage, 4), "unobs_frac": round(1 - coverage, 4),
            "cluster_cd": round(cluster_cd, 4), "asset_cd": round(asset_cd, 4),
            "asset_unobs_recon": round(asset_unobs, 4) if np.isfinite(asset_unobs) else "",
            "cluster_unobs_recon": round(cluster_unobs, 4) if np.isfinite(cluster_unobs) else "",
            "n_obs_pts": len(obs),
        })

    if not rows:
        raise SystemExit("no matched objects with both cluster and asset — check asset-run/detections")

    out = Path(args.out or f"results/paper/_tables/coverage_diag_s{args.scene}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    cov = np.array([r["coverage"] for r in rows])
    bins = [(0.0, 0.3, "low <0.3"), (0.3, 0.6, "mid 0.3-0.6"), (0.6, 1.01, "high >0.6")]
    print(f"\nmatched {len(rows)} objects | coverage: min {cov.min():.2f} med {np.median(cov):.2f} max {cov.max():.2f}")
    print("\n=== STORY 1: Chamfer (m) vs coverage — asset should stay flat, cluster explode at low cov ===")
    print(f"  {'bin':14s}{'n':>4s}{'cluster_cd':>12s}{'asset_cd':>10s}{'gap(clu-ast)':>14s}")
    for lo, hi, name in bins:
        sel = [r for r in rows if lo <= r["coverage"] < hi]
        if not sel:
            print(f"  {name:14s}{0:>4d}"); continue
        c = np.mean([r["cluster_cd"] for r in sel]); a = np.mean([r["asset_cd"] for r in sel])
        print(f"  {name:14s}{len(sel):>4d}{c:>12.3f}{a:>10.3f}{c-a:>14.3f}")
    print("\n=== STORY 2: reconstruction of the UNOBSERVED surface (m) — asset fills it, cluster can't ===")
    print(f"  {'bin':14s}{'n':>4s}{'unobs_frac':>12s}{'asset_recon':>13s}{'cluster_recon':>15s}")
    for lo, hi, name in bins:
        sel = [r for r in rows if lo <= r["coverage"] < hi and r["asset_unobs_recon"] != ""]
        if not sel:
            print(f"  {name:14s}{0:>4d}"); continue
        uf = np.mean([r["unobs_frac"] for r in sel])
        ar = np.mean([r["asset_unobs_recon"] for r in sel])
        cr = np.mean([r["cluster_unobs_recon"] for r in sel])
        print(f"  {name:14s}{len(sel):>4d}{uf:>12.3f}{ar:>13.3f}{cr:>15.3f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
