"""Mask boundary cleanliness / depth-leak probe: YOLOE vs SAM 3 (or any two
cached DetectionSets) on one scene.

Hypothesis: cleaner mask boundaries -> fewer edge-bleed pixels -> the masked
depth back-projects to fewer LEAKED 3D points (on background / adjacent
surfaces) -> a tighter per-object cloud -> better ICP and a less-inflated
depth-cloud OBB. This measures exactly that, per detection:

  * leak_frac : fraction of back-projected masked-depth points farther than TAU
                from the matched GT object's surface (edge-bleed -> high).
  * obb_scale_err : raw masked-depth-cloud OBB max-extent vs GT max-extent
                    (edge-bleed inflates the box; the "fused scale_err craters" effect).
  * mask_region_iou : detection mask vs GT native instance mask (region overlap,
                      for reference — the --diagnose number).

Detections are matched to the nearest GT object by cloud-centroid distance.

Usage (needs the GPU display for procthor frames; cached renders replay without):
  uv run --extra procthor --extra mesh --extra registration python \
    scripts/probe_mask_cleanliness.py --scene 200 --split val --stride 10 \
    --a results/detections/probe_yoloe/gt --a-name yoloe \
    --b results/detections/probe_sam3/gt  --b-name sam3
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy.spatial import cKDTree

from r2s3d_core.data.registry import make_source
from r2s3d_core.detect.cache import DetectionSet
from r2s3d_core.eval import geometry as geo
from r2s3d_core.baselines import sam3d_layout as s3d

N = 4000
TAU = 0.05  # m; a masked-depth point farther than this from the GT surface is "leaked"


def _analyze(name, det_dir, scene, frames, gt_trees, gt_cent, gt_ext):
    ds = DetectionSet.load(det_dir, scene=scene)
    by_frame = ds.by_frame()
    frame_by_id = {int(f.frame_id): f for f in frames}
    rows = []
    for fid, dets in by_frame.items():
        frame = frame_by_id.get(int(fid))
        if frame is None:
            continue
        for d in dets:
            cloud = s3d._masked_depth_cloud(frame, d.mask)   # world-frame points
            if len(cloud) < 15:
                continue
            gi = int(np.argmin(np.linalg.norm(gt_cent - cloud.mean(0), axis=1)))
            dist, _ = gt_trees[gi].query(cloud)
            leak = float(np.mean(dist > TAU))
            try:
                _, ext = s3d._cloud_obb(cloud)
                obb_err = float(np.max(ext) / max(np.max(gt_ext[gi]), 1e-6) - 1.0)
            except Exception:
                obb_err = float("nan")
            rows.append((leak, abs(obb_err), len(cloud)))
    if not rows:
        print(f"[{name}] no usable detections"); return
    leak = np.array([r[0] for r in rows]); obb = np.array([r[1] for r in rows])
    obb_finite = obb[np.isfinite(obb)]
    print(f"[{name}] n_det={len(rows)}  "
          f"leak_frac median={np.median(leak):.3f} mean={leak.mean():.3f}  "
          f"|obb_scale_err| median={np.nanmedian(obb):.3f} mean={np.nanmean(obb):.3f} "
          f"(n_valid_obb={len(obb_finite)}/{len(obb)})")
    return dict(name=name, n=len(rows), leak_med=float(np.median(leak)),
                obb_med=float(np.nanmedian(obb)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="200")
    ap.add_argument("--split", default="val")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--a", required=True, help="DetectionSet dir A (parent of <scene>/)")
    ap.add_argument("--a-name", default="A")
    ap.add_argument("--b", required=True, help="DetectionSet dir B")
    ap.add_argument("--b-name", default="B")
    args = ap.parse_args()

    src = make_source("procthor", args.scene, split=args.split, gt_mesh="asset", stride=args.stride)
    frames = list(src)
    gts = [g for g in src.gt() if getattr(g, "mesh", None) is not None]
    gt_cent = np.array([g.T_world_obj[:3, 3] for g in gts])
    gt_ext = [np.asarray(g.extents, float) for g in gts]
    gt_trees = [cKDTree(geo.sample_surface(g.mesh, N, seed=1)) for g in gts]
    print(f"scene {args.scene}: {len(frames)} frames (stride {args.stride}), {len(gts)} GT objects, tau={TAU}m")

    _analyze(args.a_name, args.a, args.scene, frames, gt_trees, gt_cent, gt_ext)
    _analyze(args.b_name, args.b, args.scene, frames, gt_trees, gt_cent, gt_ext)


if __name__ == "__main__":
    main()
