# Depth-Free ("no depth at all") experiment — status

**Question:** can we build a metric asset map with *no depth anywhere* — no depth to
SAM3D (RGB-only generation), no depth for ICP, no depth for the scale metric — using
only RGB, masks, and (metric) camera poses? **Answer so far: yes for compact/volumetric
objects in simulation; no for flat/thin objects; and no on real hardware** (odometry
drift breaks the multi-view triangulation — centroid error 0.3–1.4 m, 3D-IoU ~0, §C).
Prototype-validated in sim; NOT integrated into the pipeline; currently a
Discussion/Future-Work candidate, not a paper table row. Owner note: this is the "what does depth buy us" probe from the 2026-07-24
session (see `STATUS.md`).

## What depth does in the deployed pipeline (three roles)
1. **SAM3D input** — depth pointmap fed to generation (`--use-depth`; default OFF, but the
   paper's sim meshes were generated with it ON).
2. **Pose** — ICP of the mesh against the fused depth cloud $P_k$.
3. **Scale metric** — median masked depth converts the mask silhouette to metric size
   (single-view extent), or fixes the object's distance for multi-view render-and-compare.

The depth-free recipe replaces these with: RGB-only SAM3D; **position by triangulating
mask-centroid rays across views** (metric from the camera baselines, i.e. odometry, not a
depth sensor); **scale by multi-view silhouette**; orientation from SAM3D. Multi-view is
mandatory (a monocular silhouette is pose- and scale-ambiguous).

## Experiments run (sim, ProcTHOR s200)

### A. Depth-free PLACEMENT, depth-D-generated meshes
`scripts/figs/render_compare_depthfree.py` (default SG = `asset_layout_gt_s200`, depth-D
meshes). Triangulated centroid + silhouette scale, no depth in placement:

| object | tri-centroid err | 3D-IoU (→scale) | scale_err |
|---|---|---|---|
| fridge | 0.054 m | 0.54 → **0.87** | 0.61 → 0.04 |
| painting | 0.047 m | 0.04 → **0.80** | 0.23 → 0.14 |
| bed | 0.295 m | 0.36 → 0.68 | 0.33 → 0.21 |
| painting | 0.140 m | 0.00 → 0.49 | 0.67 → 0.15 |
| diningtable | 0.30–0.35 m | 0.11–0.51 | mixed |

Compact objects ≈ depth-full (fridge 0.87 vs depth-full 0.90); large flat tables degrade
(triangulation ~0.30 m + silhouette thin-axis ambiguity). Median 3D-IoU ≈ 0.50 vs
depth-full ≈ 0.62.

### B. FULLY depth-free (RGB-only generation + depth-free placement)
RGB-only SAM3D regen: drained `results/rgbonly_s200_queue` with the worker and **no
`--use-depth`** (72 meshes, 0 fail). Placed via `object_track_icp` →
`sim run rgbonly_placed`; then `render_compare_depthfree.py` with
`RC_DF_SG=<rgbonly_placed>/200/scene_graph.json`:

| object | 3D-IoU (→scale) | scale_err | type |
|---|---|---|---|
| toilet | 0.40 → 0.71 | 0.30 → 0.13 | volumetric ✓ |
| armchair | 0.23 → 0.72 | 0.40 → 0.10 | volumetric ✓ |
| toaster | 0.45 → 0.69 | 0.34 → 0.17 | volumetric ✓ |
| laundryhamper | 0.23 → 0.60 | 0.46 → 0.24 | volumetric ✓ |
| painting 54 | 0.00 → 0.03 | 0.60 → 0.56 | flat ✗ |
| painting 53 | 0.03 → 0.05 | 0.63 → 0.16 | flat ✗ |
| television | 0.15 → 0.12 | 3.95 → 4.97 | flat ✗ (scale blew up) |

**Bimodal.** The same painting was **0.80** with the depth-D mesh but **0.03** with the
RGB-only mesh → RGB-only SAM3D generates a much worse *shape* for flat/thin objects (a
planar object is hard without a depth pointmap), and the silhouette also can't constrain
the thin axis at placement.

### C. Depth-free on REAL data (the decisive contrast)
`scripts/figs/render_compare_depthfree_real.py lounge-0` (realsense source, detection
masks, drifting odometry poses, GT Supervisely boxes), 8 objects:

| quantity | sim (clean poses) | REAL (odometry drift) |
|---|---|---|
| triangulation-centroid err | 0.05–0.35 m | **0.29–1.38 m** (median ~0.82 m) |
| 3D-IoU | 0.5–0.9 (compact) | **~0.00–0.14** (collapses) |

Depth-free **fails on real**: the drifting camera baselines mean the mask-centroid rays
never intersect at the true location, so triangulation is off by up to 1.4 m and 3D-IoU
goes to ~0. For reference, the *depth-based* pipeline on the same real data places to
~0.22 m centroid and works (`tab:real`). This is the sharpest evidence that depth's real
value is enabling **single-view** placement (each object from one frame, immune to
cross-frame drift), which multi-view triangulation cannot substitute for when odometry
drifts.

## Takeaway (the honest framing for the paper)
**Depth earns its keep in two distinct ways:** (1) it enables *single-view, drift-robust*
placement on real hardware (the real rig's odometry drift breaks multi-view methods —
see the calibration finding in `STATUS.md`); (2) it gives SAM3D the geometry to generate
*flat/thin* objects correctly. For compact, volumetric objects with clean poses, neither
is needed — multi-view silhouettes + camera baselines suffice, and vision-only asset
mapping works.

## Artifacts
- Scripts: `scripts/figs/render_compare_depthfree.py` (triangulation + silhouette; env
  `RC_DF_SG` selects the scene graph), `scripts/figs/render_compare_sim.py`,
  `render_compare_batch.py`; RGB-only drain: `scratchpad .../drain_rgbonly.sh` (worker,
  no `--use-depth`).
- RGB-only queue: `results/rgbonly_s200_queue` (72 meshes). Placement:
  `sim run rgbonly_placed`.
- Figures (aligned, same camera): `results/paper/_figs/scene_s200_gen_depthD_top.png`
  vs `scene_s200_gen_rgbonly_top.png` — depth-D meshes are full/solid; RGB-only come out
  smaller/degenerate (esp. flat objects). Good visual for the Discussion.

## Status / what's NOT done
- Prototype only: per-object 3D-IoU on hand-picked well-observed objects at s200. **Not**
  a full-scene, harness-scored (IoU-F1/rec@.5/S2C) val-10 run.
- **Not** integrated as a pipeline registration mode; the depth-free placement is a
  standalone script that bypasses the tracker's depth path.
- **Not** a Table 1 row: a fair full-scene number would be dragged down by flat objects
  (common in scenes) and would confuse the C1 registration ablation. Belongs in
  Discussion/Future-Work.

## If pursued (next steps)
1. Integrate depth-free placement (triangulation + silhouette Sim(3)) as a real pipeline
   mode; run val-10 harness-scored for an honest aggregate.
2. RGB-only generation across val-10 (not just s200) to quantify the flat-object gap.
3. A flat-object fix: constrain/regularize the thin axis (or detect planarity) so
   silhouette scale doesn't explode (the TV 3.95→4.97 case).
