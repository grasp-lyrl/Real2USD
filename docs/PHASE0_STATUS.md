# Phase 0 status — dataset + harness + naive baseline

Executed 2026-07-07 on the FieldAI desktop. Package: `humble_ws/src_Real2USD/r2s3d_core`.

## Done (verified)

- **`r2s3d_core` uv package** (Python 3.10, ROS-free). `uv sync --extra dev`; 23 tests pass.
- **`SequenceSource` + Replica backend** — verified against `room0_mesh.ply` (depth
  back-projects 100% within room bounds) and GT OBBs verified to 100% mesh-vertex
  containment. Oracle scores perfectly on **all 8 Replica eval scenes** (room0-2,
  office0-4; 323 GT objects total).
- **Metrics** (`eval/metrics.py`, `geometry.py`) — exact OBB-IoU, Hungarian matching,
  centroid/rotation(symmetry)/scale errors, Chamfer + F-score@5/2cm, Scan2CAD accuracy,
  scene P/R/F1, duplicate rate, count ratio. 11 golden tests.
- **Runner + table** — `python -m r2s3d_core.eval.run`, `...eval.table`. run.json per run.
- **Baselines** — `oracle`/`oracle_noisy` (sanity); `sam3d_layout`(+`_icp`) implemented
  with unit-tested placement math + best-view selection.

## Data

Data root: **`~/Data/datasets`** (this workstation has no writable `/data`; override with
`$R2S3D_DATA`). Replica RGB-D + GT semantic downloaded via
`scripts/datasets/download_replica{,_semantic}.sh`.

## Frame conventions — verified empirically (important; save future pain)

- NICE-SLAM Replica **world is already gravity-aligned Z-up** (`gravity_dir ≈ [0,0,-1]`;
  room-mesh vertical extent on Z). No world rotation applied.
- NICE-SLAM `traj.txt` c2w poses are used **directly, no axis flip** — the documented
  `c2w[:3,1:3]*=-1` flip pairs with OpenGL ray dirs; with OpenCV back-projection the raw
  poses are already OpenCV-optical.
- Replica `info_semantic.json` `oriented_bbox`: `abb.center` is in the object's **local**
  frame, `orientation` (xyzw quat + translation) maps local→world. World box center =
  `R @ abb.center + translation`, extents = `abb.sizes`.
- Structural classes (wall/floor/ceiling/window/door/blinds/… ) excluded from object
  metrics (`_STRUCTURAL` in `data/replica.py`).

## Blocked (human-gated) — the actual SAM3D baseline numbers

`sam3d_layout` runs Meta `sam-3d-objects` via the disk-queue worker, needing: the
**gated HF checkpoint** (access request — human step), the external repo clone, and its
`sam3d-objects` conda env. Code is ready and input-hash-cached; it raises a loud,
actionable error until the worker can run. Fill the `sam3d_layout` / `sam3d_layout_icp`
table rows once access lands — that is the paper's motivating layout-error experiment.

## Not committed yet (per request). Suggested first commit: the `r2s3d_core` package +
the two download scripts + this doc.
