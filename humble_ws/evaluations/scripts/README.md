# Evaluation scripts

Entrypoints used by the evaluation pipeline and bash scripts.

## Scripts

- **run_eval.py** – Single-run evaluation: load predictions (scene_graph or CLIO), load GT, match with greedy IoU, compute metrics and diagnostics, write `results_root/by_run/<scene>_<run_id>_<method_tag>_<ts>_<hash>.json`. Use `--prediction-json` or `--prediction-path`, `--gt-json`, `--run-dir`, `--results-root`, `--method-tag`, etc.
- **plot_overlays.py** – Plot bounding boxes overlaid with GT, and (if point cloud exists) pred+GT on point cloud. Boxes are drawn as **OBB footprints** (rotated rectangles). Requires `--by-run-json`; optional `--run-dir` for PC, `--out-dir` for PNGs.

## OBB orientation and matching

When `matching.iou_mode` is `"obb"` (see `eval_config.json`), boxes are stored with the axis most aligned with world Z as the **third**; `obb_dimensions` = [dx, dy, dz] with dx, dy = horizontal footprint, dz = height. Matching uses 3D OBB IoU; overlay plots draw the XY footprint (dx × dy, rotated by the in-plane yaw).

## Axis-aligned frame (frame_normalization)

Evaluation can apply a **single global yaw rotation** (from `eval_config.json` → `frame_normalization: gt_global_yaw`) so that the dominant GT orientation aligns with the axes. The same rotation is applied to both predictions and GT when loading; the `by_run` JSON therefore contains boxes in this normalized frame.

- **Bbox vs GT plots**: Preds and gts in `instances` are already in this frame, so the plot is axis-aligned by default.
- **Point cloud**: If the pipeline saved point clouds in the run directory (`run_dir/diagnostics/pointclouds/lidar/*.npy` or `realsense/*.npy`), they are in the original odom frame. The overlay script applies the **same** yaw rotation to the point cloud before plotting, so the PC is aligned with the boxes. This is mathematically correct: applying the same rigid transform to all data preserves relative geometry (IoU, distances). You do not need the experiment to be axis-aligned at capture time; alignment is done at eval/visualization time.

## Running from repo root

From the repository root (or anywhere), run with the evaluations directory on `PYTHONPATH` or from inside `evaluations/`:

```bash
cd humble_ws/evaluations
python3 scripts/run_eval.py --prediction-json run_20260222_203756_lounge_lidar/scene_graph.json --gt-json supervisely/lounge-0_voxel_pointcloud.pcd.json --run-dir run_20260222_203756_lounge_lidar --scene lounge --run-id run_20260222_203756_lounge_lidar --method-tag pipeline_full --results-root run_20260222_203756_lounge_lidar/results
```

Then:

```bash
python3 scripts/plot_overlays.py --by-run-json run_20260222_203756_lounge_lidar/results/by_run/<latest>.json --run-dir run_20260222_203756_lounge_lidar
```

Or use the bash wrappers under `bash_scripts/` (e.g. `run_eval_pair.sh`).
