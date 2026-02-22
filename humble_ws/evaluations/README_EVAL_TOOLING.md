# Evaluation Tooling (Phase 1)

This tooling lets you recompute metrics from saved run artifacts (no ROS/SAM3D rerun).

## Inputs

- Prediction:
  - `scene_graph.json` or `scene_graph_sam3d_only.json` (pipeline)
  - CLIO GraphML (`.graphml`)
- GT: Supervisely `*_voxel_pointcloud.pcd.json`
- Configs:
  - `eval_config.json`
  - `canonical_labels.json`
  - `label_aliases.json`

## Run single evaluation

```bash
python3 run_eval.py \
  --prediction-type scene_graph \
  --prediction-path /abs/path/to/run_dir/scene_graph.json \
  --gt-json /abs/path/to/supervisely_gt.json \
  --run-dir /abs/path/to/run_dir \
  --scene hallway1 \
  --run-id run_20260218_120000
```

Legacy equivalent (still supported):

```bash
python3 run_eval.py \
  --prediction-json /abs/path/to/run_dir/scene_graph.json \
  --gt-json /abs/path/to/supervisely_gt.json \
  --run-dir /abs/path/to/run_dir \
  --scene hallway1 \
  --run-id run_20260218_120000
```

CLIO comparison run:

```bash
python3 run_eval.py \
  --prediction-type clio \
  --prediction-path /abs/path/to/clio_scene.graphml \
  --gt-json /abs/path/to/supervisely_gt.json \
  --scene hallway1 \
  --run-id clio_hallway1
```

Pipeline vs CLIO side-by-side (same GT):

```bash
python3 compare_pipeline_clio.py \
  --results-root /abs/path/to/results \
  --scene hallway1 \
  --run-id run_20260218_120000
```

evaluations/bash_scripts/run_compare_pipeline_vs_clio.sh   /data/sam3d_queue/run_20260222_051228   evaluations/clio/lounge_0.graphml   evaluations/supervisely/lounge-0_voxel_pointcloud.pcd.json   lounge-0   run_20260222_051228   /data/sam3d_queue/run_20260222_051228/results


Writes:
- `results/comparisons/<scene>_<run_id>_pipeline_vs_clio_summary.csv`
- `results/comparisons/<scene>_<run_id>_pipeline_vs_clio_per_class.csv`

Outputs:
- `results/by_run/<scene>_<run_id>_<evalhash>.json`
- `results/by_run/<scene>_<run_id>_<evalhash>_match_details.csv`
- append row to `results/tables/benchmark_compat_latest.csv`

Notes:
- Unknown labels can be mapped via embedding-style resolver and persisted for deterministic reuse in
  `learned_label_aliases.json` (configured in `eval_config.json`).
- Per-class TP/FP/FN and per-class P/R/F1 are written into each by-run JSON under `per_class`.
- Matching supports `matching.iou_mode` in `eval_config.json`:
  - `"obb"`: oriented 3D IoU (default)
  - `"aabb"`: axis-aligned 3D IoU
- Optional scene normalization for fairer AABB metrics:
  - `matching.frame_normalization = "gt_global_yaw"` rotates GT+pred by a single GT-estimated scene yaw before box extraction.
  - Set to `"none"` to disable.
- By-run JSON also includes richer diagnostics under `diagnostics`:
  - `registration_summary` (helped/hurt rates from initial→final centroid error on matched objects)
  - `retrieval_summary` (swap rate and swap TP rate)
- By-run JSON includes `instances.predictions` and `instances.gts`, plus explicit match/unmatched indices so you can audit label mapping and pairing quality.
- Open-vocabulary cleanup: `eval_config.json -> prediction.ignore_*` lets you ignore noisy non-target labels before matching (saved in by-run JSON under `prediction_filter`).

## Build paper tables from all by-run JSON

```bash
python3 collect_ablation_table.py --results-root /home/hsu/repos/Real2USD/humble_ws/evaluations/results
```

Outputs:
- `results/tables/ablation_main_<timestamp>.csv`
- `results/tables/ablation_main_<timestamp>.md`
- `results/tables/ablation_main_<timestamp>.tex`
- `results/tables/benchmark_compat_<timestamp>.csv`
- `results/tables/per_class_<timestamp>.csv`
- Main tables now include additional diagnostics columns such as:
  - `iou_mode`
  - `registration_helped_rate`, `registration_hurt_rate`, `registration_delta_centroid_m`
  - `retrieval_swapped_rate`, `retrieval_swapped_tp_rate`

## Plotting (matplotlib/seaborn)

```bash
python3 plot_eval.py --results-root /home/hsu/repos/Real2USD/humble_ws/evaluations/results
```

Writes:
- `results/plots/quality_vs_latency.png`
- `results/plots/per_class_f1.png`
- `results/plots/per_class_tpfpfn.png`
- `results/plots/registration_vs_retrieval.png`

Per-run bbox overlay plot:

```bash
python3 plot_bbox_overlays.py --by-run-json /abs/path/to/results/by_run/<scene>_<run_id>_<evalhash>.json
```

Writes:
- `results/plots/<scene>_<run_id>_<evalhash>_bbox_overlay_xy.png`

Alias suggestion helper from match diagnostics:

```bash
python3 suggest_aliases.py --results-root /abs/path/to/results
```

Or for a specific run CSV:

```bash
python3 suggest_aliases.py \
  --match-csv /abs/path/to/results/by_run/<scene>_<run_id>_<evalhash>_match_details.csv \
  --output-json /abs/path/to/results/diagnostics/alias_suggestions.json
```

Sweep evaluation (geometry ceiling + label penalty):

```bash
python3 sweep_eval.py \
  --prediction-json /abs/path/to/run_dir/scene_graph.json \
  --gt-json /abs/path/to/supervisely_gt.json \
  --run-dir /abs/path/to/run_dir \
  --scene hallway1 \
  --run-id run_20260218_120000 \
  --thresholds 0.05,0.10,0.15,0.20,0.25 \
  --label-match-options false,true
```

Writes:
- `results/sweeps/sweep_raw_<scene>_<run_id>_<timestamp>.csv`
- `results/sweeps/sweep_summary_<scene>_<run_id>_<timestamp>.csv`
- `results/sweeps/by_setting/*.json`

## Sensor geometry alignment (lidar vs realsense)

Use this to validate geometry overlap and estimate `T_lidar_from_realsense` before applying correction in the pipeline.

```bash
python3 sensor_alignment_eval.py \
  --realsense-points /abs/path/to/realsense_points.npy \
  --lidar-points /abs/path/to/lidar_points.npy \
  --scene hallway1 \
  --run-id run_20260218_120000 \
  --config /home/hsu/repos/Real2USD/humble_ws/evaluations/sensor_alignment_config.json \
  --results-root /home/hsu/repos/Real2USD/humble_ws/evaluations/results
```

Point cloud formats:
- `.npy` (`Nx3`) is supported directly.
- `.ply` and `.pcd` are also supported when `open3d` is installed.

Outputs:
- `results/diagnostics/sensor_alignment_<scene>_<run_id>.json`
- append row to `results/tables/sensor_alignment_latest.csv`

Key reported fields:
- Before/after nearest-neighbor median and p95 distance (m)
- Before/after Chamfer mean (m)
- ICP fitness and residual (m)
- Estimated `transform_lidar_from_realsense` (`4x4`, translation, Euler angles)
- Acceptance decision based on thresholds in `sensor_alignment_config.json`
