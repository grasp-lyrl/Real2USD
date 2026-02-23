# Evaluation Bash Scripts

These scripts are the minimal supported set for evaluation runs.

## Scripts

- `run_eval_pair.sh`
  - Runs `scripts/run_eval.py` for:
    - `scene_graph.json` (pipeline_full)
    - `scene_graph_sam3d_only.json` (sam3d_only)
    - optionally **CLIO** if you pass a 5th argument: `CLIO_GRAPHML` path to a `.graphml` file
  - Writes results under `<RUN_DIR>/results/by_run/`
  - Runs `scripts/plot_overlays.py` per method: bbox vs GT and (if available) bbox+GT on point cloud, axis-aligned
- `run_sweep_eval.sh`
  - Runs IoU/label-match sweeps to estimate geometry ceiling and label penalty.
- `run_full_three_method_comparison.sh`
  - Single command for 3-method comparison:
    - pipeline full (`scene_graph.json`)
    - SAM3D-only (`scene_graph_sam3d_only.json`)
    - CLIO (`.graphml`)
  - Writes unified tables/plots + 3-way comparison CSV/plots + overlay plots (bbox vs GT, bbox on PC when available) for all methods.
  - Also runs pose-sensitivity diagnostics per method and a lidar-vs-realsense decision report.

## One-time setup

Make scripts executable:

```bash
chmod +x evaluations/bash_scripts/*.sh
```

## Usage

```bash
# Pipeline + SAM3D-only only
evaluations/bash_scripts/run_eval_pair.sh \
  <RUN_DIR> \
  evaluations/supervisely/lounge-0_voxel_pointcloud.pcd.json \
  lounge

# Include CLIO (5th arg = path to CLIO .graphml)
evaluations/bash_scripts/run_eval_pair.sh \
  <RUN_DIR> \
  evaluations/supervisely/lounge-0_voxel_pointcloud.pcd.json \
  lounge \
  "" \
  /path/to/clio_scene.graphml
```

```bash
evaluations/bash_scripts/run_sweep_eval.sh \
  /data/sam3d_queue/run_20260222_051228 \
  evaluations/supervisely/lounge-0_voxel_pointcloud.pcd.json \
  lounge-0
```

```bash
evaluations/bash_scripts/run_full_three_method_comparison.sh \
  /data/sam3d_queue/run_20260222_051228 \
  /abs/path/to/clio_scene.graphml \
  evaluations/supervisely/lounge-0_voxel_pointcloud.pcd.json \
  lounge-0 \
  run_20260222_051228 \
  /data/sam3d_queue/run_20260222_051228/results_labels
```
