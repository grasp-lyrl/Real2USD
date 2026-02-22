# Evaluation Bash Scripts

These scripts are the minimal supported set for evaluation runs.

## Scripts

- `run_eval_pair.sh`
  - Runs `run_eval.py` for:
    - `scene_graph.json` (registration output)
    - `scene_graph_sam3d_only.json` (baseline)
  - Writes results under `<RUN_DIR>/results/`
  - Writes per-run bbox overlay plots
- `run_sweep_eval.sh`
  - Runs IoU/label-match sweeps to estimate geometry ceiling and label penalty.
- `run_full_three_method_comparison.sh`
  - Single command for 3-method comparison:
    - pipeline full (`scene_graph.json`)
    - SAM3D-only (`scene_graph_sam3d_only.json`)
    - CLIO (`.graphml`)
  - Writes unified tables/plots + 3-way comparison CSV/plots + bbox overlays for all methods.

## One-time setup

Make scripts executable:

```bash
chmod +x evaluations/bash_scripts/*.sh
```

## Usage

```bash
evaluations/bash_scripts/run_eval_pair.sh \
  /data/sam3d_queue/run_20260222_051228 \
  evaluations/supervisely/hallway-1_voxel_pointcloud.pcd.json \
  hallway1
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
