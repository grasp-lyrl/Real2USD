# Evaluation Bash Scripts

These scripts help you quickly re-run evaluation for an existing run directory.

## Scripts

- `run_eval_pair.sh`
  - Runs `run_eval.py` for:
    - `scene_graph.json` (registration output)
    - `scene_graph_sam3d_only.json` (baseline)
  - Writes results under `<RUN_DIR>/results/`
  - Writes per-run bbox overlay plots
- `collect_and_plot.sh`
  - Rebuilds consolidated tables and plots from all `<RESULTS_ROOT>/by_run/*.json`.
- `rerun_eval_full.sh`
  - Runs both commands above in sequence.
- `run_eval_clio.sh`
  - Runs CLIO GraphML vs Supervisely GT in the same `run_eval.py` format.
- `run_compare_pipeline_vs_clio.sh`
  - Evaluates pipeline + CLIO against the same GT and writes side-by-side comparison CSVs.

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
evaluations/bash_scripts/collect_and_plot.sh
```

Custom results root:

```bash
evaluations/bash_scripts/collect_and_plot.sh /data/sam3d_queue/run_20260222_051228/results
```

```bash
evaluations/bash_scripts/rerun_eval_full.sh \
  /data/sam3d_queue/run_20260222_051228 \
  evaluations/supervisely/hallway-1_voxel_pointcloud.pcd.json \
  hallway1
```

```bash
evaluations/bash_scripts/run_eval_clio.sh \
  /abs/path/to/clio_scene.graphml \
  evaluations/supervisely/hallway-1_voxel_pointcloud.pcd.json \
  hallway1 \
  clio_hallway1 \
  /abs/path/to/results_labels
```

```bash
evaluations/bash_scripts/run_compare_pipeline_vs_clio.sh \
  /data/sam3d_queue/run_20260222_051228 \
  /abs/path/to/clio_scene.graphml \
  evaluations/supervisely/lounge-0_voxel_pointcloud.pcd.json \
  lounge-0 \
  run_20260222_051228 \
  /data/sam3d_queue/run_20260222_051228/results_labels
```
