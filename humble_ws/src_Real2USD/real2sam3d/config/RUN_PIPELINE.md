# Real2SAM3D Run + Debug Guide

This is the only runbook you need for daily use.

## What the code does now

- Worker writes vanilla SAM3D outputs (`object.glb`, `object.ply`) plus raw transform inputs in `pose.json`.
- Injector applies the demo_go2-equivalent transform chain and writes:
  - `transform_odom_from_raw`
  - `transform_odom_from_cam`
  - `transform_cam_from_raw`
  - `initial_position`, `initial_orientation`
  - `object_odom.glb`
- Registration uses `initial_*` (z-up odom from injector) as ICP init and publishes final pose on `/usd/StringIdPose`. With `yaw_only_registration:=true` (default), the result is constrained to rotation about Z so objects stay z-up.
- Scene buffer writes scene outputs from published poses.

**Registration frames:** Source = object mesh (object.glb) in its local frame; target = segment or global point cloud in **odom** (z-up). The bridge sends the injector’s `initial_position` / `initial_orientation` (z-up odom) so the initial pose is already in the same frame as the target. ICP then refines position and yaw only (if yaw_only_registration is true).

**Registration visualization:** Add in RViz (frame: odom):
- `/registration/overlay` — registered object (red) + target (cyan) in one cloud.
- `/registration/pc` — registered object points only.
- `/debug/registration/source_pc` — raw source at origin; `/debug/registration/target_pc` — target.

## 1) Build

From `humble_ws`:

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select custom_message real2sam3d --symlink-install
source install/setup.bash
```

## 2) Start ROS pipeline (Docker)

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch real2sam3d real2sam3d.launch.py
```

Notes:
- Keep queue path shared between host and container (commonly `/data/sam3d_queue`).
- Launch writes `current_run.json`. Worker/indexer/render scripts use this by default.
- Use `no_faiss_mode:=true` to bypass retrieval and use candidate object directly.
- Segmentation model: default is prompt-free YOLOE (`models/yoloe-11l-seg-pf.pt`). Set `use_yolo_pf:=false` to use prompted weights (`models/yoloe-11l-seg.pt`).
- Optional strict pre-SAM3D admission gate: set `enable_pre_sam3d_quality_filter:=true`. This loads tracker + filter settings from `config/tracking_pre_sam3d_filter.json` (and tracker YAML from `config/botsort_lenient_go2.yaml`) so tracker assignment is more lenient while SAM3D enqueue is stricter.
  - Run logs written in run dir by job writer:
    - `pre_sam3d_filter_log.json`: counts for received/enqueued/skipped and skip reasons.
    - `unique_labels_log.json`: unique labels seen/enqueued/skipped and per-label counts.
- Full point cloud snapshots (for alignment/eval): enabled by default via `save_full_pointcloud:=true`.
  - Saved as `.npy` (`Nx3`, odom frame) under:
    - `run_dir/diagnostics/pointclouds/lidar/`
    - `run_dir/diagnostics/pointclouds/realsense/`
  - Cadence is `pointcloud_save_period_sec` (default `5.0`) plus one final snapshot at shutdown.
- **init_odom:** Default is **off** (`use_init_odom:=false`). The injector uses raw odom; no first-frame normalization. Set `use_init_odom:=true` to normalize poses by first-frame odom (demo_go2-style) when you want to compare or match that behavior.
- **Run config (experiment arguments):** Each run directory gets `run_config.json` with:
  - **launch:** all launch arguments (e.g. `use_realsense_cam`, `sam3d_retrieval`, `no_faiss_mode`, `glb_registration_bridge`, `pipeline_profiler`, …) so you know how the pipeline was started.
  - **tracking_pre_sam3d_filter / tracker_yaml:** embedded contents + resolved paths of the filter JSON and tracker YAML used for the run.
  - **worker:** written when you run `run_sam3d_worker.py` against this run dir (e.g. `use_depth`, `dry_run`, `sam3d_repo`, `once`, …). One file per experiment with both launch and worker args.
- **Profiler timing logs:** With `pipeline_profiler:=true` (default), the pipeline profiler writes timing data into the **run directory** so you can report inference times across pipeline variations (ablations). Each run dir gets:
  - `timing_events.csv`: one row per event (`stamp_sec`, `node_name`, `step_name`, `duration_ms`).
  - `timing_summary.json`: per (node, step) stats: `count`, `mean_ms`, `std_ms`, `min_ms`, `max_ms` (e.g. `sam3d_worker` / `inference` for SAM3D latency). Use these files for tables and comparison across sensor, pointmap, retrieval, registration settings.

## 3) Run worker (host conda)

```bash
conda activate sam3d-objects
cd /path/to/Real2USD/humble_ws/src_Real2USD/real2sam3d/scripts_sam3d_worker
python run_sam3d_worker.py --sam3d-repo /path/to/sam-3d-objects
```

Useful flags:
- `--once` process one job and exit.
- `--dry-run` no SAM3D inference (debug queue plumbing only).
- `--no-current-run --queue-dir /abs/path/to/run_dir` target a specific run.
- `--write-demo-go2-compare` write worker debug compare artifacts.
- `--use-init-odom` is deprecated (injector handles init_odom normalization).

## 4) Optional retrieval tooling

Indexer:

```bash
python index_sam3d_faiss.py
```

Multi-view renderer:

```bash
python render_sam3d_views.py
```

Both also support `--no-current-run --queue-dir ...` for manual targeting.

## 5) Transform debug workflow (important)

### Enable injector debug dump

```bash
export REAL2SAM3D_DEBUG_DUMP=1
export REAL2SAM3D_DEBUG_TRANSFORM=1
```

Per job output:
- `output/<job_id>/transform_debug.json`

Contains:
- constants (`R_flip_z`, `R_yup_to_zup`, etc.)
- inputs (odom, SAM3D scale/rot/trans, init_odom)
- matrices (`T_cam_to_odom`, `T_raw_to_cam`, `T_raw_to_odom`)
- basis test with `odom_direct_minus_matrix`

### Enable worker compare dump

Run worker with:

```bash
python run_sam3d_worker.py --sam3d-repo /path/to/sam-3d-objects --write-demo-go2-compare
```

Per job output:
- `object_demo_go2_cam.glb`
- `object_demo_go2_odom.glb`
- `demo_go2_compare.json`

### Apples-to-apples check

Compare for same job:
- `output/<job_id>/transform_debug.json`
- `output/<job_id>/demo_go2_compare.json`

Focus on:
- `basis_test.cam_*`
- `basis_test.odom_*`
- `basis_test.odom_direct_minus_matrix` (should be near zero)

## 6) Quick sanity checklist

- `pose.json` after injector has `frame = "z_up_odom"`.
- `transform_odom_from_raw` exists and is 4x4.
- `object_odom.glb` exists in each completed output job.
- Quaternion convention defaults:
  - odom: `xyzw`
  - SAM3D: `wxyz`
- If odom is `wxyz`, set:

```bash
export REAL2SAM3D_ODOM_QUAT_WXYZ=1
```
