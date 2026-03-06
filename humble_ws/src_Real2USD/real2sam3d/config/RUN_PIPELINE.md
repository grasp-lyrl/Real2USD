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
- Scene buffer writes scene outputs from published poses. **scene_graph.json** is **decoupled from GLB**: on each timer tick the node writes **scene_graph.json** and **scene_graph_sam3d_only.json** first (synchronously), then starts the GLB write in a background thread. So the JSON files are always complete even if scene.glb fails or the process is killed (e.g. OOM) during GLB. JSON is also updated when poses arrive (`write_on_pose` debounce). **scene.glb** and **scene_sam3d_only.glb** are updated on the same timer in a single background thread (one at a time to avoid OOM).

**Why is the registration backlog so large? (I thought the bottleneck was SAM3D.)** In steady state it is: SAM3D takes ~10 min per job, and registration takes ~0.6 s per slot, so registration easily keeps up. The backlog happens when **many slots become ready at once**. The injector runs a **periodic scan** (`watch_interval_sec`, default 1 s): on each tick it looks at `output/` and publishes **SlotReady for every job that has output but isn’t yet in _published_job_ids**. So if you start the pipeline when `output/` already has 37 completed jobs (e.g. from a previous run or a pre-filled queue), the **first** scan publishes 37 SlotReady messages in one burst. Retrieval and the bridge then publish 37 registration requests in quick succession. Registration processes them one at a time (~0.6 s each), so it takes ~22 s to clear the queue. If you stop the run before that, you see 22 in scene.glb and 15 still in the registration queue. So the bottleneck in steady state is SAM3D; the burst comes from the injector’s “publish all ready slots” behavior on each scan. To avoid a big backlog on startup, either run with an empty `output/` or let the pipeline run long enough after startup for registration to drain the queue. You can also set **injector** parameter **max_slots_per_scan:=1** (or 2) so at most that many new slots are published per watch tick; registration then gets a steady stream and won’t be hit with 37 requests at once.

**Registration frames:** Source = object mesh (object.glb) in its local frame; target = segment or global point cloud in **odom** (z-up). The bridge sends the injector’s `initial_position` / `initial_orientation` (z-up odom) so the initial pose is already in the same frame as the target. ICP then refines position and yaw only (if yaw_only_registration is true). **RealSense:** When `use_realsense_cam:=true`, the registration target is the **segment** (local) point cloud — the same depth+mask unprojection that was fed into SAM3D for that job — not the accumulated global point cloud. This avoids relying on the RealSense accumulated world PC and matches registration to the observation that produced the mesh. Lidar pipeline continues to use the global accumulated PC by default (`registration_target_mode:=global`).

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

### Run pipeline after worker only (existing output folder)

If you have a run directory whose `output/` is already filled by the SAM3D worker (e.g. you ran the worker offline or copied results), you can run the rest of the pipeline (injector → retrieval → bridge → registration → scene buffer) without the camera or job writer:

```bash
ros2 launch real2sam3d real2sam3d.launch.py post_worker_only:=true sam3d_queue_dir:=/path/to/your/run
```

- **post_worker_only:=true** — Disables the camera nodes and job writer; only injector, retrieval (if not no_faiss_mode), bridge, registration, and scene buffer run. The launch does not create a new run subdir; it uses `sam3d_queue_dir` as the run directory.
- Point **sam3d_queue_dir** at the run that contains `output/<job_id>/` with at least `pose.json` and `object.glb` per job. For registration with segment target, each `output/<job_id>/` should also have `depth.npy`, `mask.png`, and `meta.json` (the worker copies these when it runs).
- Use **no_faiss_mode:=true** if you don’t have a FAISS index and want the injector to publish the slot’s own object directly (no retrieval).
- Use **registration_target_mode:=segment** so registration uses the segment point cloud from each job dir (depth.npy, mask.png, meta.json). With no camera running, global target has no data.
- After the pipeline runs, use the offline scene buffer script if you need to regenerate scene files: `ros2 run real2sam3d run_offline_scene_buffer --run-dir /path/to/your/run`

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
- **Depth reliability gates (new):**
  - `realsense_min_depth_m` / `realsense_max_depth_m` control usable RealSense depth range before RGBD point generation and global pointcloud accumulation.
  - `lidar_min_range_m` / `lidar_max_range_m` control usable lidar range before depth projection and global pointcloud accumulation.
- **Registration robustness controls (new):**
  - `registration_min_fitness`, `registration_icp_distance_threshold_m`, `registration_icp_max_iteration`, `registration_min_target_points`, `registration_max_translation_delta_m`.
  - Bridge target mode and locality: `registration_target_mode` (`global` or `segment`) and `registration_target_radius_m` (local crop around initial pose when using global target).
  - Per-attempt diagnostics are appended to `run_dir/diagnostics/registration_metrics.jsonl`.

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

## 7) Troubleshooting

### Slot appears in logs but not in scene_graph.json

If you see a slot (e.g. id 111) go through injector → retrieval → bridge → registration and a log like “low fitness; using initial pose as result”, but that id is missing from **scene_graph.json**, the cause is **timing**: **scene_graph.json** is built from poses that have already been received on `/usd/StringIdPose` at the time of the periodic write. Registration can take tens of seconds; if it finishes *after* the last periodic write, that slot is not yet in the buffer when the file is written.

- **Check:** The same slot should appear in **scene_graph_sam3d_only.json** and in **scene_sam3d_only.glb** (those are built by scanning `output/`, not from registration).
- **Two causes:** (1) **Empty target:** If the global point cloud had no points within `registration_target_radius_m` of the slot's initial pose, registration returns without publishing (you may see `Registration skipped id=... empty clouds`). The code now publishes the initial pose in that case so the slot still appears. (2) **Timing:** Registration finished after the last scene write; use **write_on_pose** (default on) or wait for the next write.
- **Fix:** Keep the pipeline running so another periodic write runs after registration completes, or enable **write_on_pose** (default on): the scene buffer will schedule an extra write a few seconds after each new pose. Parameter: `write_on_pose:=true`, `write_on_pose_debounce_sec:=2.0`.

### simple_scene_buffer_node dies with exit code -9 (SIGKILL)

Exit code -9 usually means the process was killed by the OOM killer (out of memory). The node loads many meshes (scene.glb + scene_sam3d_only.glb, and optionally per-slot GLBs) in background threads. Previously, a new GLB write could start every 5 s while earlier writes were still running, so multiple threads could each hold 80+ meshes in memory and exhaust RAM. **Fix:** The node now allows only one GLB write at a time; if a write is still in progress when the timer fires, that tick is skipped. If you still see OOM with very large runs (e.g. 200+ jobs), consider disabling per-slot GLB writing (`write_per_slot_glb:=false`) or increasing system memory.

### Re-run only the buffer (rebuild scene from existing run)

To regenerate **scene_graph.json** and **scene.glb** from an existing run directory (e.g. after a run where the buffer node didn’t write everything, or to refresh the files from disk):

```bash
source install/setup.bash
ros2 run real2sam3d run_offline_scene_buffer --run-dir /data/sam3d_queue/run_20260224_053223
```

Or from the package root (with `install/setup.bash` sourced):

```bash
python scripts_sam3d_worker/run_offline_scene_buffer.py --run-dir /path/to/run
```

Use `--no-glb` to write only the JSON files (faster). The script scans `output/<job_id>/` for `pose.json` + `object.glb`. When registration has run, each `pose.json` contains **position**, **orientation**, and **registered_data_path** (written by the registration node); the offline script uses those when present, otherwise falls back to initial poses (injector output).

### Refine scene graph (deduplicate objects)

After you have **scene_graph.json**, you can reduce duplicate objects (same real-world instance appearing as multiple slots) using CLIP label similarity and 3D IoU:

```bash
ros2 run real2sam3d refine_scene_graph --input /path/to/run_dir/scene_graph.json \
  --output /path/to/run_dir/scene_graph_reduced.json --iou-threshold 0.2 --clip-threshold 0.8
```

Use `--no-clip` for exact-label-only matching (no torch/transformers). Use `--position-only --position-radius 0.5` when mesh paths are not available or to avoid loading GLBs. See `real2sam3d/refine_scene_graph.py` docstring for full options and suggested workflow.

## 8) Files and config expected in the repo

- **Entry points:** Every `setup.py` console_script has a matching module under `real2sam3d/` (e.g. `realsense_cam_node.py`, `registration_node.py`, …). The commented-out `usd_buffer_node` is intentional.
- **Config:** `config/tracking_pre_sam3d_filter.json` and `config/botsort_lenient_go2.yaml` are required when using `enable_pre_sam3d_quality_filter:=true`. The JSON references the YAML by name; both are installed under `share/real2sam3d/config/`.
- **Gemini key (navigator_llm_node):** `config/gemini_key.py` is in the repo with an empty stub. Copy `config/gemini_key_template.py` to `config/gemini_key.py` and set `GEMINI_API_KEY` if you use the LLM navigator; otherwise the node will start but API calls will fail.
- **Models:** YOLO weights (`models/yoloe-11l-seg.pt` / `models/yoloe-11l-seg-pf.pt`) are not in the repo; see USER_NEXT_STEPS.md for obtaining them.
- **resource/real2sam3d:** Present for ament package index; no other files needed there.
