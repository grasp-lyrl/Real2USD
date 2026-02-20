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
- Registration uses `initial_*` as ICP init and publishes final pose on `/usd/StringIdPose`.
- Scene buffer writes scene outputs from published poses.

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
