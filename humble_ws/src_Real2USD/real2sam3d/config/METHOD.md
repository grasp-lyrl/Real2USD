# Real2SAM3D Methodology

## Abstract

`real2sam3d` is a ROS2-centered perception-to-scene pipeline that converts online object crops into reusable 3D assets, retrieves semantically similar assets via CLIP+FAISS, and estimates metric object poses in odom/world coordinates through ICP registration. The system is intentionally split across environments: ROS nodes run in Docker, while SAM3D inference runs on the host in a conda environment. A disk queue provides asynchronous handoff. The transform pipeline from SAM3D raw mesh coordinates to odom is implemented to match `demo_go2` behavior exactly, with per-stage debug artifacts for parity validation.

## 1. Problem Setup

Given synchronized RGB/depth/segmentation/odometry from a robot:

1. Generate a 3D object asset from a segmented crop (SAM3D worker).
2. Convert object geometry from SAM3D/raw coordinates into odom/world.
3. Retrieve the best reusable object candidate from an indexed asset bank (multi-view CLIP+FAISS).
4. Register the selected source object to target scene geometry.
5. Publish final object poses and write scene artifacts.

This package assumes multi-view indexing and FAISS-based retrieval are active during normal operation.

## 2. System Architecture

Pipeline graph (default mode):

1. `lidar_cam_node` -> publishes crop/depth/odom stream.
2. `sam3d_job_writer_node` -> writes jobs to `queue_dir/input/<job_id>/`.
3. `run_sam3d_worker.py` (host process) -> writes outputs to `queue_dir/output/<job_id>/`.
4. `sam3d_injector_node` -> computes transform fields + `object_odom.glb`; publishes slot readiness.
5. `sam3d_retrieval_node` -> chooses best object path using FAISS (fallback to candidate).
6. `sam3d_glb_registration_bridge_node` -> builds source/target point clouds + initial pose.
7. `registration_node` -> ICP, publishes final `UsdStringIdPose`.
8. `simple_scene_buffer_node` -> writes `scene_graph.json`, `scene.glb`, and SAM3D-only comparison artifacts.

Launch-level orchestration is in `launch/real2sam3d.launch.py`.

## 3. Data Contracts and Queue Semantics

Per-job queue structure:

- Input job (`input/<job_id>/`):
  - `rgb.png`
  - `mask.png`
  - `depth.npy`
  - `meta.json` (includes camera intrinsics and odom pose)

- Worker output (`output/<job_id>/`):
  - `object.glb` (vanilla SAM3D geometry)
  - `object.ply` (when available)
  - `pose.json` (raw fields, then injector-augmented transforms)
  - `rgb.png`, `depth.npy`, `mask.png`, `meta.json` (copied for self-contained downstream use)
  - Optional debug outputs (`demo_go2_compare.json`, comparison GLBs)

Run management:

- By default launch creates `run_YYYYMMDD_HHMMSS` and writes `current_run.json`.
- Worker/indexer/renderer scripts read current run by default unless `--no-current-run` is used.

## 4. Geometric Method

### 4.1 Coordinate Conventions

- Odom/world frame: Z-up (robot localization frame).
- SAM3D object transform:
  - scale: 3-vector or scalar
  - rotation: quaternion in `wxyz`
  - translation: 3-vector
- Odom quaternion default: `xyzw` (configurable via `REAL2SAM3D_ODOM_QUAT_WXYZ=1`).

### 4.2 Raw-to-Camera-to-Odom Chain

For raw vertex row-vector `v_raw`:

1. Axis preparation:
   - `v1 = v_raw @ R_flip_z`
   - `v2 = v1 @ R_yup_to_zup`
2. SAM3D affine:
   - `v3 = (v2 * s) @ R_obj + t`
3. Pointmap/camera alignment:
   - `v_cam = v3 @ R_pytorch3d_to_cam`
4. Camera -> world -> odom:
   - `v_world = v_cam @ R_world_to_cam.T + t_world_to_camera`
   - `v_odom = v_world @ R_odom.T + t_odom_rel`

Where `t_odom_rel = t_odom - [init_odom.x, init_odom.y, 0]` when `init_odom` normalization is active.

The implementation is in `real2sam3d/ply_frame_utils.py` and used by `sam3d_injector_node.py`.

### 4.3 Affine Matrix Construction

To avoid algebraic drift, 4x4 transforms are constructed from mapped basis points:

- Map `[0, e1, e2, e3]` through literal vertex chain.
- Recover affine matrix from mapped origin and basis vectors.

This produces:

- `T_cam_to_odom`
- `T_raw_to_cam`
- `T_raw_to_odom = T_cam_to_odom @ T_raw_to_cam`

These are written into `pose.json`.

## 5. Retrieval with Multi-View FAISS

Indexer (`scripts_sam3d_worker/index_sam3d_faiss.py`):

1. Scans completed output jobs.
2. Uses `views/*.png` if present; otherwise `rgb.png`.
3. Adds each image embedding (CLIP) to FAISS with object path metadata.
4. Saves index to `<index_path>/faiss/index.faiss` + metadata.

Multi-view renderer (`scripts_sam3d_worker/render_sam3d_views.py`):

1. Loads `object.glb` preferentially (or `object.ply` fallback).
2. Renders azimuthal views around centered geometry.
3. Writes `output/<job_id>/views/0..N-1.png`.

Retrieval node (`real2sam3d/sam3d_retrieval_node.py`):

1. On `/usd/SlotReady`, reads slot crop image (`output/<job_id>/rgb.png`).
2. Computes embedding, queries FAISS top-1 cosine result.
3. Publishes best `data_path` on `/usd/Sam3dObjectForSlot`.
4. Falls back to candidate object when index unavailable/empty/error.

## 6. Registration Method

Bridge (`sam3d_glb_registration_bridge_node.py`):

- Source point cloud:
  - sampled from selected object file (`.glb` preferred, `.ply` fallback)
- Target point cloud:
  - **Lidar (default):** global accumulated world point cloud (`/global_lidar_points`), cropped around initial pose.
  - **RealSense:** when `use_realsense_cam:=true`, target is the **segment** (local) point cloud from the job dir (`depth.npy` + `mask.png` + `meta.json`) — the same unprojected points that were fed into SAM3D. This avoids relying on the accumulated RealSense world PC and registers against the observation that produced the mesh. Lidar can use segment by setting `registration_target_mode:=segment`.
- Initial pose:
  - loaded from `pose.json` (`initial_position`, `initial_orientation`)
  - forwarded in `UsdStringIdSrcTargMsg.initial_pose`

Registration (`registration_node.py`):

- If initial pose exists:
  - single ICP against full target cloud from initial transform
  - low-fitness safeguard: fallback to initial pose if fitness < threshold
- Else:
  - cluster-based path (DBSCAN + FGR + ICP multi-yaw initialization)

Final outputs are published on `/usd/StringIdPose`.

## 7. Scene Composition Outputs

`simple_scene_buffer_node.py` writes:

- `scene_graph.json` + `scene.glb` (registration results)
- `scene_graph_sam3d_only.json` + `scene_sam3d_only.glb` (pre-registration transform-only results)
- per-slot artifacts:
  - `object_odom.glb`
  - `object_registered.glb`

This dual output supports separation of transform errors from ICP errors.

## 8. Debugging and Verification Methodology

### 8.1 Injector-side transform trace

Enable:

- `REAL2SAM3D_DEBUG_DUMP=1`
- `REAL2SAM3D_DEBUG_TRANSFORM=1`

Produces `transform_debug.json` with:

- constants, inputs, transform matrices
- basis-point direct-chain vs matrix-chain comparison
- numerical residuals (`odom_direct_minus_matrix`)

### 8.2 Worker-side parity trace

Run worker with `--write-demo-go2-compare` to produce:

- `object_demo_go2_cam.glb`
- `object_demo_go2_odom.glb`
- `demo_go2_compare.json`

See **config/DEMO_GO2_PARITY.md** for a full demo_go2 vs sam3d_only comparison and how to get similar results.

### 8.3 Apples-to-apples criterion

For same `job_id`, compare:

- `transform_debug.json`
- `demo_go2_compare.json`

The basis-chain outputs should match within floating-point tolerance. Deviations localize the first failing stage in the transform stack.

## 9. Design Rationale

1. **Asynchronous queue split** isolates GPU-heavy SAM3D from ROS runtime concerns.
2. **Literal transform replication** prioritizes deterministic parity over elegant but risky algebraic rewrites.
3. **Multi-view indexing** increases retrieval robustness to viewpoint mismatch.
4. **Initial-pose ICP** reduces search space and improves registration stability.
5. **Dual scene outputs** (sam3d-only vs registered) provide fast causal debugging.

## 10. Known Assumptions and Failure Modes

- FAISS/CLIP may be unavailable in Docker; retrieval then falls back to candidate.
- If output job lacks copied crop/depth/mask/meta, segment-target registration can degrade.
- Quaternion convention mismatch (`xyzw` vs `wxyz`) can silently corrupt transforms; env override is provided.
- Sparse/partial point clouds can trigger low ICP fitness and fallback behavior.

## 11. Practical Reading Map (Code to Method Sections)

- Queue writer: `real2sam3d/sam3d_job_writer_node.py`
- Worker: `scripts_sam3d_worker/run_sam3d_worker.py`
- Transform core: `real2sam3d/ply_frame_utils.py`
- Injector: `real2sam3d/sam3d_injector_node.py`
- Multi-view renderer: `scripts_sam3d_worker/render_sam3d_views.py`
- FAISS indexer: `scripts_sam3d_worker/index_sam3d_faiss.py`
- Retrieval: `real2sam3d/sam3d_retrieval_node.py`
- Bridge: `real2sam3d/sam3d_glb_registration_bridge_node.py`
- Registration: `real2sam3d/registration_node.py`
- Scene buffer: `real2sam3d/simple_scene_buffer_node.py`
- Launch orchestration: `launch/real2sam3d.launch.py`
