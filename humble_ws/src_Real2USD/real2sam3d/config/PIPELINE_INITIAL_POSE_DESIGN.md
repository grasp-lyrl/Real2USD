# Pipeline: SAM3D Initial Pose + Global Registration + Retrieval

## Goal

1. **Slots** = track_id; each slot gets an **initial pose** from SAM3D output + robot pose (object-agnostic).
2. **Retrieval** picks the best object for the slot (e.g. past “nice chair” from FAISS).
3. **Registration** uses that **initial pose** and aligns the chosen object to the **global (accumulated) point cloud**.
4. **Simple buffer** aggregates final poses → scene_graph.json + scene.glb.

## Current vs proposed

| Step | Current | Proposed |
|------|--------|----------|
| Initial pose | None; registration uses cluster centroid from target | SAM3D scale/rotation/translation + robot odom → object pose in world; use as init for ICP |
| Registration target | Segment PC from job or world PC fallback | **Default: global** accumulated lidar; registration starts from **initial pose** (SAM3D+go2). Set `registration_target:=segment` for segment-based experiments. |
| Retrieval | FAISS on crop image → best path; slot = job_id + track_id | Unchanged conceptually; verify FAISS index and logging |

## 1. SAM3D initial pose (worker + convention)

**Reference:** [demo_go2.py](https://github.com/christopher-hsu/sam-3d-objects/blob/main/demo_go2.py): `transform_glb(out)` puts mesh in camera frame; `transform_glb_to_world(mesh, odom, init_odom)` uses `R_world_to_cam`, `t_world_to_camera`, then odom (R, t) to get mesh vertices in world. Object pose = centroid of transformed mesh + orientation from SAM3D rotation + camera→world.

**Worker today:** Writes `pose.json` with robot odom only (`position`, `orientation`). Exports GLB/PLY **centered at origin** (no world transform). SAM3D output has `scale`, `rotation` (quat), `translation` in camera/pointmap frame.

**Change:**

- In **run_sam3d_worker.py**, after SAM3D inference:
  - Compute **object pose in world (odom frame)** from SAM3D `scale`, `rotation`, `translation` + meta `odometry`, using the same convention as demo_go2 (camera→world then world→odom; go2: `R_world_to_cam`, `t_world_to_camera`).
  - Write **initial pose** to `pose.json`: e.g. `initial_position` [x,y,z], `initial_orientation` [qx,qy,qz,qw] in odom frame. Keep existing `position`/`orientation` (robot pose) for compatibility if needed.
- Reuse or mirror logic from **ply_frame_utils** (`build_world_T_camera`, transform_vertices_camera_to_world_local) and demo_go2’s `transform_glb_to_world` so the same math lives in the worker (or a shared helper).

**Output:** `output/<job_id>/pose.json` contains `initial_position` and `initial_orientation` (object in world) for the bridge/registration node.

**Note:** Initial pose uses SAM3D `translation` (object center in camera frame) and `rotation`; `scale` affects exported geometry but is not used for the pose (center = translation when local centroid is origin).

## 2. Bridge: target = global point cloud when initial pose present

**Current:** Target = segment PC from `output/<job_id>/` (depth.npy + mask.png + meta.json); fallback = latest world PC.

**Change:**

- Bridge reads **initial pose** from `output/<job_id>/pose.json` (e.g. `initial_position`, `initial_orientation`).
- If initial pose is valid:
  - **Target = global accumulated point cloud** (subscribe to `/global_lidar_points` or keep using `world_point_cloud_topic`; ensure it’s the accumulated buffer from lidar_cam_node, not a single scan).
  - Publish **initial_pose** in the registration message (see below).
- If initial pose missing (e.g. legacy job): keep current behavior (segment PC or world PC fallback).

**Topic for target:** Prefer **`/global_lidar_points`** (accumulated in lidar_cam_node) so registration aligns to the full scene. Parameter: e.g. `world_point_cloud_topic` default `/global_lidar_points` when using initial pose, or a separate `registration_target_topic`.

## 3. Messages: pass initial pose to registration

- **UsdStringIdSrcTargMsg:** Add optional `geometry_msgs/Pose initial_pose`. If all zeros or not set, registration node behaves as today (no init pose).
- **Bridge:** When publishing StringIdSrcTarg, set `initial_pose` from job’s pose.json when available; otherwise leave unset/zero.
- **Sam3dObjectForSlotMsg** already has `geometry_msgs/Pose pose`; retrieval currently sets identity. Optionally have **injector** (or bridge) fill `pose` from `output/<job_id>/pose.json` so downstream sees initial pose in one place; bridge can still read from job dir to fill UsdStringIdSrcTarg.

Decision: **Bridge** reads `output/<job_id>/pose.json` and sets `UsdStringIdSrcTargMsg.initial_pose`; no change to SlotReady or Sam3dObjectForSlot for pose (or we add initial_pose to SlotReady later if we want retrieval to see it).

## 4. Registration node: use initial pose and register to global PC

**Current:** Cluster target (DBSCAN), try each cluster with FGR+ICP; init = cluster centroid + identity rotation; multiple yaw inits.

**Change when `initial_pose` is set and valid:**

- Do **not** cluster the target (or use a single “cluster” = full cloud).
- **trans_init** = 4×4 from initial_pose (position + quaternion).
- Optionally try a small number of yaw offsets around initial orientation if desired (e.g. ±15°).
- Run ICP (and optionally FGR with this init) **once** against the full target cloud (or a downsampled version). This avoids wrong cluster and uses the good init.

**When initial_pose is not set:** Keep current behavior (cluster target, FGR+ICP per cluster, centroid init).

**Robustness:** If ICP fails or fitness is very low, fall back to publishing initial_pose as the pose (or skip publish and log).

## 5. Retrieval / FAISS

- **Concept:** Slot = track_id + job_id; retrieval runs on slot’s crop image, returns best object path (FAISS or candidate). Initial pose is tied to the **slot** (from SAM3D+robot), not to the chosen object.
- **Checks:** Ensure indexer runs so `faiss/` has entries; ensure retrieval node uses correct `faiss_index_path` (run dir). We already added logging: “FAISS best match same as candidate”, “using segment PC” vs “using world PC”.
- **Optional:** Add a small doc or launch note: “For cross-slot retrieval, run indexer so current run’s faiss is populated; for cross-run retrieval, point index at a larger asset set.” — **Done:** RUN_PIPELINE.md §8.1 (retrieval verification) and §4 (FAISS indexer) describe indexer and current run; scripts default to current run.
- No code change required for retrieval logic beyond verification.

## 6. End-to-end flow (proposed)

1. **Images + lidar** → lidar_cam_node accumulates points → publishes `/global_lidar_points`; job writer writes jobs per track_id (slot).
2. **Worker** runs SAM3D, writes object (GLB/PLY) + **initial pose** (from SAM3D + odom) to `output/<job_id>/pose.json`.
3. **Injector** sees new output, publishes **SlotReady** (job_id, track_id, candidate path).
4. **Retrieval** uses slot’s rgb.png, queries FAISS, publishes **Sam3dObjectForSlot** (job_id, track_id, best data_path).
5. **Bridge** loads source (chosen object), reads **initial pose** from `output/<job_id>/pose.json`, subscribes to **/global_lidar_points**, publishes **StringIdSrcTarg** (src_pc, targ_pc = global PC, **initial_pose**).
6. **Registration** uses initial_pose as trans_init, registers source to **global** target, publishes **StringIdPose**.
7. **Simple buffer** aggregates → scene_graph.json + scene.glb.

## 7. Implementation order (done)

1. **Worker:** Compute and write initial pose (SAM3D + odom) in pose.json (demo_go2 convention). — **Done:** `ply_frame_utils.initial_pose_from_sam3d_output`, worker writes `initial_position` / `initial_orientation`.
2. **UsdStringIdSrcTargMsg:** Add `geometry_msgs/Pose initial_pose`. — **Done.**
3. **Bridge:** Read initial pose from job dir; when present, use `/global_lidar_points` as target and set `initial_pose`; otherwise keep current target logic. — **Done:** default `registration_target=global`, `world_point_cloud_topic=/global_lidar_points`; bridge reads pose.json and sets `initial_pose`.
4. **Registration:** If `initial_pose` set, use it as trans_init and register to full target (no clustering); else keep current behavior. — **Done:** `_get_pose_from_initial`; fallback to publishing initial_pose when ICP fitness < 0.1.
5. **Docs/config:** Update RUN_PIPELINE or USER_NEXT_STEPS with “initial pose + global registration” and retrieval/FAISS notes. — **Done:** RUN_PIPELINE §8.2, §8.3; USER_NEXT_STEPS quick run guide.

## 8. Resolved / parameters

- **init_odom:** Worker has **`--use-init-odom`**. When set, the first job’s odom is written to `queue_dir/init_odom.json`; all jobs’ pose.json position/orientation are expressed relative to that frame (position = odom − init, orientation = relative quat). Use for “first frame as origin” style.
- **Segment vs global target:** Bridge has **`registration_target`** parameter: **`global`** (default) = use accumulated world point cloud (`/global_lidar_points`) and initial pose from SAM3D+go2; **`segment`** = target from job dir (depth+mask+meta), with fallback to world PC if segment unavailable. Use `segment` for A/B experiments.
- **FAISS:** Best match will often be the candidate until the index is populated; that’s expected. Run the indexer so the run’s `faiss/` has entries for retrieval to return different objects.
