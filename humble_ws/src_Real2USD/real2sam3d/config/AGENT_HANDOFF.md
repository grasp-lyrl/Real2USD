# SAM3D Pipeline — Session Handoff

**For a new Cursor/agent session:** Read this file first. It documents what was implemented and where to look so you can continue without re-discovering the codebase.

---

## 1. Design and run docs (read these to understand the pipeline)

| Document | Purpose |
|----------|--------|
| **`config/PIPELINE_INITIAL_POSE_DESIGN.md`** | **Primary design.** Initial pose (SAM3D+go2), global vs segment target, messages, registration flow, retrieval, implementation order (all marked done). Reference this when changing registration, worker pose, or bridge. |
| **`config/RUN_PIPELINE.md`** | How to run: Docker vs host, commands, current run default, `--no-current-run`, FAISS indexer, verification (§8). |
| **`config/USER_NEXT_STEPS.md`** | Architecture (pipeline in Docker, worker on host), quick run, checklist, what’s still to do. |

**If the user says “continue the pipeline” or “fix registration/initial pose”:** Start by reading **`config/PIPELINE_INITIAL_POSE_DESIGN.md`** and the “Key files” section below.

---

## 2. What was implemented (this session)

- **Default: global registration + initial pose**
  - Registration **target** default = **global** (accumulated `/global_lidar_points`); **initial pose** from SAM3D+go2 is used as ICP starting point.
  - Worker writes **`initial_position`** and **`initial_orientation`** to `output/<job_id>/pose.json` (see `ply_frame_utils.initial_pose_from_sam3d_output`).
  - Bridge reads them and sets **`UsdStringIdSrcTargMsg.initial_pose`**; registration node uses it as `trans_init` when set (no clustering, single ICP). Low ICP fitness (<0.1) → publish initial_pose as result.

- **Scripts default to current run**
  - Worker, indexer, render_sam3d_views **use current run by default** (read `current_run.json`). Use **`--no-current-run`** and **`--queue-dir`** (and **`--index-path`** for indexer) to override.

- **Parameters**
  - **Worker:** **`--use-init-odom`** — poses in pose.json relative to first job’s odom (`init_odom.json`).
  - **Bridge:** **`registration_target`** = **`global`** (default) or **`segment`** (per-slot segment PC for experiments). **`world_point_cloud_topic`** default **`/global_lidar_points`**.

- **Other**
  - **Simple scene buffer node:** `simple_scene_buffer_node.py` — subscribes `/usd/StringIdPose`, writes **`scene_graph.json`** and **`scene.glb`** (joint GLB) to queue/run dir. Enable with **`simple_scene_buffer:=true`**.
  - **FAISS per run:** `current_run.json` and launch set **`faiss_index_path`** to the **run dir** (index under `<run_dir>/faiss/`).
  - **Bridge:** Segment target uses **`camera_info.K`** or **`camera_info.k`** (job writer uses `K`).
  - **Registration node:** Guards so FGR is not called with 0 points (avoids Open3D “low must be < high”); when **initial_pose** is set, **`_get_pose_from_initial`** runs (single ICP, fallback to initial_pose if fitness < 0.1).

---

## 3. Key files (where to edit)

| Area | Files |
|------|--------|
| **Initial pose (worker)** | `scripts_sam3d_worker/ply_frame_utils.py` (`initial_pose_from_sam3d_output`, `GO2_*`), `scripts_sam3d_worker/run_sam3d_worker.py` (writes `initial_position` / `initial_orientation` to pose.json). |
| **Bridge (target + initial_pose)** | `real2sam3d/sam3d_glb_registration_bridge_node.py` — `registration_target`, `world_point_cloud_topic`, reads pose.json, sets `reg_msg.initial_pose`. |
| **Registration** | `real2sam3d/registration_node.py` — `callback_src_targ` (checks `initial_pose`), `_get_pose_from_initial`, `get_pose` (clustering path when no initial_pose). |
| **Messages** | **`custom_message/msg/UsdStringIdSrcTargMsg.msg`** — has **`geometry_msgs/Pose initial_pose`**. Rebuild **custom_message** after editing. |
| **Launch** | `launch/real2sam3d.launch.py` — bridge params (`world_point_cloud_topic`, etc.), simple_scene_buffer, run dir, faiss_index_path. |
| **Current run / scripts** | `scripts_sam3d_worker/current_run.py` (`resolve_queue_and_index`). Worker/indexer/render use **`--no-current-run`** to bypass. |
| **Scene buffer** | `real2sam3d/simple_scene_buffer_node.py` — writes scene_graph.json + scene.glb. |
| **Retrieval / FAISS** | `real2sam3d/sam3d_retrieval_node.py` (logging), `scripts_sam3d_worker/index_sam3d_faiss.py`. |

---

## 4. Build and run (quick ref)

- **Build (after changing .msg or Python nodes):**  
  `colcon build --packages-select custom_message real2sam3d` then `source install/setup.bash` (in Docker or workspace where you run nodes).

- **Pipeline (Docker):**  
  `ros2 launch real2sam3d real2sam3d.launch.py`  
  (Creates run dir, writes `current_run.json`; bridge uses `/global_lidar_points` and reads initial_pose from job dir.)

- **Worker (host, conda):**  
  `python run_sam3d_worker.py --sam3d-repo /path/to/sam-3d-objects`  
  (Uses current run by default; no need for `--queue-dir` if launch already ran.)

- **Optional:**  
  `simple_scene_buffer:=true` for scene_graph.json + scene.glb; **`registration_target:=segment`** for segment-based registration experiments.

---

## 5. If you need to…

- **Change how initial pose is computed** → `config/PIPELINE_INITIAL_POSE_DESIGN.md` §1, then `ply_frame_utils.initial_pose_from_sam3d_output` and worker’s pose.json write.
- **Change registration target or fallback** → Design §2 and §8, then `sam3d_glb_registration_bridge_node.py` (`registration_target`, segment/global branch).
- **Change registration when initial_pose is set** → Design §4, then `registration_node.py` (`_get_pose_from_initial`, fitness threshold 0.1).
- **Change scripts (worker/indexer/render) default or args** → `run_sam3d_worker.py`, `index_sam3d_faiss.py`, `render_sam3d_views.py` (default current run; `--no-current-run`), and `current_run.py`.
- **Add or change a message** → Edit `.msg` under `custom_message/msg/`, then rebuild **custom_message** and **real2sam3d** (nodes that use the message).

---

## 6. Architecture (unchanged)

- **Pipeline in Docker only** (ROS2 Humble, no SAM3D). **Worker on host** in conda (sam3d-objects). Queue dir **`/data/sam3d_queue`** on both (mount into container). See **USER_NEXT_STEPS.md**.
- **Editable:** `real2sam3d` and `custom_message` only. **Reference only:** `real2usd` (do not change for this integration).

---

## 7. Tests

- **Inside Docker (after build):**  
  `python3 -m pytest src_Real2USD/real2sam3d/test/test_sam3d_*.py -v`  
- Tests cover job writer, worker, track_id, dedup. No tests were added in this session for initial pose or registration; those are manual/integration.

---

*Last updated to reflect: default global registration + initial pose (SAM3D+go2), scripts default to current run, init_odom and registration_target params, simple scene buffer, FAISS per run, camera_info.K, registration FGR/initial_pose robustness.*
