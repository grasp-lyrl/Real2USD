# SAM3D Integration — Where We Left Off

**Read this first when continuing SAM3D integration work in real2sam3d.** It summarizes what’s done, what’s next, and how to run things.

---

## Current status (as of last session)

- **Phases 0, 1, 2 are done** (foundation, job writer + worker dry-run, injector). Job writer dedup (track_id + position+label) is implemented and tested. All pytest tests pass.
- **Next:** Phase 3 (hybrid retrieval: low CLIP confidence → enqueue SAM3D job), then Phase 4 (worker full path: real SAM3D + GLB→USD), Phase 5 (dedup in injector/buffer if needed), Phase 6 (scene USDA/physics).

---

## What’s implemented

| Component | Location | Notes |
|-----------|----------|--------|
| Context & plan | `config/context_sam3d_integration.txt`, `config/SAM3D_BUILD_PLAN.md` | Design rules, pipeline overview, run dir, track_id/dedup |
| User steps | `config/USER_NEXT_STEPS.md` | Pipeline in Docker only; worker on host (conda); queue dir mount |
| Launch | `launch/real2sam3d.launch.py` | lidar_cam, registration, usd_buffer; optional job_writer, injector, run_sam3d_worker (dry-run); per-run dir `run_YYYYMMDD_HHMMSS` |
| Job writer | `real2sam3d/sam3d_job_writer_node.py` | Subscribes `/usd/CropImgDepth`, writes jobs to `queue_dir/input/<job_id>/`. Dedup via `Sam3dDedupState` (track_id + position+label). |
| Injector | `real2sam3d/sam3d_injector_node.py` | Watches `queue_dir/output/`, publishes `UsdStringIdPoseMsg` on `/usd/StringIdPose` |
| Worker | `scripts_sam3d_worker/run_sam3d_worker.py` | Reads `input/`, runs SAM3D or `--dry-run` (writes pose.json + placeholder object.ply) |
| Track ID helper | `scripts_r2s3d/track_id_utils.py` | `track_ids_from_boxes_id()` — no ultralytics dep; used by segment_cls |
| Tests | `test/test_sam3d_*.py`, `test/README_TESTS.md` | Job writer, worker, track_id + dedup (`Sam3dDedupState`). Run **inside Docker** after build. |

---

## How to run (quick ref)

- **Build (in Docker):**  
  `cd /home/me && source /opt/ros/humble/setup.bash && colcon build --packages-select custom_message real2sam3d && source install/setup.bash`

- **Tests:**  
  `python3 -m pytest src_Real2USD/real2sam3d/test/test_sam3d_*.py -v`  
  (Use `python3 -m pytest`; run from workspace root after `source install/setup.bash`.)

- **Pipeline (dry-run, no SAM3D):** Inside Docker: `ros2 launch real2sam3d real2sam3d.launch.py run_sam3d_worker:=true` (camera + lidar + odom or bag).

- **Pipeline + real SAM3D:** Docker: mount queue dir `-v /data/sam3d_queue:/data/sam3d_queue`, run launch without worker. On host: `conda activate sam3d-objects` then `python run_sam3d_worker.py --queue-dir /data/sam3d_queue`.

- **Queue dir:** `/data/sam3d_queue` on both host and container. Create on host: `mkdir -p /data/sam3d_queue`; mount into container. Per-run subdir optional (`use_run_subdir:=false` for single folder).

---

## What to do next (in order)

1. **Phase 3 — Hybrid retrieval**  
   When CLIP/FAISS confidence is below threshold (or no match), enqueue a SAM3D job instead of publishing UsdStringIdPC. See `config/SAM3D_BUILD_PLAN.md` Phase 3 for deliverables and test idea.

2. **Phase 4 — Worker full path**  
   Worker runs real SAM3D, converts output to USD, writes `object.usd`; injector reads it. See build plan Phase 4.

3. **Phase 5 — Dedup (injector/buffer)**  
   Avoid duplicate generated objects for same physical object (same label + nearby 3D). Job writer already dedups enqueue; may need injector or usd_buffer policy. See Phase 5.

4. **Phase 6 — Scene USDA**  
   Generated objects in main scene with SemanticsAPI and PhysicsCollisionAPI. See Phase 6.

---

## Architecture

- **Pipeline in Docker only** (ROS2, no SAM3D). **Worker on host** in a conda env (sam3d-objects). Queue dir: `/data/sam3d_queue` on both host and container (mount into container). See USER_NEXT_STEPS.md.
- **sam-3d-objects:** Keep **outside** the Real2USD repo; install and run on the host only.

## Edits allowed

- **Editable:** Only `real2sam3d` and `custom_message` (for integration).  
- **Reference only:** `real2usd` — do not change for this integration.

---

## File map (real2sam3d)

- Config: `config/context_sam3d_integration.txt`, `config/SAM3D_BUILD_PLAN.md`, `config/USER_NEXT_STEPS.md`, **`config/AGENT_HANDOFF.md`** (this file)
- Launch: `launch/real2sam3d.launch.py`
- Nodes: `real2sam3d/sam3d_job_writer_node.py`, `real2sam3d/sam3d_injector_node.py`, `real2sam3d/lidar_cam_node.py`
- Worker: `scripts_sam3d_worker/run_sam3d_worker.py`
- Segment / track_id: `scripts_r2s3d/segment_cls.py`, `scripts_r2s3d/track_id_utils.py`
- Tests: `test/test_sam3d_job_writer.py`, `test/test_sam3d_worker.py`, `test/test_sam3d_track_id_and_dedup.py`, `test/README_TESTS.md`
