# SAM3D Integration Build Plan

Methodical phases for building the pipeline in `context_sam3d_integration.txt`. Each phase has deliverables and a test users can run to verify that portion of the code.

**Main pipeline is run via ROS2 launch** (tests are pytest). Single command:
```bash
source install/setup.bash
ros2 launch real2sam3d real2sam3d.launch.py
```
Optional: `run_sam3d_worker:=true` to run the SAM3D worker process in the same launch (e.g. for dry-run). See context_sam3d_integration.txt "RUNNING THE FULL PIPELINE".

---

## Phase 0: Foundation (DONE)

**Deliverables:** Job writer node, worker script, crop_bbox in CropImgDepthMsg, disk queue layout.

**Tests:**
- Phase 1 tests cover job writer and worker.

---

## Phase 1: Job Writer + Worker Dry-Run (Current)

**Goal:** Confirm ROS2 → disk → worker path without SAM3D installed.

**Deliverables:**
- `sam3d_job_writer_node` writes valid jobs (rgb.png, mask.png, depth.npy, meta.json).
- `run_sam3d_worker.py --once --dry-run` reads a job and writes output/pose.json + placeholder object.ply.
- Pure function `write_sam3d_job(...)` so job-writing logic can be unit-tested without ROS.

**Tests (run from repo root or humble_ws):**
```bash
cd humble_ws && source install/setup.bash
python3 -m pytest src_Real2USD/real2sam3d/test/test_sam3d_job_writer.py -v
python3 -m pytest src_Real2USD/real2sam3d/test/test_sam3d_worker.py -v
```

- **test_sam3d_job_writer:** Builds a synthetic job (numpy rgb/depth/seg_pts + meta), calls write_sam3d_job, asserts output dir contains rgb.png, mask.png, depth.npy, meta.json with expected structure.
- **test_sam3d_worker:** Creates a fixture job dir under a temp path, runs worker's load_job and process_one_job(..., dry_run=True), asserts output dir has pose.json and object.ply.

**User manual check:** Run pipeline (lidar_cam + sam3d_job_writer), then `python scripts_sam3d_worker/run_sam3d_worker.py --queue-dir /data/sam3d_queue --once --dry-run`; confirm output/<job_id>/ exists.

---

## Phase 2: Generated-USD Injector Node (DONE)

**Goal:** When the worker writes a result, a ROS2 node publishes it so usd_buffer_node and overlay see it.

**Deliverables:**
- `sam3d_injector_node`: watches `queue_dir/output/` for new job result dirs (pose.json + object.ply or object.usd). On new result: read pose.json and object path, publish `UsdStringIdPoseMsg` on `/usd/StringIdPose` (data_path, id=track_id, pose).
- Included in `real2sam3d.launch.py` (sam3d_injector:=true by default). Optional `run_sam3d_worker:=true` runs the worker process in the same launch so the full pipeline can be exercised with one `ros2 launch` command.

**Tests:**
- **test_sam3d_injector:** Unit test: create output/<job_id>/ with pose.json and placeholder object.ply; run injector logic (or spin node briefly); assert one UsdStringIdPoseMsg was published with correct data_path and pose.

**User manual check:** `ros2 launch real2sam3d real2sam3d.launch.py run_sam3d_worker:=true` (with camera/lidar); after worker writes output, check `ros2 topic echo /usd/StringIdPose`.

---

## Phase 3: Hybrid Decision (Trigger SAM3D on Low Confidence)

**Goal:** When CLIP retrieval confidence is below threshold (or no match), enqueue a SAM3D job instead of publishing UsdStringIdPC.

**Detailed plan (FAISS index for SAM3D objects + multi-view capture):** See `config/PHASE3_FAISS_MULTIVIEW_PLAN.md`.

**Deliverables:**
- Port or add retrieval (CLIP + FAISS) into real2sam3d, or a thin "hybrid_retrieval_node" that:
  - Subscribes to `/usd/CropImgDepth`.
  - Runs CLIP embedding + FAISS search (and optional Gemini).
  - If best score >= T: publish UsdStringIdPCMsg (existing path to registration).
  - If best score < T or no label match: call same job-writing logic as job writer (write to queue_dir/input/), do not publish UsdStringIdPC.
- Parameter for confidence threshold T and queue_dir.

**Tests:**
- **test_sam3d_hybrid_retrieval:** With a mock FAISS/CLIP that returns low score, assert no UsdStringIdPC is published and one job is written to input/. With high score, assert UsdStringIdPC is published and no job written (or configurable).

**User manual check:** Run full stack with retrieval; point camera at "unseen" object; confirm input/ gets a job and (after worker) output/ gets result and injector publishes pose.

---

## Phase 4: Worker Full Path (SAM3D + GLB→USD)

**Goal:** Worker runs real SAM3D inference and produces .usd for the injector.

**Deliverables:**
- In worker: after SAM3D output (e.g. gaussian splat .ply or mesh), convert to .usd (e.g. via pxr/Usd or trimesh+usd-core), save to output/<job_id>/object.usd.
- Transform mesh/splat from camera frame to world frame using meta["odometry"]; write final pose to pose.json (or keep odom as pose if registration is skipped).
- Injector reads object.usd path from output dir (and pose.json).

**Tests:**
- Still use --dry-run for CI; optional test that runs real SAM3D if env and checkpoint exist (marked slow/optional).

**User manual check:** Run worker without --dry-run in sam3d-objects env; confirm output/<job_id>/object.ply and object.usd; run injector and usd_buffer; confirm object appears in buffer/overlay.

---

## Phase 5: Deduplication (Repeated Objects)

**Goal:** Avoid duplicate generated objects for the same physical object (same label + overlapping 3D).

**Deliverables:**
- In injector (or usd_buffer_node): before publishing a new UsdStringIdPoseMsg for a generated object, check existing buffer for same label + 3D IoU above threshold; if match, skip or replace by confidence.
- Or: injector publishes all; usd_buffer_node already clusters by position/label—ensure generated objects participate in same clustering and conflict resolution.

**Tests:**
- **test_sam3d_deduplication:** Inject two pose messages with same label and nearby positions; assert only one is kept (or lower-confidence one dropped) according to policy.

**User manual check:** Run pipeline on scene with one object seen multiple times; confirm only one generated asset in buffer.

---

## Phase 6: Scene USDA and Physics

**Goal:** Generated objects in the main scene with SemanticsAPI and PhysicsCollisionAPI.

**Deliverables:**
- When saving or composing the main scene (e.g. hallway.usda), include generated objects as Xform with payload to generated .usd, SemanticsAPI, PhysicsCollisionAPI (convex hull approximation).
- Schema as in context_sam3d_integration.txt section 8.

**Tests:**
- **test_sam3d_usda_schema:** Generate a minimal USDA string for one generated object; parse or regex-assert it contains expected apiSchemas and payload path.

**User manual check:** Export scene from usd_buffer or overlay; open in Isaac Sim; confirm generated objects are solid and have semantics.

---

## Test Index (Quick Reference)

| Test file                    | Phase | What it verifies                          |
|-----------------------------|-------|-------------------------------------------|
| test_sam3d_job_writer.py    | 1     | Job writer produces valid job files       |
| test_sam3d_worker.py        | 1     | Worker loads job, dry-run writes output   |
| test_sam3d_injector.py     | 2     | Injector publishes UsdStringIdPoseMsg     |
| test_sam3d_hybrid_retrieval.py | 3  | Low confidence → job written, no StringIdPC |
| test_sam3d_deduplication.py| 5     | Duplicate objects deduplicated            |
| test_sam3d_usda_schema.py  | 6     | Generated object USDA has correct schema  |

**Run all SAM3D tests:**
```bash
cd humble_ws && source install/setup.bash
python3 -m pytest src_Real2USD/real2sam3d/test/test_sam3d_*.py -v
```

---

## Updating This Plan

As you implement, adjust phase order if needed and add new tests. Keep `context_sam3d_integration.txt` in sync with design decisions and file locations.

**Agent handoff:** For a new Cursor session, read `config/AGENT_HANDOFF.md` for current status and next steps.
