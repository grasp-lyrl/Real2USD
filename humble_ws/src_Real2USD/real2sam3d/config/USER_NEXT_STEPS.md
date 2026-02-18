# What You (the User) Need to Do Next

**For a new Cursor agent session:** Point the agent at `config/AGENT_HANDOFF.md` for where we left off and what to do next.

---

## Architecture (best practice)

- **Pipeline:** Runs **only in Docker** (ROS2 Humble, real2sam3d). No conda, no PyTorch/SAM3D. Small image, fast rebuilds.
- **SAM3D worker:** Runs **on the host** in a conda env (sam3d-objects). Keep the sam-3d-objects repo outside the Real2USD repo (e.g. `~/repos/sam-3d-objects`).
- **Handoff:** Queue directory is **`/data/sam3d_queue`** on both host and container (create on host and mount into the container, e.g. `-v /data/sam3d_queue:/data/sam3d_queue`). Pipeline writes `input/`; worker on the host reads and writes `output/`; injector sees `output/` and publishes to ROS.

---

## 1. Confirm the pipeline without SAM3D (dry-run, inside Docker)

Run the full path (job writer → worker → injector) **without** the SAM3D conda env by using the worker in **dry-run** mode inside the container.

**Inside your Docker container:**

1. **Build the workspace:**
   ```bash
   cd /home/me && source /opt/ros/humble/setup.bash
   colcon build --packages-select custom_message real2sam3d
   source install/setup.bash
   ```

2. **Queue directory:** Create once: `mkdir -p /data/sam3d_queue`. Each launch can create a subdir `run_YYYYMMDD_HHMMSS` (default); use `use_run_subdir:=false` to use a single folder.

3. **Run pipeline with worker in dry-run** (no conda or SAM3D needed):
   ```bash
   ros2 launch real2sam3d real2sam3d.launch.py run_sam3d_worker:=true
   ```
   With camera + lidar + odom (or a bag), jobs go to `input/`, worker writes placeholder `pose.json` + `object.ply`, injector publishes to `/usd/StringIdPose`.

4. **Optional — run tests:**
   ```bash
   python3 -m pytest src_Real2USD/real2sam3d/test/test_sam3d_*.py -v
   ```

---

## 2. Real SAM3D: worker on host (conda env)

**Setup:** Install sam3d-objects **on the host** (outside Docker) in a conda env. Keep the repo outside Real2USD (e.g. `~/repos/sam-3d-objects`). For **RTX 5090** (sm_120), use the procedure from [sam-3d-objects issue #15](https://github.com/facebookresearch/sam-3d-objects/issues/15): PyTorch 2.8+cu128, then build pytorch3d and kaolin from source. Download `sam3d-objects-single.yml` from that issue; pin pytorch3d (e.g. V0.7.8) and kaolin (v0.18.0). After `conda env update`, run `conda deactivate` and `conda activate sam3d-objects` before building. Verify with `python demo.py` in the sam-3d-objects repo (after getting checkpoints from HuggingFace).

**Running the pipeline with real SAM3D:**

1. **On the host:** Create the queue directory: `mkdir -p /data/sam3d_queue`.

2. **Start the pipeline container** with that path mounted:
   ```bash
   docker run ... -v /data/sam3d_queue:/data/sam3d_queue ...
   ```
   Inside the container:
   ```bash
   source install/setup.bash
   ros2 launch real2sam3d real2sam3d.launch.py
   ```
   Do **not** set `run_sam3d_worker:=true` (worker runs on the host). Jobs appear under `/data/sam3d_queue/input/` on the host.

3. **On the host**, in a second terminal (conda env + Real2USD available):
   ```bash
   conda activate sam3d-objects
   cd /path/to/Real2USD/humble_ws/src_Real2USD/real2sam3d/scripts_sam3d_worker
   python run_sam3d_worker.py --queue-dir /data/sam3d_queue --sam3d-repo /path/to/sam-3d-objects
   ```
   Replace `/path/to/sam-3d-objects` with the path to your sam-3d-objects repo (the directory that contains `notebook/` and `checkpoints/`).
   If you use per-run subdirs, pass the same run dir: `--queue-dir /data/sam3d_queue/run_YYYYMMDD_HHMMSS`. The injector in the container will see new results in `output/` and publish them to `/usd/StringIdPose`.

---

## 3. Checklist summary

| Step | What to do |
|------|------------|
| Queue dir | `mkdir -p /data/sam3d_queue` on the host. Mount into the container: `-v /data/sam3d_queue:/data/sam3d_queue`. Same path on both sides. |
| Pipeline | Docker only. Build: `colcon build --packages-select custom_message real2sam3d`, then `ros2 launch real2sam3d real2sam3d.launch.py` (no worker in launch). |
| Worker | On host: `conda activate sam3d-objects` then `python run_sam3d_worker.py --queue-dir /data/sam3d_queue`. |
| Dry-run (no conda) | Inside Docker: `ros2 launch real2sam3d real2sam3d.launch.py run_sam3d_worker:=true`. |
| Dedup | Job writer skips same object within 60 s (track_id) or 0.5 m (label+position). Set `dedup_track_id_sec:=0` and `dedup_position_m:=0` to disable. |
| Tests | Inside Docker after build: `python3 -m pytest src_Real2USD/real2sam3d/test/test_sam3d_*.py -v`. |

---

## 4. What still needs to be done (code-side)

From the build plan (config/SAM3D_BUILD_PLAN.md):

- **Phase 3:** Hybrid retrieval (CLIP confidence → trigger SAM3D when low).
- **Phase 4:** Worker outputs `.usd` and world-frame transform.
- **Phase 5:** Deduplication of repeated generated objects.
- **Phase 6:** Scene USDA and physics schema for generated objects.
