# What You (the User) Need to Do Next

**For a new Cursor agent session:** Point the agent at `config/AGENT_HANDOFF.md` for where we left off and what to do next.

---

## Architecture (best practice)

- **Pipeline:** Runs **only in Docker** (ROS2 Humble, real2sam3d). No conda, no PyTorch/SAM3D. Small image, fast rebuilds.
- **SAM3D worker:** Runs **on the host** in a conda env (sam3d-objects). Keep the sam-3d-objects repo outside the Real2USD repo (e.g. `~/repos/sam-3d-objects`).
- **Handoff:** Queue directory is **`/data/sam3d_queue`** on both host and container (create on host and mount into the container, e.g. `-v /data/sam3d_queue:/data/sam3d_queue`). Pipeline writes `input/`; worker on the host reads and writes `output/`; injector sees `output/` and publishes to ROS.

---

**Quick run guide:** See **config/RUN_PIPELINE.md** for step-by-step commands (which terminal: Docker vs host conda) and **`--use-current-run`** so worker/indexer read paths from `current_run.json` and you don’t set paths manually.

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

4. **Worker output and frame:** The worker writes **object.ply** (gaussian splat) and **object.glb** (mesh, when the inference returns `output["glb"]`) in **object-local frame** (centroid at origin, world axes), plus `pose.json` and a copy of `rgb.png`. Both assets are transformed from SAM3D’s camera frame to world using `meta["odometry"]`; the frame convention matches [demo_go2.py](https://github.com/christopher-hsu/sam-3d-objects/blob/main/demo_go2.py). When both are present, the same world centroid (from the PLY) is used so pose and origin are consistent. For a camera offset (e.g. go2), include in `meta.json`: `"R_world_to_cam"` and `"t_world_to_camera"`. The worker needs **plyfile** for the PLY transform; if the inference provides a trimesh as `output["glb"]`, it is exported as object.glb (no extra deps beyond the inference API).

5. **FAISS indexer (Phase 3):** To build a similarity-search index over SAM3D-generated objects, run the indexer where CLIP and FAISS are installed (e.g. same host as worker or a conda env with `clip`, `faiss-cpu`/`faiss-gpu`, `torch`). It watches `queue_dir/output/` and adds each new job’s `rgb.png` (or `views/*.png` if present) + object path to the index. You do **not** need to create a faiss folder: the script creates `<index-path>/faiss/` and puts all outputs there (`index.faiss`, `index.pkl`, `indexed_jobs.txt`). Example:
   ```bash
   cd /path/to/Real2USD/humble_ws/src_Real2USD/real2sam3d/scripts_sam3d_worker
   python index_sam3d_faiss.py --queue-dir /data/sam3d_queue --index-path /data/sam3d_faiss --once
   ```
   Or run continuously: omit `--once` so it scans every `--watch-interval` seconds (default 5). Requires **real2usd** in the workspace (for `scripts_r2u/clipusdsearch_cls.py`).

6. **Multi-view rendering (Phase 3b):** For richer FAISS retrieval, render multiple views **with appearance** (vertex colors / materials) into `output/<job_id>/views/0.png .. N-1.png`. The script prefers **object.glb** when present (mesh + materials), then **object.ply** (mesh or point cloud; vertex colors used when available). So retrieval matches on look, not just shape. Requires **open3d** (`pip install open3d`). For GLB: **trimesh** (`pip install trimesh`). Example:
   ```bash
   python render_sam3d_views.py --job-dir /data/sam3d_queue/output/<job_id>
   ```
   Or scan the queue: `python render_sam3d_views.py --queue-dir /data/sam3d_queue --once`. Uses 8 azimuth views by default. For best results, export **object.glb** from SAM3D (e.g. in the worker or in demo_go2) so views are rendered with full appearance.

---

## 3. Checklist summary

| Step | What to do |
|------|------------|
| Queue dir | `mkdir -p /data/sam3d_queue` on the host. Mount into the container: `-v /data/sam3d_queue:/data/sam3d_queue`. Same path on both sides. |
| Pipeline | Docker only. Build: `colcon build --packages-select custom_message real2sam3d`, then `ros2 launch real2sam3d real2sam3d.launch.py` (no worker in launch). |
| Worker | On host: `conda activate sam3d-objects` then `python run_sam3d_worker.py --queue-dir /data/sam3d_queue`. |
| FAISS indexer | On host (CLIP/FAISS env): `python index_sam3d_faiss.py --queue-dir /data/sam3d_queue --index-path /data/sam3d_faiss` (or `--once`). |
| Multi-view render | Optional: `python render_sam3d_views.py --queue-dir /data/sam3d_queue --once` (needs open3d). Indexer then uses `views/*.png` when present. |
| Dry-run (no conda) | Inside Docker: `ros2 launch real2sam3d real2sam3d.launch.py run_sam3d_worker:=true`. |
| Phase 3 retrieval | **sam3d_injector_node** publishes **slot ready** on `/usd/SlotReady` (job_id, track_id, candidate_data_path). **sam3d_retrieval_node** loads the slot's crop image (`output/<job_id>/rgb.png`), queries the SAM3D FAISS index (CLIP), and publishes the **best object** for that slot on `/usd/Sam3dObjectForSlot`. If FAISS is empty or unavailable, the candidate (newly generated) object is used. |
| No-FAISS mode | To run without FAISS/retrieval: **no_faiss_mode:=true**. The retrieval node is not started and the injector publishes `/usd/Sam3dObjectForSlot` directly with the candidate object, so the bridge and registration run as usual. Use when you don't have a FAISS index or want to skip CLIP lookup. |
| Registration (ICP) | **sam3d_glb_registration_bridge_node** subscribes to `/usd/Sam3dObjectForSlot` (job_id, track_id, data_path). It loads source geometry from `data_path` and **target segment PC from the slot job** (`output/<job_id>/` or `input_processed/<job_id>/`), publishes `/usd/StringIdSrcTarg`. **registration_node** runs FGR+ICP and publishes the **final** pose on `/usd/StringIdPose` (used by usd_buffer_node and overlay). Disable bridge with `glb_registration_bridge:=false` if not needed. |
| Simple scene buffer | Optional: **simple_scene_buffer_node** subscribes to `/usd/StringIdPose` and writes a single **scene_graph.json** (id, data_path, position, orientation per object) and a **joint scene.glb** (all objects merged at their poses). Enable with `simple_scene_buffer:=true`. Outputs go to the same directory as the rest of the pipeline (queue/run dir, e.g. `sam3d_queue/run_YYYYMMDD_HHMMSS/`). Set `output_dir` param to override. Requires **trimesh** for GLB export. |
| Dedup | Job writer skips same object within 60 s (track_id) or 0.5 m (label+position). Set `dedup_track_id_sec:=0` and `dedup_position_m:=0` to disable. |
| Tests | Inside Docker after build: `python3 -m pytest src_Real2USD/real2sam3d/test/test_sam3d_*.py -v`. |

---

## 4. What still needs to be done (code-side)

From the build plan (config/SAM3D_BUILD_PLAN.md):

- **Phase 3:** Hybrid retrieval (CLIP confidence → trigger SAM3D when low).
- **Phase 4:** Worker outputs `.usd` and world-frame transform.
- **Phase 5:** Deduplication of repeated generated objects.
- **Phase 6:** Scene USDA and physics schema for generated objects.
