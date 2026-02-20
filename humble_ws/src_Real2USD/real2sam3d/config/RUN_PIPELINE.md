# How to Run the SAM3D Pipeline

Short guide: which terminal (Docker vs host conda), in what order, and how paths are handled.

**Path simplification:** When you start the pipeline with `use_run_subdir:=true` (default), the launch creates a run folder `sam3d_queue/run_YYYYMMDD_HHMMSS` and writes **`sam3d_queue/current_run.json`**. Scripts on the host (worker, indexer, render) **default to using current run** (they read paths from that file). Use **`--no-current-run`** and pass **`--queue-dir`** (and **`--index-path`** for indexer) when you want to point at a specific run or path.

**Shared path:** Use the same directory on host and in Docker (e.g. create on host and mount into the container):

```bash
mkdir -p /data/sam3d_queue
# When running Docker: -v /data/sam3d_queue:/data/sam3d_queue
```

---

## 1. Start the pipeline (Docker)

**Terminal 1 — Docker (ROS2):**

```bash
# In container: source workspace, then launch
source /opt/ros/humble/setup.bash
source install/setup.bash   # from Real2USD workspace root in container

ros2 launch real2sam3d real2sam3d.launch.py
```

- This creates `sam3d_queue/run_YYYYMMDD_HHMMSS/` and writes **`sam3d_queue/current_run.json`**.
- Optional: **no-FAISS mode** (no retrieval, use candidate object only):
  ```bash
  ros2 launch real2sam3d real2sam3d.launch.py no_faiss_mode:=true
  ```

---

## 2. Run the SAM3D worker (host, conda)

**Terminal 2 — Host, conda env `sam3d-objects`:**

```bash
conda activate sam3d-objects
cd /path/to/Real2USD/humble_ws/src_Real2USD/real2sam3d/scripts_sam3d_worker

# Default: uses queue_dir from current_run.json (start launch first)
python run_sam3d_worker.py --sam3d-repo /path/to/sam-3d-objects
```

- **Current run is the default:** no need to pass `--use-current-run`; scripts read `queue_dir` from `current_run.json`. Use **`--no-current-run`** and **`--queue-dir /path`** to target a specific run.
- **`--use-init-odom`**: express poses relative to first job’s odom (writes `init_odom.json`; useful for “first frame as origin”).
- For a single job then exit: add **`--once`**.
- Without SAM3D (placeholder only): add **`--dry-run`** (no conda/SAM3D needed).

---

## 3. (Optional) Multi-view render for FAISS

Renders `output/<job_id>/views/0.png .. N-1.png` from each object so the indexer can use multiple views per object (better retrieval). Run after the worker has written object.glb/object.ply; the indexer will use `views/*.png` when present.

**Terminal 3 — Host, conda with open3d (and trimesh for GLB):**

```bash
conda activate your_env_with_open3d
cd /path/to/Real2USD/humble_ws/src_Real2USD/real2sam3d/scripts_sam3d_worker

# Default: uses queue_dir from current_run.json
python render_sam3d_views.py
```

- **Current run is the default**; use **`--no-current-run`** and **`--queue-dir`** to target a specific run. Scans `output/` for job dirs that have object.glb or object.ply and no `views/` yet.
- One-shot: add **`--once`**. Otherwise it keeps watching.
- Single job: `python render_sam3d_views.py --job-dir /data/sam3d_queue/run_XXX/output/<job_id>` (no current_run needed).

---

## 4. (Optional) FAISS indexer for retrieval

**Terminal 4 — Host, conda env with CLIP/FAISS (e.g. same as real2usd):**

```bash
conda activate your_clip_faiss_env
cd /path/to/Real2USD/humble_ws/src_Real2USD/real2sam3d/scripts_sam3d_worker

# Default: uses queue_dir and faiss_index_path from current_run.json
python index_sam3d_faiss.py
```

- **Current run is the default**; use **`--no-current-run`** and **`--queue-dir`** / **`--index-path`** to override.
- One-shot: add **`--once`**. Otherwise it keeps watching for new jobs.

---

## 5. Summary table

| Step              | Where        | Command / note |
|-------------------|-------------|----------------|
| 1. Launch pipeline| **Docker**  | `ros2 launch real2sam3d real2sam3d.launch.py` |
| 2. SAM3D worker   | **Host conda** | `python run_sam3d_worker.py --sam3d-repo /path/to/sam-3d-objects` (current run default) |
| 3. Multi-view render | **Host conda** | `python render_sam3d_views.py` (optional; needs open3d) |
| 4. FAISS indexer  | **Host conda** | `python index_sam3d_faiss.py` (optional) |

---

## 6. With `--no-current-run` (manual paths)

If you want to point at a specific run or run scripts without current_run.json:

- **Worker:**  
  `python run_sam3d_worker.py --no-current-run --queue-dir /data/sam3d_queue/run_YYYYMMDD_HHMMSS --sam3d-repo /path/to/sam-3d-objects`

- **Multi-view render:**  
  `python render_sam3d_views.py --no-current-run --queue-dir /data/sam3d_queue/run_YYYYMMDD_HHMMSS`

- **Indexer:**  
  `python index_sam3d_faiss.py --no-current-run --queue-dir /data/sam3d_queue/run_YYYYMMDD_HHMMSS --index-path /data/sam3d_faiss`

Or set **`SAM3D_QUEUE_BASE`** (default `/data/sam3d_queue`) so `current_run.json` is looked up there.

---

## 7. What `current_run.json` contains

Written by the launch (when `use_run_subdir:=true`) at `sam3d_queue/current_run.json`:

```json
{
  "queue_dir": "/data/sam3d_queue/run_20260219_123456",
  "base_dir": "/data/sam3d_queue",
  "run_name": "run_20260219_123456",
  "faiss_index_path": "/data/sam3d_queue/run_20260219_123456",
  "created_at": "2026-02-19T12:34:56.789012"
}
```

- **queue_dir**: run directory (input/, output/ live here).
- **faiss_index_path**: Base path for this run’s FAISS index; index files live in `<faiss_index_path>/faiss/` (same as run dir when using run subdir). Used by the indexer and retrieval node. The indexer creates `faiss/` (e.g. `index.faiss`, `index.pkl`) under this path.

---

## 8. Verifying retrieval and registration

If **scene_graph.json** shows (1) every object’s `id` matching the object’s job in `data_path`, or (2) objects congregating at a few positions, check the following.

### 8.1 Retrieval: “FAISS best match is same as candidate”

- **Meaning:** The chosen object for each slot is the one that was generated for that slot (candidate), not a different object from the index.
- **Logs to look for:** `[retrieval] job_id=... FAISS best match is same as candidate` or `FAISS index empty` or `using candidate`.
- **Causes:**  
  - FAISS index is empty (indexer not run, or run after jobs were published).  
  - Index only contains objects from this run; the best CLIP match for the slot’s crop is often that same object.  
  - No `rgb.png` for the slot, or **FAISS load failed** → node falls back to candidate.
- **“FAISS load failed”:** The node logs one **ERROR** with the exception and the path it tried (`<faiss_index_path>/faiss/index.faiss`). Each slot then logs a **WARN** with the same reason. Common causes: (1) **Index not built** — run `python index_sam3d_faiss.py` (defaults to current run) on the host so `<run_dir>/faiss/index.faiss` exists; (2) **CLIP/FAISS not in Docker** — retrieval runs inside the container; if the image has no CLIP/FAISS, use candidate-only mode (`no_faiss_mode:=true`) or run retrieval on the host with CLIP available.
- **What to do:**  
  - Run the indexer so the run’s `faiss/` has entries (`python index_sam3d_faiss.py`; current run is default).  
  - For cross-slot retrieval, index needs objects from earlier runs or a pre-built index; otherwise “best match” will often be the candidate.

### 8.2 Registration: target and initial pose (default: global + SAM3D+go2 init)

- **Meaning:** By default registration uses the **global accumulated point cloud** and an **initial pose** (SAM3D + go2 odom) so ICP starts near the correct pose. If you set `registration_target:=segment`, the bridge uses the segment from the job dir instead; if that segment is missing it falls back to world PC.
- **Logs to look for:**  
  - Good: `[registration target] job_id=... using segment PC from job dir (N pts) — correct association`.  
  - Bad: `[registration target] job_id=... segment unavailable (reason); using world PC — poses may congregate`.
- **Reasons for “segment unavailable”:** Missing `depth.npy`, `mask.png`, or `meta.json` in `output/<job_id>/`; invalid `crop_bbox`/`camera_info`/`odometry` in meta; mask has no valid depth pixels; too few points after filters.
- **What to do:**  
  - The worker copies `depth.npy`, `mask.png`, `meta.json` from input to output for each job; ensure the job writer wrote them and the worker completed.  
  - Check `output/<job_id>/` for each job_id that shows “segment unavailable” and fix or add the segment files so registration uses the correct per-slot target.
- **Default:** **Global** accumulated point cloud (`/global_lidar_points`) and **initial pose** from SAM3D+go2 (worker writes `initial_position` / `initial_orientation` in pose.json). Registration runs ICP from that init. **Parameter** `registration_target` (`global` | `segment`): set **`segment`** to use per-slot segment PC instead (for experiments).
