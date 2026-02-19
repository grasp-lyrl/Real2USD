# Phase 3: FAISS Index + Multi-View Capture — Plan

This plan covers (1) building and maintaining a FAISS similarity-search index over SAM3D-generated objects, and (2) capturing multi-view images for those objects so retrieval can match from multiple viewpoints. It feeds into the Phase 3 hybrid retrieval (search index first; on low confidence, enqueue SAM3D).

**Constraints:** Edit only real2sam3d (and custom_message if needed). real2usd is reference only. Pipeline runs in Docker; worker on host; queue at `/data/sam3d_queue`.

**Frame convention (worker):** SAM3D outputs geometry in camera frame. The worker transforms it to **world (odom) frame** then to **object-local** (centroid at origin, axes aligned with world). So `object.ply` is in object-local frame and `pose.json` holds position = world centroid and orientation = identity. Downstream (overlay, FAISS indexer, multi-view) all see a consistent world frame. See `scripts_sam3d_worker/ply_frame_utils.py` and the worker’s post-save transform step.

---

## 1. Goals

- **FAISS index:** As SAM3D creates objects in the queue `output/` directory, add them to a CLIP+FAISS index so they can be retrieved like SimReady assets (by visual similarity).
- **Multi-view:** Prefer multiple views per object (not just the single YOLO crop) for richer matching. Plan a path from single-view (quick) to multi-view (render from 3D).

---

## 2. FAISS Index for SAM3D Objects

### 2.1 Where and format

- **Index location:** Configurable, e.g. `/data/sam3d_faiss` or `/data/FAISS/sam3d`. Same format as real2usd’s CLIPUSDSearch: `index.faiss` + `index.pkl` (metadata: `image_paths`, `usd_paths`, `image_embeddings`).
- **Stored “USD” path:** For generated objects we store the path to the actual asset file: `object.ply` or `object.usd` (so overlay/registration can load it). No need to change real2usd code; the hybrid node in real2sam3d will use this index and publish the same message types.

### 2.2 When to add to the index

- **Trigger:** When a new completed job appears under `queue_dir/output/<job_id>/` (same completion signal as the injector: `pose.json` + `object.ply` or `object.usd`).
- **Who runs indexing:** A dedicated **indexer** process (script or small node) that:
  - Watches `queue_dir/output/` for new job dirs (or is invoked by the worker after each job, or runs on a timer like the injector).
  - For each new job: builds or uses one image per object (see Multi-view below), computes CLIP embeddings, adds entries to the FAISS index and metadata, then saves the index.

### 2.3 Single-view image source (Phase 3a)

- So that each result is self-contained and we don’t rely on `input_processed/`:
  - **Worker change:** When the worker finishes a job, copy the input crop into the output dir: e.g. copy `input/<job_id>/rgb.png` → `output/<job_id>/rgb.png` before moving the job to `input_processed/`. Then `output/<job_id>/` contains: `pose.json`, `object.ply` (or `object.usd`), `rgb.png`.
- **Indexer:** For each new `output/<job_id>/`:
  - Image: `output/<job_id>/rgb.png`
  - Object path: `output/<job_id>/object.ply` or `object.usd`
  - Call `add_image_to_index(image_path, object_path)`, then `save_index(sam3d_faiss_path)`.

### 2.4 Reuse of CLIP + FAISS logic

- **Option A (preferred):** Depend on real2usd’s `scripts_r2u.clipusdsearch_cls.CLIPUSDSearch` (same workspace). Indexer loads or creates index, calls `add_image_to_index`, `save_index`. No duplication of CLIP/FAISS code.
- **Option B:** If real2usd is not importable from real2sam3d, implement a minimal indexer in real2sam3d that uses the same interface (load image, CLIP encode, FAISS add, pkl metadata). Prefer A to avoid drift.

---

## 3. Multi-View Images (Phase 3b)

### 3.1 Why multi-view

- One crop (single view) can be viewpoint-specific; multiple views improve retrieval when the query viewpoint differs.
- Options: (1) render from 3D, (2) re-observation from the pipeline (harder: need to associate detections to generated objects).

### 3.2 Recommended: render from 3D

- **When:** After the worker writes `output/<job_id>/object.ply` (and optionally `object.usd`).
- **How:** A small script (e.g. `render_sam3d_views.py`) that:
  - Takes `output/<job_id>/` (or path to `object.ply`).
  - Loads the mesh (e.g. with `trimesh` or `open3d`); if only point cloud, either skip multi-view or use a simple splat render (optional).
  - Renders N views (e.g. 8 azimuths at 45°, fixed elevation), saves to `output/<job_id>/views/0.png .. N-1.png`.
- **Indexer:** If `output/<job_id>/views/` exists and is non-empty, add **each** view to the index with the **same** object path (`object.ply` or `object.usd`). So one object has N FAISS entries (same `usd_path`); at query time the best match (any view) returns that object. Alternatively, one embedding per object = mean of N view embeddings (one entry per object); mean is simpler for “one result per object” but multiple entries give more chances to match.

**Recommendation:** Add each view as a separate index entry (same `usd_path`). No change to search API; we just get more diverse matches for the same asset.

### 3.3 Fallback

- If `views/` is missing or empty, indexer uses only `output/<job_id>/rgb.png` (single-view). So Phase 3a works without any renderer; Phase 3b adds optional multi-view when the render script is available.

### 3.4 Implementation options for rendering

- **Lightweight:** `trimesh` + `pyglet` offscreen or `trimesh` scene export to images (if available), or `open3d` offscreen rendering.
- **Heavier:** Headless Blender for higher quality (optional later).
- Prefer a single dependency (e.g. `trimesh` or `open3d`) and document it in USER_NEXT_STEPS / README.

---

## 4. Indexer Design (Summary)

- **Inputs:** `queue_dir`, `sam3d_faiss_index_path` (e.g. `/data/sam3d_faiss`), optional “use multi-view” (if `views/` present).
- **Logic:**
  1. Watch `queue_dir/output/` for new job dirs (same completion condition as injector: `pose.json` + `object.ply` or `object.usd`).
  2. For each new job_id not yet indexed:
     - If `output/<job_id>/views/*.png` exists: for each view image, `add_image_to_index(view_path, object_path)`.
     - Else: `add_image_to_index(output/<job_id>/rgb.png, object_path)`.
  3. After each add (or batch): `save_index(sam3d_faiss_index_path)`.
- **Run as:** Standalone script (e.g. `scripts_sam3d_worker/index_sam3d_faiss.py` or under `real2sam3d/scripts_*`) or a ROS2 node that runs the same logic on a timer. Standalone is easier to run on the host (where CLIP/FAISS may be available) without ROS.

---

## 5. Hybrid Retrieval (Phase 3 — already in SAM3D_BUILD_PLAN)

- **Hybrid node** (in real2sam3d): Subscribes to `/usd/CropImgDepth`, runs CLIP + FAISS.
  - **Two indices:** (1) Asset index `faiss_index_path` (SimReady), (2) SAM3D index `sam3d_faiss_index_path`.
  - **Flow:**
    1. Compute CLIP embedding of the crop.
    2. Search asset index; get best score `s_asset`.
    3. If `s_asset >= T`: publish `UsdStringIdPCMsg` with that asset path.
    4. Else search SAM3D index; get best score `s_sam3d`.
    5. If `s_sam3d >= T2`: publish `UsdStringIdPCMsg` (or `UsdStringIdPoseMsg`-compatible path) with that generated object path.
    6. If both below thresholds: do **not** publish; call job-writing logic to enqueue a SAM3D job to `queue_dir/input/`.
- **Parameters:** `faiss_index_path`, `sam3d_faiss_index_path`, `queue_dir`, confidence thresholds `T`, `T2`.

---

## 6. Implementation Order

| Step | Task | Notes |
|------|------|--------|
| 6.1 | Worker: copy `rgb.png` into `output/<job_id>/` after writing result | Keeps result self-contained. |
| 6.2 | Indexer script: watch `output/`, use CLIPUSDSearch (or minimal clone), add single-view (rgb.png) + object path, save index | Can depend on real2usd’s clipusdsearch_cls. |
| 6.3 | Optional: `render_sam3d_views.py` for N views from `object.ply` → `views/*.png` | **Done.** Phase 3b; indexer prefers views/ when present. Requires open3d. |
| 6.4 | Hybrid retrieval node: two-index search + job enqueue on low confidence | Uses both `faiss_index_path` and `sam3d_faiss_index_path`. |

---

## 7. Files to Add/Change (real2sam3d)

- **Worker** (`scripts_sam3d_worker/run_sam3d_worker.py`): Copy `rgb.png` from job dir to `output/<job_id>/rgb.png` before moving job to `input_processed/`.
- **New:** Indexer script (e.g. `scripts_sam3d_worker/index_sam3d_faiss.py` or `real2sam3d/scripts_sam3d_faiss/index_sam3d_faiss.py`): watch output, add to FAISS, save.
- **New (Phase 3b):** `render_sam3d_views.py`: load mesh from `object.ply`, render N views, save under `output/<job_id>/views/`.
- **New:** Hybrid retrieval node (e.g. `hybrid_retrieval_node.py`): subscribe `/usd/CropImgDepth`, two-index search, thresholds, job enqueue.
- **Config:** Document `sam3d_faiss_index_path`, indexer usage, and optional multi-view in `USER_NEXT_STEPS.md` (and optionally in launch args).

---

## 8. Testing

- **Indexer:** Unit test: create a fake `output/<job_id>/` with `rgb.png` + `pose.json` + `object.ply`; run indexer; assert index file exists and contains one entry (and optional views).
- **Hybrid node:** With mock FAISS returning low score, assert job written to `input/` and no UsdStringIdPC published; with high score, assert published and no job (see SAM3D_BUILD_PLAN Phase 3 tests).

Once this plan is confirmed, implementation can proceed in the order above.
