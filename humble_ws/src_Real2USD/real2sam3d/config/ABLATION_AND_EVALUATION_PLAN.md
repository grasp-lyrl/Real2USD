# Ablation and Evaluation Plan

This document outlines how to run ablations (sensor, pointmap, retrieval, registration) and how to rewrite the evaluation scripts for 3D bboxes, oriented boxes, and better label matching.

---

## 1. Ablation Matrix

### 1.1 Dimensions

| Dimension | Options | How to control |
|-----------|---------|----------------|
| **Sensor** | RealSense2 vs Lidar+Camera | Launch: `use_realsense_cam:=true` vs `false` (default). |
| **Pointmap / depth to SAM3D** | On vs Off | Worker: `--use-depth` (on) vs omit (off). Depth is always saved in the job and copied to output; SAM3D only receives it when `--use-depth` is set. |
| **Retrieval** | On vs Off | Launch: `sam3d_retrieval:=true` + `no_faiss_mode:=false` (on) vs `sam3d_retrieval:=false` + `no_faiss_mode:=true` (off, use candidate only). |
| **Registration** | On vs Off | Launch: `glb_registration_bridge:=true` (on) vs `false` (off). When off, use scene from initial poses only (e.g. `scene_sam3d_only.glb` / corresponding scene_graph). |

### 1.2 Running combinations

- Use a **fixed bag** (same RealSense or same lidar+camera bag) and **fixed queue base** with `use_run_subdir:=true` so each launch gets its own `run_YYYYMMDD_HHMMSS`.
- For each combination, launch once, play the bag, run the worker, then record:
  - `sam3d_queue_dir` (the run dir),
  - config tag, e.g. `sensor=realsense,pointmap=on,retrieval=on,registration=on`.
- **Outputs to evaluate**:
  - With registration on: `scene_graph.json` + `scene.glb` from the run’s output (or wherever simple_scene_buffer writes).
  - With registration off: `scene_sam3d_only`-style output (initial poses only).
- Convert pipeline output to a **common prediction format** (e.g. one USDA or one “evaluation JSON” per run) so the same evaluation script can consume it. We can stick to JSON per run in a common prediction format. In the scene_graph.json ie some common prediction format we want to include the label from yolo not just the job label.

### 1.3 Quantifying effectiveness (TP/FP/FN and detection metrics)

Object-level evaluation is treated as **detection**: each predicted instance and each GT instance is one “item.” There is no natural **true negative** at the instance level (see below); the main counts are **TP, FP, FN**, from which we get Precision, Recall, and F1.

**Definitions (instance-level, 3D boxes):**

- **Match criterion:** A predicted box and a GT box form a **match** iff:
  - **Label:** After label mapping, predicted label equals GT label (or is in the allowed set for that GT class).
  - **Geometry:** Overlap is above threshold, e.g. 3D IoU ≥ θ (e.g. 0.25 or 0.5) **or** centroid distance &lt; d (e.g. 0.5 m). Use one primary criterion (e.g. IoU) and optionally report the other.
- **Assignment:** Enforce **1-to-1** pairing so each GT is matched at most once and each prediction at most once. Options:
  - **Greedy:** Sort all candidate (pred, GT) pairs by overlap (e.g. IoU descending), then assign in order, skipping pred/GT already assigned.
  - **Hungarian:** Cost = −IoU (or + distance); minimize cost so that each GT and each pred is used at most once.
- **Counts:**
  - **True Positive (TP):** Number of GT instances that received a matching prediction (i.e. number of assigned pairs that satisfy the match criterion).
  - **False Negative (FN):** Number of GT instances with no matching prediction (unmatched GT).
  - **False Positive (FP):** Number of predicted instances that did not match any GT (unmatched predictions). Optionally split into: (a) wrong label but overlapping geometry, (b) correct label but insufficient overlap, (c) no overlapping GT at all.
- **True Negative (TN):** Not defined in the usual way for instance detection — we do not have a fixed set of “negative” object locations. Options if needed later:
  - **Omit TN** and report only Precision, Recall, F1 (standard in detection).
  - **Voxel / grid level:** Define a background class and count “correctly predicted background” as TN (requires voxelizing the scene and is more expensive).

**Derived metrics:**

- **Precision** = TP / (TP + FP)  (fraction of predictions that are correct).
- **Recall** = TP / (TP + FN)  (fraction of GT instances that were detected).
- **F1** = 2 · Precision · Recall / (Precision + Recall).

Report these **overall** (all classes) and **per class** (TP/FP/FN and P/R/F1 per canonical class). Optionally report **mean IoU over matched pairs** (and centroid/rotation error) as a quality-of-match metric in addition to counts.

**Practical choices to make:**

- **IoU threshold:** e.g. 0.25 (lenient) vs 0.5 (strict). Can report both or pick one as primary.
- **Label:** Require label match for a pair to count as TP; otherwise a prediction with perfect overlap but wrong label is FP (and the GT it overlaps is still FN unless another prediction matches it).
- **Multi-class:** Compute TP/FP/FN per class (after label mapping), then macro- or micro-average P/R/F1. Micro: pool all TP/FP/FN across classes and compute one P, R, F1. Macro: P/R/F1 per class, then average (good when class balance matters).

---

## 2. Evaluation Rewrite Plan

### 2.1 Goals

- Support **3D bounding boxes** (not only 2D XY).
- Support **oriented (rotated) boxes** (Supervisely already has `rotation` in cuboid_3d; USDA has orientation but current eval uses axis-aligned bbox_min/max).
- **Better label matching**: reduce or eliminate per-task manual mapping; support optional semantic/embedding-based matching.

### 2.2 Current state (brief)

- **GT**: Supervisely `*_voxel_pointcloud.pcd.json` — `objects` (classTitle, id) + `figures` with `geometryType: "cuboid_3d"`: `position`, `dimensions`, `rotation` (Euler x,y,z).
- **Predictions**: USDA (axis-aligned `boundingBoxMin/Max`, position, orientation) or Clio GraphML (position, dimensions, rotation matrix).
- **Current eval**: 2D XY axis-aligned IoU and centroid containment; per-task manual maps in `mappings.py` (e.g. `GROUND_TRUTH_LABEL_MAP_*`, `USDA_LABEL_MAP_*`, `CLIO_LABEL_MAP_*`).

### 2.3 Proposed evaluation pipeline

1. **Load GT**  
   - Reuse/refactor `parse_supervisely_bbox_json.parse_cuboid_data()` so that each box has: `position`, `dimensions`, `rotation` (Euler or quaternion), and a **canonical label** (after optional GT label map or as-is).

2. **Load predictions**  
   - One adapter per format: USDA (parse_usda), Clio (parse_clio), and optionally a single “evaluation JSON” format (list of boxes with position, dimensions, rotation, label).  
   - Normalize to a common internal representation: e.g. center, half-extents, rotation (quat or 3x3), label.

3. **Label matching**  
   - **Option A (keep current):** Per-task and per-source (USDA/Clio) manual mapping files (e.g. YAML or Python dict).  
   - **Option B (recommended):** **Automatic matching** with fallback to map:
     - **Embedding-based:** For each predicted label and each GT label (or canonical class set), compute a text embedding (e.g. sentence-transformers or CLIP text encoder); match predicted → GT by maximum cosine similarity above a threshold; optionally use Hungarian assignment per scene to avoid many-to-one.  
     - **String-based fallback:** Normalize strings (lowercase, replace separators), then match by substring or simple similarity (e.g. “chair” in “Seating/Newman” → chair), or Levenshtein.  
     - **Override:** Small config file (e.g. `label_map.yaml`) that forces pred_label → canonical_label; applied before or after automatic matching.

4. **Box representation and frame**  
   - Ensure GT and predictions are in the **same coordinate frame** (e.g. odom or world). Supervisely and USDA may differ; document and optionally add a static transform or offset in config.  
   - Internal representation: center `c`, half-extents `h`, rotation `R` (3x3). For axis-aligned, `R = I` and boxes are `[c - h, c + h]`.

5. **Metrics**  
   - **Detection counts (primary):** TP, FP, FN as in §1.3 (1-to-1 assignment, match = label + IoU/centroid threshold).  
   - **Detection rates:** Precision, Recall, F1 (overall and per class; optionally micro vs macro).  
   - **Axis-aligned 3D IoU:** For pairing and for quality of match; mean IoU over matched pairs.  
   - **Oriented 3D IoU** (optional): for pairing if desired; otherwise keep axis-aligned for speed.  
   - **Pose error:** For matched pairs: centroid distance (m), rotation error (rad or degrees).  
   - **Legacy:** Optional strict/relaxed accuracy (centroid containment) for backward comparison.  
   - **Aggregate:** Per-class and overall: TP/FP/FN, P/R/F1, mean IoU (and optionally pose errors).

6. **Matching predicted ↔ GT**  
   - Same as today: for each predicted box, find overlapping GT boxes (by 3D IoU or 3D centroid distance); optionally enforce 1-to-1 (e.g. greedy by IoU or Hungarian). Record pairs and unmatched GT.  
   - When comparing oriented boxes, use chosen metric (axis-aligned 3D IoU or oriented 3D IoU) for overlap and pairing.

### 2.4 Implementation layout (suggested)

- **`evaluations/`** (or a subfolder):
  - `eval_common.py`: Load GT (Supervisely), load predictions (USDA / Clio / eval JSON), normalize to common box format (center, half-extents, rotation), same frame.
  - `label_matching.py`: Embedding-based and string-based matching; optional YAML map override; API: `resolve_label(pred_label, gt_labels, method="embedding"|"map"|"string", map_file=None)`.
  - `metrics_3d.py`: Axis-aligned 3D IoU; optional oriented 3D IoU; centroid distance; rotation error; strict/relaxed containment; aggregation (per-class, overall).
  - `run_eval.py`: CLI that takes GT path, prediction path, prediction type (usda/clio/json), optional label map path and method; outputs metrics JSON and a short summary table.
- **Ablation runner** (optional): Script that, for a list of run dirs and config tags, (1) finds the right prediction file (USDA or exported scene_graph → evaluation JSON), (2) calls `run_eval.py`, (3) collects metrics into a single table (CSV + optional Markdown/LaTeX). See §3 for output formats.

### 2.5 Backward compatibility

- Keep existing `label_mapped_metrics` logic available behind a “legacy” or “2D” mode so current results can be reproduced (2D IoU, axis-aligned, manual maps only). New default can be 3D + optional oriented + optional auto label matching.

### 2.6 Diagnostics and guiding improvement

The evaluation scripts should not only report aggregate metrics but also **pinpoint where the pipeline fails** so you can decide how to improve **registration**, **retrieval**, and upstream (detection/segmentation). Below: what to record, how to interpret it, and concrete next steps.

**Principle:** Save enough per-slot and per-error detail so that (1) you can see whether registration/retrieval help or hurt, and (2) you can attribute failures to a stage (detection, retrieval, registration) and act on it.

---

#### A. Registration: is it helping?

**What to record (when registration is used):**

- For each **placed object** that gets a registered pose: you have an **initial pose** (from SAM3D + injector) and a **final pose** (after ICP). If that object is **matched to a GT box**, you can compute:
  - **Initial error:** centroid distance and rotation error from initial pose to GT.
  - **Final error:** centroid distance and rotation error from registered pose to GT.
  - **Registration delta:** change in pose (translation magnitude, rotation change) and whether error went down or up.

**Outputs the scripts can produce:**

- **Registration summary (per run):** For all matched slots that had registration:
  - Count / fraction where **final_error < initial_error** (registration helped).
  - Count / fraction where **final_error > initial_error** (registration hurt).
  - Mean improvement (e.g. centroid error reduction in m) when it helped; mean degradation when it hurt.
- **Per-slot fields** (in per-run JSON or a diagnostics table): `slot_id`, `initial_centroid_err_m`, `registered_centroid_err_m`, `improvement_m`, `registration_helped` (bool). Optionally: ICP fitness or iteration count if the pipeline exposes them.

**How this guides improvement:**

- If **registration often hurts:** ICP may be converging to wrong local minimum; try better initialization, segment PC instead of full scene, or tune ICP (distance threshold, max iterations, fitness threshold).
- If **registration rarely changes pose (tiny delta):** Registration may be redundant or initial pose already good; focus improvement elsewhere, or try a different target (e.g. segment-only).
- If **registration helps in some classes but not others:** Inspect failed cases (e.g. small or thin objects); consider class-specific thresholds or skipping registration when segment is too small.

---

#### B. Retrieval: is it choosing the right object?

**What to record:**

- For each slot the pipeline produces: **candidate** (SAM3D output) and **placed object** (candidate if retrieval off, or FAISS-retrieved if retrieval on). Optionally: whether retrieval was used and which object (path or id) was placed.
- When you **match a placed object to a GT** (by position/label), you can ask: did the **placed object's label** (or semantic) match the GT label? So you get:
  - **Correct object / correct class:** placed label matches GT (after mapping).
  - **Wrong class:** placed label ≠ GT (retrieval or candidate chose wrong type).

**Outputs the scripts can produce:**

- **Retrieval summary (per run):** When retrieval is on:
  - Fraction of slots where **retrieved ≠ candidate** (retrieval actually swapped the object).
  - Of those swapped: fraction where the placed object's **label matched GT** (retrieval helped) vs wrong label (retrieval hurt).
- **Comparison: retrieval on vs off:** Same scene, same GT; run with retrieval on and with retrieval off. Compare F1 (or per-class Recall/Precision). If F1 is **worse** with retrieval on, retrieval is likely picking wrong objects; if **better**, retrieval is helping.
- **Per-slot (optional):** `slot_id`, `candidate_label`, `placed_label`, `retrieval_used`, `gt_label` (if matched), `correct_class` (bool). Enables "when retrieval swaps, how often is it correct?"

**How this guides improvement:**

- If **retrieval off ≈ or better than retrieval on:** Retrieval may be harmful (wrong objects from FAISS). Next steps: improve FAISS index (more/better examples), tune CLIP embedding (e.g. crop vs full image), or use retrieval only when confidence is high; consider fallback to candidate when retrieval score is low.
- If **retrieval helps overall but wrong-class rate is high:** Focus on label/semantic consistency (better label map, or retrieval by semantics not just appearance).
- If **candidate is often wrong:** Then retrieval is critical; focus on making retrieval more reliable (index quality, scoring, or multi-view).

---

#### C. Detection and FP/FN: where do we miss or hallucinate?

**What to record (already in §1.3, add structure for "why"):**

- **Per unmatched GT (FN):** `gt_id`, `gt_label`, `gt_centroid` (or bbox). Tells you *which* objects we never predicted (e.g. "all doors are FN" → improve detection or label map for door).
- **Per unmatched prediction (FP):** `pred_id`, `pred_label`, and **reason** (or category):
  - **No overlapping GT:** prediction is in empty space (hallucination or wrong location).
  - **Overlapping GT but wrong label:** we have a GT at a similar location but predicted a different class (label/retrieval issue).
  - **Overlapping GT but low IoU:** same class (or nearby) but bad pose/size (registration or geometry issue).

**Outputs the scripts can produce:**

- **FN summary:** Count per class (e.g. "door: 5 FN, chair: 2 FN"). List or export FN instances for inspection.
- **FP breakdown:** Count by reason (no_overlap / wrong_label / low_iou) and per class. Optionally: centroid or bbox so you can visualize "where are FPs?"
- **Confusion-style view:** Rows = GT class, columns = predicted class (for matched + unmatched); shows swaps (e.g. table predicted as chair).

**How this guides improvement:**

- **High FN for one class:** Improve detection or segmentation for that class; or fix label mapping so that class is not collapsed into another.
- **FP mostly "no overlapping GT":** Too many hallucinated or misplaced objects; check segmentation (over-segmentation), or retrieval placing wrong instances.
- **FP mostly "wrong label":** Improve retrieval semantics or label consistency; or add a post-hoc classifier.
- **FP mostly "low IoU":** Pose or scale is off; focus on registration or SAM3D/scale estimation.

---

#### D. Where to store diagnostics and how to use them

- **Per-run JSON:** Include a `diagnostics` block: `registration_summary`, `retrieval_summary`, `fn_per_class`, `fp_breakdown` (and optionally per-slot lists). Same file as §3.2 `by_run/<scene>_<run_id>_metrics.json`.
- **Optional: diagnostics table:** e.g. `tables/diagnostics_<date>.csv` with one row per run and columns for registration_helped_pct, retrieval_swapped_pct, retrieval_correct_when_swapped_pct, fn_by_class (as columns or JSON), fp_no_overlap / fp_wrong_label / fp_low_iou counts. Lets you compare "which config gives better registration impact?" across ablations.
- **Next-steps checklist:** When implementing the scripts, add a small "interpretation" helper or doc that maps: "if registration_helped_pct < 50% → …", "if retrieval_off F1 > retrieval_on F1 → …", "if fp_no_overlap is high → …", so that after each run you can quickly decide what to try next.

---

## 3. Saving data for papers (tables and comparison)

Make it easy to extract results for writing: one place for all metrics, standard table layouts, and optional LaTeX/Markdown so you can paste directly into a paper or appendix.

### 3.1 What to save at pipeline run time

When you run the pipeline for an ablation, **write a small config manifest** next to the run so evaluation and tables can identify the setup without guessing:

- **Where:** e.g. `<run_dir>/eval_manifest.json` or `<run_dir>/config_tag.txt`, written by the launch script or by a one-line wrapper that records the flags used.
- **Contents (minimal):**
  - `scene` or `bag_id`: which scene/bag (e.g. `hallway-1`, `lounge-0`).
  - `sensor`: `"realsense"` | `"lidar_cam"`.
  - `pointmap`: `true` | `false`.
  - `retrieval`: `true` | `false`.
  - `registration`: `true` | `false`.
  - `run_id` or `timestamp`: e.g. `run_20250218_143022` or ISO timestamp.
- **Optional:** `gt_path` (path or name of the Supervisely GT JSON used for this scene) so the evaluator knows which GT to use.

If the launch does not write this automatically, keep a **run log** (CSV or spreadsheet) with columns: `run_id`, `run_dir`, `scene`, `sensor`, `pointmap`, `retrieval`, `registration`, `notes`.

### 3.2 Standard output layout for evaluation

Use a single **results root** (e.g. `evaluations/results/`) with a clear layout:

```
evaluations/results/
├── by_run/                          # one JSON per (run, GT) evaluation
│   ├── hallway-1_run_20250218_143022_metrics.json
│   ├── hallway-1_run_20250218_150011_metrics.json
│   └── ...
├── tables/                          # consolidated tables for papers
│   ├── ablation_main_20250218.csv   # one row per run/config, columns = metrics
│   ├── ablation_main_20250218.md    # same, markdown (for quick view / GitHub)
│   ├── ablation_per_class_20250218.csv  # optional: per-class P/R/F1 per run
│   └── ablation_main_20250218.tex   # optional: LaTeX tabular snippet
└── run_registry.csv                 # optional: run_id, run_dir, config columns, gt_path
```

- **Per-run JSON** (`by_run/<scene>_<run_id>_metrics.json`): Full output of the evaluator for that run and GT: TP/FP/FN, P/R/F1 (overall + per class), mean IoU, pose errors, and the eval config (IoU threshold, label method). Enables later re-aggregation or different thresholds without re-running inference.
- **Consolidated table** (`tables/ablation_main_*.csv`): One row per run (or per unique config if you aggregate over runs). Columns = config identifiers + key metrics. This is the main artifact for “performance comparison” in the paper.
- **Profiler timing (per run dir):** With the pipeline profiler enabled (default), each run directory contains `timing_events.csv` and `timing_summary.json` (see RUN_PIPELINE.md). Use `timing_summary.json` for inference and other step latencies (e.g. `sam3d_worker` / `inference` → mean_ms, std_ms) so you can add timing columns to ablation tables and report inference times across variations.

### 3.3 Table format (paper-ready)

**Main ablation table (one row per run or per config):**

| Column type   | Column names (example) |
|---------------|------------------------|
| **Identity**  | `scene`, `run_id` (or `config_id`) |
| **Config**    | `sensor`, `pointmap`, `retrieval`, `registration` |
| **Counts**    | `TP`, `FP`, `FN` |
| **Rates**     | `Precision`, `Recall`, `F1` |
| **Quality**   | `mean_IoU`, `mean_centroid_err_m`, `mean_rotation_err_deg` (optional) |
| **Timing**    | `inference_mean_ms`, `inference_std_ms` (from run dir `timing_summary.json`; see RUN_PIPELINE.md) — report inference times per pipeline variation |
| **Eval setup**| `iou_threshold`, `label_method` (optional, for reproducibility) |

- **CSV:** Same columns; open in Excel/Sheets/pandas; easy to filter (e.g. by scene, or registration on vs off) and to convert to LaTeX.
- **Markdown:** Same table for README or supplementary; render in GitHub or in docs.
- **LaTeX:** Script can write a `\begin{tabular}{...}...\end{tabular}` block (or `booktabs` style) so you paste into the paper; columns can be rounded (e.g. 3 decimal places) and aligned.

**Per-class table (optional):**

- Rows = same runs/configs. Columns = for each class (e.g. chair, table, door): `TP_<class>`, `FP_<class>`, `FN_<class>`, `P_<class>`, `R_<class>`, `F1_<class>`. Or a separate small table per class. Use for “comparison by object type” in the paper.

### 3.4 Comparison views for the paper

- **Main comparison:** Use `ablation_main_*.csv` (or `.tex`). Sort by F1 or Recall; group by scene, then by config. Typical figures: (1) Table: config × metrics. (2) Bar chart: F1 (or P/R) per config, optionally per scene.
- **Per-class comparison:** Use `ablation_per_class_*.csv` to show which object types improve most with pointmap, retrieval, or registration.
- **Side-by-side (two configs):** A small script or notebook that takes two `run_id`s (or two rows from the main table), loads the two per-run JSONs, and prints a short diff or side-by-side (e.g. TP/FP/FN and P/R/F1 for each, plus difference). Useful for “with vs without registration” in the text.

### 3.5 Suggested scripts (to implement)

1. **`run_eval.py`** (already in §2.4): Writes per-run metrics JSON; optionally appends one row to a running `tables/ablation_main_*.csv` or creates it.
2. **`collect_ablation_table.py`**: Input: list of run dirs (or `run_registry.csv`) + GT path per scene. For each run: read manifest, run evaluator if needed (or read existing `by_run/*.json`), then write/update `tables/ablation_main_<date>.csv` and `.md` (and optionally `.tex`). Single command to refresh the paper table after new runs.
3. **`export_latex_table.py`** (or a flag in `collect_ablation_table.py`): Read the main CSV, format as LaTeX (e.g. `booktabs`), round numbers, output to `tables/ablation_main_<date>.tex`.

With this, “saving data for the paper” = (1) run pipeline with manifest, (2) run evaluator per run, (3) run `collect_ablation_table.py` → open `tables/ablation_main_*.csv` or paste `*.tex` into the paper.

---

## 4. Summary

- **Ablations:** Vary sensor (RealSense vs lidar_cam), pointmap to SAM3D, retrieval, and registration via existing launch flags; run same bag and same eval per combination; record config in a manifest per run (§3.1).
- **Evaluation rewrite:** Single pipeline: load GT (Supervisely 3D cuboids), load predictions (USDA/Clio/eval JSON), normalize boxes and frame; match labels (embedding or string with optional map); compute TP/FP/FN, P/R/F1, mean IoU, pose errors; write per-run metrics JSON and consolidated tables (§2, §3).
- **Label matching:** Prefer embedding-based (or string) matching with an optional small override map so new tasks don’t require maintaining large manual dictionaries.
- **Paper-ready output:** One results layout (§3.2), main ablation table in CSV/Markdown/LaTeX (§3.3), optional per-class table and side-by-side comparison (§3.4); scripts to collect and export tables (§3.5) so you can extract results for tables and performance comparison without manual copy-paste.
- **Guiding improvement:** Evaluation scripts should output diagnostics (§2.6) so you can see how well registration and retrieval work: registration summary (helped vs hurt), retrieval summary (swapped and correct when swapped), FP breakdown (no_overlap / wrong_label / low_iou), FN per class. Use these to decide next steps (tune ICP, improve FAISS/index, fix detection for a class, etc.) and optionally a small “interpretation” checklist.
