# Phase implementation specs (companion to REWORK_PLAN.md)

REWORK_PLAN.md says *what and why*; this file pins the *how* for decisions that are
load-bearing across phases. Anything not specified here is the executor's call — prefer
the simplest thing that satisfies the phase's definition of done. When a default below
proves wrong in practice, change it and note it in the run record; don't treat this file
as immutable.

## Global working agreements

- **Package layout:** `r2s3d_core/` (uv project, Python 3.10) with
  `src/r2s3d_core/{data,frames,tracks,registration,refine,recon,eval}/` and `tests/`.
  Location: `humble_ws/src_Real2USD/r2s3d_core/`. ROS wrappers import it; it never
  imports ROS.
- **Experiment records:** every eval run writes
  `results/<phase>_<name>/run.json` = `{git_sha, config (full), dataset, scenes,
  metrics, wall_time, created_at}`. Plots/tables are always regenerated from run.json
  files, never hand-edited. `results/` is gitignored except `*.json`.
- **Flags, not forks:** v1 behaviors stay callable behind flags (e.g.
  `registration.mode = v1_yaw_sweep | teaser | teaser+refine`) — the ablation table
  falls out of config sweeps.
- **Tests:** `frames` and `eval.metrics` get unit tests (golden values, round-trips).
  Nothing else requires tests unless it bites twice.
- **Provenance:** every output object records which code path produced it (see JSON
  schema below). No silent fallbacks anywhere.
- **Commits:** one per milestone/definition-of-done; never commit datasets or results
  binaries.

## Core interfaces (Phase 0, everything depends on these)

```python
# r2s3d_core/data/base.py
class Frame(NamedTuple):
    rgb: np.ndarray        # (H,W,3) uint8
    depth: np.ndarray      # (H,W) float32, meters, 0/NaN = invalid
    K: np.ndarray          # (3,3) intrinsics for THIS resolution
    T_world_cam: np.ndarray  # (4,4); camera frame = OpenCV optical
                             # (x right, y down, z forward); T_a_b maps
                             # column-vector points in frame b into frame a
    stamp: float           # seconds
    frame_id: int

class SequenceSource(Protocol):        # backends: replica, scannet, rosbag
    def __iter__(self) -> Iterator[Frame]: ...
    def __len__(self) -> int: ...
    def gt(self) -> list[GTObject] | None: ...

class GTObject(NamedTuple):
    instance_id: int
    label: str
    T_world_obj: np.ndarray   # (4,4)
    extents: np.ndarray       # (3,) full OBB dims, meters
    mesh: trimesh.Trimesh | None   # Replica: extracted per-instance from semantic ply
```

- Replica backend reads the NICE-SLAM render layout (`results/frame%06d.jpg/.png`,
  `traj.txt` 4x4-per-line world_cam poses, intrinsics from `replica_intrinsics.yaml`).
  GT from Replica's `*_semantic.ply` + `info_semantic.json` (per-instance vertex
  extraction; cache per scene as npz/trimesh).
- **One frame convention everywhere.** All world-frame quantities are gravity-aligned
  Z-up. Convert at the loader boundary, never downstream.

## Phase 0 — dataset + harness + naive baseline

**Resolved design decision:** the SAM3D-layout baseline uses **GT instance masks** to
isolate *layout error* from detection error — the motivating experiment ("even with
perfect masks, SAM3D placement is off by X"). A detector-driven variant is Phase 2+.

*Masks are always rendered, not provided.* The NICE-SLAM Replica release ships no
instance/semantic images, so `render_instance_mask` rasterizes each GT object's mesh
into a frame (project faces with z>0, `cv2.fillPoly`). Caveat: this is the object's full
silhouette — occlusion by *other* objects is not modeled (occlusion-aware masking is a
possible refinement; it did not bite in Phase 0).

*One view per object, on purpose.* An object is visible in ~100 sampled frames;
`select_best_view` picks the **single best** (largest visible mask area, excluding views
whose mask touches the image border; falls back to largest-area if all touch it) and
SAM3D runs **once** on that crop. Phase 0 does **not** fuse multiple views — single-best-
view is exactly the v1 weakness that **Phase 2 (ObjectTrack multi-view fusion)** exists to
beat. So the Phase 0 number is the "best single view + SAM3D layout" baseline.

Baseline procedure per scene: sample every 20th frame; select best view per GT instance
(above); run SAM3D via the disk-queue worker on that crop+mask (+depth pointmap); place
the mesh via SAM3D's predicted layout composed with that frame's `T_world_cam` (variant A);
variant B additionally refines with the v1-style ICP against masked depth. SAM3D outputs
are cached by **input hash** under `$SAM3D_QUEUE` — reruns are free. The eval writes all
pending jobs in one pass and reports partial results with a loud pending message; run the
worker (`conda run -n sam3d-objects ...`) then re-run to collect.

Job/worker contract (learned the hard way): the worker builds the depth pointmap from the
crop using the **full-image K + `crop_bbox`** (not a crop-adjusted K) and **requires** a
`meta.odometry` field (identity for datasets with no robot). The worker reads
`CONDA_PREFIX`, so launch it via `conda run`/activation, not the env's python directly.

**Metric definitions (r2s3d_core/eval/metrics.py):**
- Matching: Hungarian on 3D oriented-box IoU, threshold 0.25 (report 0.5 as secondary).
  Label-agnostic matching by default; label correctness reported separately (this
  decouples geometry from semantics — v1 conflated them).
- Per matched pair: centroid L2 (m); rotation error (deg) and per-axis scale ratio error,
  both **axis-labeling-invariant** (`geo.box_pose_error`); Chamfer-L1 and F-score@5cm (and
  @2cm) on 10k surface-sampled points in world frame.
  - **Why labeling-invariant (load-bearing, cost us a scare):** predicted orientation
    comes from the min-volume OBB of the posed mesh, whose principal-axis *order and sign
    are arbitrary*. Naively comparing that rotation matrix to the canonically-labeled GT
    axes inflated rotation error massively (~110° where the object was actually ~28° off)
    and scale likewise. `box_pose_error` resolves the correspondence over the 24 cube
    symmetries (composed with the class's yaw-symmetry group), then reports scale error
    under that *same* alignment — so a genuinely 90°-flipped elongated box reads as low
    rotation + high scale (and still fails IoU/Scan2CAD, correctly). IoU/F1 were always
    labeling-invariant. General lesson: rotations on boxes are only defined up to the box's
    symmetry — never compare raw OBB rotation matrices.
- Scan2CAD alignment accuracy: fraction of GT with a prediction within 20cm ∧ 20° ∧
  20% scale.
- Scene-level: precision/recall/F1 at IoU 0.25; **duplicate rate** = (matched
  predictions beyond the first per GT object) / #GT; count ratio #pred/#GT.
- Golden tests: hand-constructed box pairs with known IoU/rotation/scale answers.

**Done when:** `python -m r2s3d_core.eval.run --source replica --scene room0 --method
sam3d_layout` produces run.json with all metrics on ≥1 scene, plus variant B; a table
script renders both rows.

**Verified (room0, 43 objects, perfect GT masks + depth) — the motivating result:**
F1@.25 0.58, R@.5 0.05, **Scan2CAD acc 0.00**, centroid 10 cm, rotation ~28°, scale ~54%.
SAM3D is a strong shape prior but a weak metric localizer: depth anchors *position* well,
but rotation and (especially) scale are far past the 20°/20% gates, so 0/43 objects meet
the Scan2CAD criterion. Placement composition validated independently (posed SAM3D mesh
sits **3.8 cm median** from its own masked-depth cloud → the frame math is correct; the
error is SAM3D's, not ours). Every later phase must beat this.

**Inspection exports:** `r2s3d_core/recon/scene_glb.py` writes viewable pred/gt/compare
GLBs. SAM3D meshes are ~650k faces each (a full room ≈ 400 MB), so exports default to a
**lite** version — decimated to ~6k faces/object (`fast-simplification`, `uv sync --extra
viz`) → ~5–8 MB, pose/shape preserved. Rule: **debug views = lite by default, full-
fidelity opt-in** for figures/deliverables. Exported `.glb` stay gitignored.

## Phase 1 — frames + validation

- `r2s3d_core/frames.py`: named transforms with docstrings
  (`T_ODOM_CAM_GO2`, `R_CAM_PT3D`, `R_ZUP_YUP`, ...), composition helpers, and a
  `Convention` note at module top. Kill the 5 constants in `ply_frame_utils.py` and the
  3 duplicated extrinsics; Go2 extrinsics move to `config/go2_calibration.yaml`.
- Round-trip tests at 1e-9; a fixture test that reproduces one known-good v1 pose.json
  placement end-to-end through the new module (regression).
- Input validation at boundaries: depth NaN/hole fraction, SAM3D scale ∈ [0.05, 20],
  finite translations; violations raise or mark provenance, never pass silently.

**Done when:** Phase 0 numbers reproduce exactly through refactored code.

## Phase 2 — ObjectTrack

**Resolved decisions (2026-07-08, in practice):**
- **Detector-in-sim, not GT masks.** Phase 0 fed SAM3D perfect GT masks (1 call/GT
  instance → zero duplicates by construction). Phase 2 drives the tracker with a **real
  detector (YOLOE) run over the Replica RGB frames** so we can (a) measure placement
  degradation vs the GT-mask ceiling, (b) study detector prompting, and (c) show
  multi-view association + late-merge collapsing detector fragmentation. Realistic
  corruptions (`detect/corrupt.py`) are an optional stress-amplifier on top.
- **Detector = YOLOE** (`ultralytics`, ungated, auto-downloads weights). SAM 3 ([AI-6])
  is the gated upgrade path, not a blocker. Runs as a standalone caching step
  (`detect/run.py`, `detector` uv extra: torch cu128 for the 5090) → per-scene
  `DetectionSet` on disk; tracker/eval/tests import no torch (mirrors the SAM3D queue).
- **Prompt modes** (the prompting study): `gt` (scene GT label vocabulary — oracle-vocab
  upper bound), `generic` (fixed broad indoor noun-phrases), `pf` (prompt-free native
  vocab, `yoloe-11l-seg-pf.pt`). Diagnose reports GT detection-recall + mask IoU per mode.
- **Appearance/re-ID = masked-crop HSV color histogram** (`tracks/appearance.py`), behind
  an `Appearance` interface; replaces PHASE_SPECS's `clip_cos`. Swap CLIP/SigLIP in later
  without touching the associator.
- **Late-merge deviation:** the spec's registered-*mesh*-IoU criterion needs Phase-3
  registration; Phase 2 substitutes fused-cloud geometry (AABB-IoU > 0.3 ∨ voxel-overlap
  > 0.5 ∧ appearance_cos > 0.85). Recorded in run provenance; mesh-IoU merge lands with
  Phase 3.
- **Placement reuses Phase 0.** Mature tracks feed their best view to
  `sam3d_layout.{run_sam3d, place_from_sam3d, refine_icp, _fit_scale_to_extent}`; registration
  runs against the track's **fused multi-view cloud** (the natural upgrade of Phase-0
  `--icp-accumulate`). Methods: `object_track` (re-ID + late-merge on),
  `object_track_naive` (both off; `--v1-dedup` adds v1's 0.5 m same-label suppression), and the
  registration variants `object_track_{icp,scale,scale_icp}` (or `--registration <mode>`).
  Scene stats (`sam3d_invocations`, `n_mature`, `tracks_per_gt`, `n_merged`) land in
  run.json — these are the fragmentation-cleanup headline and are computable from the
  detector cache alone (no SAM3D worker needed).
- **Scene graph (parity with sam3d_layout).** object_track emits per-scene
  `<run>/<scene>/scene_graph.json` (via `config['_placements']`, written by `eval.run`
  unless `--no-placements`): per object `label` + metric `center`/`extents`/`T_world_obj`
  + `T_world_mesh` (raw `object.glb` verts → world; a visualizer rebuilds the posed mesh
  with no re-run — verified exact to ~1e-14 m) + best-view camera pose + detector lineage
  (`n_obs`, `det_track_ids`, `merged_from`). `label_source=detector`. This is the
  object-centric map deliverable and is emitted independent of GLB export.

- Lifecycle: TENTATIVE (created on unmatched detection) → ACTIVE (≥3 associated
  observations) → mature (≥6 kept views or sequence end) → RECONSTRUCTED → REGISTERED;
  MERGED/REJECTED are terminal. Only ACTIVE+ tracks accumulate clouds; only mature
  tracks trigger SAM3D.
- Association cascade (first match wins): (1) detector track id; (2) Hungarian over
  cost = 1 - [0.5·IoU(projected mask, track's reprojected cloud hull) +
  0.3·clip_cos + 0.2·centroid_gate], gate: centroid distance <
  max(0.5 m, 0.02 m/s · seconds_since_last_seen) (drift-aware), clip_cos > 0.75 hard
  floor.
- View score = mask_area_norm · (1 - edge_contact) · sharpness_norm ·
  (1 + 0.5·angle_novelty); keep top-6.
- Fused cloud: 1 cm voxel hash, per-voxel max 1 point, mask-filtered depth only.
- Late merge (reconciliation): candidate pairs by centroid < 1 m; merge if
  registered-mesh IoU > 0.3 ∨ (cloud overlap > 50% ∧ clip_cos > 0.85).

**Done when:** duplicate rate on ≥2 Replica scenes drops vs the Phase 0 detector-free
count baseline and vs v1 behavior replayed (if cheap); SAM3D invocations/scene logged
and reduced; association decisions inspectable in a per-scene debug HTML/rerun-io dump.

## Phase 3 — localization stack

- Mesh sampling: 20k surface points. FPFH at 2 cm voxel. TEASER++
  `estimate_scaling=True`, noise bound 5 cm. ICP: point-to-plane, 3 stages
  (5 cm → 2 cm → 1 cm correspondence), scale frozen after TEASER.
- Scale init sanity: reject registration if TEASER scale outside [0.3, 3.0]× the
  extent-ratio init; fall back to extent-ratio scale + rotation-only search, mark
  provenance.
- Gravity soft prior: after ICP, if roll+pitch < 10° snap to upright; if > 10°, keep
  but flag (`tilted=true`) — replaces v1's hard yaw-only projection.
- Confidence ∈ [0,1] = logistic over (inlier_frac, inlier_rmse, mean multi-view
  silhouette IoU); calibrate the three weights on Replica GT once, freeze.
- Multi-view refinement (`refine/`): nvdiffrast; optimize (quaternion, t, log-scale)
  with Adam lr 1e-2, ≤200 iters, loss = Huber depth residual (masked) +
  (1 - silhouette IoU) + 0.1·upright regularizer, summed over the track's kept views;
  early-stop on plateau. Runs offline after registration; flag-gated.
- **Design for the joint upgrade (REWORK_PLAN 2.7):** structure the refiner as a scene
  optimizer from the start — it takes a list of objects; per-object mode is N=1 with
  cross-object terms disabled. `refine.joint = true` adds pairwise SDF non-penetration +
  ground-contact terms over all objects' (R,t,s) simultaneously. Same code path, one
  flag; the per-object-vs-joint ablation falls out.

**Done when:** on Replica, v2 (teaser+refine) beats both sam3d_layout and v1_yaw_sweep
on Scan2CAD-accuracy and centroid error, with per-stage timing in run.json.
**Go/no-go:** if it doesn't clearly beat sam3d_layout here, stop and debug before any
benchmark scale-out — nothing downstream can compensate.

## Phase 4 — reconciliation + export

- Isaac settle: headless `python.sh` script; import scene USD, enable physics on
  objects (convex decomposition colliders), settle 2 s sim-time, record per-object
  displacement; objects moving > 25 cm get flagged (bad placement or floater).
- Scene graph JSON schema v2 (canonical output; USD generated from it):
  ```json
  {"schema": 2, "objects": [{
      "id": "track_000123", "label": "chair",
      "label_votes": {"chair": 5, "stool": 1},
      "mesh": "assets/track_000123.glb",
      "T_world_obj": [[...4x4...]], "scale": [sx, sy, sz],
      "confidence": 0.87, "tilted": false,
      "provenance": {"source": "sam3d|retrieval", "registration":
        "teaser+icp+refine", "fallbacks": [], "views_used": 6,
        "settle_displacement_m": 0.02}}]}
  ```

**Done when:** post-settle inter-mesh penetration < 1 cm everywhere; JSON validates
against the schema; USD round-trips into Isaac.

## Phase 5 — benchmark campaign

Each external baseline runs from its own repo/env at a pinned commit recorded in
run.json; reuse *published* numbers where the protocol matches exactly (say so in the
table caption); otherwise rerun. Order: Replica tables first (no gate), then
Clio-protocol, then ScanNet/Scan2CAD/MetaScenes as access arrives. Real-robot bags rerun
last with the frozen config — no per-scene tuning after benchmark numbers are locked.

### Phase 5 side-thread — ProcTHOR / MolmoSpaces scene-graph comparison

A coworker maintains a holistic hierarchical scene-graph benchmark (Hydra-lineage:
Objects / Rooms / Places / Building / Mesh / Trajectory / Grounding) comparing
Hydra, DAAAM, ConceptGraphs, Clio, Khronos, HOV-SG, OpenGraph, ConceptFusion,
Kimera-Semantics, Memory-Over-Maps. It runs on a fixed slice of **ProcTHOR-10k** houses
sourced through Ai2 **MolmoSpaces** (ids `137, 200, 428, 534, 569, 573, 683, 771, 912`
+ a 10th TBD). MolmoSpaces itself ships only *manipulation/navigation* eval — the
scene-graph metrics are the coworker's own harness (get it; see AI-7).

Real2USD is object-centric, so it honestly answers **only two row-groups**: **Objects**
(P/R/F1, counts, class-free geo recall) and **Mesh** (Chamfer / footprint IoU). Rooms /
Places / Building / Trajectory / Grounding are out of scope until those layers exist
(future work — the plan is to keep climbing the hierarchy).

**`ProcThorSource`** (`r2s3d_core/data/procthor.py`, `procthor`/`molmospaces` source key,
`procthor` uv extra) implements the standard `SequenceSource` contract via AI2-THOR
native rendering: loads a house by id (`prior.load_dataset("procthor-10k")[split][id]`),
drives a deterministic reachable-position × yaw × horizon trajectory, yields posed RGB-D
`Frame`s and OBB `GTObject`s in our Z-up OpenCV-optical world. **Transform is
round-trip-validated** (`tests/test_procthor.py::test_backprojection_inside_gt_obb`:
masked depth back-projected into GT OBBs, median containment 0.86, vertical FOV confirmed
over horizontal). Unity(LH,Y-up)→world(RH,Z-up) via the `(x,z,y)` permutation `_M_WU`.

**Render cache (default-on, load-bearing for detector-driven runs).** AI2-THOR RGB is
non-deterministic (shading/exposure re-randomized; |Δrgb|≤~205/255 across renders) while
depth/pose/seg are stable. That RGB jitter reshuffles appearance-based re-ID → different
tracker `track_id`s between the SAM3D queue and collect passes → cached meshes bind to the
wrong objects (measured: bogus F1 0.10 vs 0.54). `ProcThorSource` therefore renders each
scene **once** to disk and replays it deterministically (RGB Δ=0, stable ids, no controller
on replay, ~1.2s). Auto path `data_root()/procthor_cache/<scene>_<params-hash>`, shared
transparently by `detect.run` and `eval.run` (key = render params; stride/max_frames excluded,
applied at replay; registration mode excluded so all variants reuse one mesh set). ~200MB/scene.
Disable with `cache=False` or `R2S3D_PROCTHOR_NOCACHE=1`; bump `_CACHE_VERSION` on layout
changes. Tests: `tests/test_procthor_cache.py` (synthetic cache, no ai2thor). See
[[procthor-render-cache]].

Metric parity — **reconciled to the coworker's answered defs (2026-07-14, AI-7).** `evaluate()`
reports **two matching protocols side by side**:
- **Coworker-comparable (`cd_*`, THE comparison set):** Hungarian on **centroid distance ≤ τ**
  (default 1 m; swept [0.25,0.5,0.75,1.0,1.5] in `centroid_{f1,recall}_by_tau`). `cd_f1`/`cd_
  precision`/`cd_recall` label-agnostic; `cd_micro_f1`/`cd_macro_f1`/`cd_per_class` label-aware
  (their Object Micro/Macro F1). `class_free_recall_1m` = their Class-Free Geo Recall (GT found
  if ANY pred centroid ≤ 1 m, label-ignored, not one-to-one). `scene_chamfer_mean_m` = their
  scene-level pooled symmetric Chamfer (**convention confirmed ✓**).
- **Ours (stricter, non-coworker):** headline `f1`/`precision`/`recall` + `micro_f1`/`macro_f1`
  on **3D OBB IoU ≥ 0.25** — kept as a harder diagnostic, NOT compared to their table.
- `surf_{recall,precision,fscore}@{0.05,0.02}`: surface-recon point coverage (renamed from
  the mislabeled `geo_*`; a DIFFERENT metric from `class_free_recall_1m`). NaN until GT meshes.
- `chamfer_symmetric_mean_m` (per-pair ½·chamfer_l1), named per-scene counts.

**Still open (AI-7):** (a) GT must be **filtered to the shared ProcTHOR/DAAAM vocab** like
predictions (≈74.7 obj/scene) — needs the coworker's lexicon; (b) verify macro averaging set +
exact "many-to-one F1" vs `ec2-ma` source; (c) **Option B** association family (many-to-one,
fragmentation, merge, pairwise) via `scripts/scene_graph_metrics.py:compute_track_metrics` +
per-detection→track provenance; (d) Footprint (2D top-down) IoU. **Split = `val`** (not train);
canonical 10 ids include **434**; trajectory parity (their `poses.csv`) deferred (Option 3 —
render `val` ourselves for now, flag "our trajectory").

### Perception robustness — detector-driven ProcTHOR (NEXT, the crux)

Strategy/why: `REWORK_PLAN.md §2.10`. All current ProcTHOR results use GT (native THOR) masks
= perfect-perception ceiling; the benchmark methods use their own perception, so a fair column
needs **detector-driven** numbers. Diagnosis: recall-dominated (YOLOE 35–70% on Replica).

Steps (prioritized; reuse Phase-2 `detect/` + `tracks/` on the new adapter):

1. **Measure the gap. ✅ scene 200 done (2026-07-14); extend to 137/428.** Generate YOLOE
   detections on ProcTHOR RGB, run the detector-driven method, compare to the native-GT-mask
   `sam3d_layout_scale_icp` ceiling. **Use `object_track_icp`, NOT `scale_icp`** (see step 4 —
   the scale-fit hurts on detector masks). Commands:
   ```
   uv sync --extra procthor --extra dev --extra detector --extra registration --extra viz
   uv run python -m r2s3d_core.detect.run --source procthor --scene 200 --prompt gt \
       --out results/detections/procthor --diagnose   # detections dir resolves <out>/gt/<scene>
   uv run python -m r2s3d_core.eval.run --source procthor --scene 200 \
       --method object_track_icp --detections results/detections/procthor/gt --no-geometry \
       --phase procthor --name procthor_object_track_icp_s200 \
       --sam3d-queue results/procthor_procthor_object_track_icp_s200/sam3d_queue
   # then run the SAM3D worker over that queue (conda sam3d-objects), then re-run eval to collect.
   # SINGLE worker only — concurrent workers OOM the 5090 (silent per-job failures → input_failed/).
   ```
   The collect re-run emits, by default (no `--no-glb`/`--no-placements`), the lite viz GLBs
   (`<scene>/scene_{pred,gt,compare}_lite.glb`) and `<scene>/scene_graph.json` — both are
   on-disk-only (gitignored, regenerable), keep them for the leading system.
   (The track path needs the SAM3D worker for its new meshes — detector masks ≠ native masks →
   new job keys, so the `sam3d_layout_scale_icp` queue won't hit. Registration mode is NOT in the
   job key, so `object_track`/`_icp`/`_scale`/`_scale_icp` share one mesh cache — collect once,
   ablate modes freely with no re-queue.) **Scene-200 result:** ceiling F1 0.638 (recall 0.577);
   detector `object_track_icp` F1 0.538 (recall 0.481, TP 30→25); `scale_icp` F1 0.237. Gap with
   icp is ~0.10 F1, recall-dominated. **Accept:** ceiling-vs-detector table, degradation
   attributed to recall vs mask-IoU vs placement.
2. **Multi-view recall recovery (headline).** Report **per-frame recall vs per-track recall**
   (union of detections over the trajectory via ObjectTrack association). Expect per-track ≫
   per-frame — the asset-centric/tracking payoff. **Accept:** the two recall curves + the count
   of objects recovered only by multi-view. (Scene 200: per-track 0.48 ≈ per-frame 0.50 — few
   objects, weak showcase; 137/428 have more objects and should separate the two.)
3. **Segment-everything → track → label.** Class-agnostic SAM2/SAM3 masks (high object recall)
   → track → open-vocab (CLIP) label per track; decouples recall from a fixed vocabulary.
4. **Robustify the scale-fit to noisy masks — HIGH priority (2026-07-14 finding).** The
   depth-extent scale-fit is the GT-mask *win* but a *liability* on detector masks: scene-200
   detector F1 layout 0.452 → +scale 0.151 → +scale_icp 0.237, vs +icp 0.538. `_observed_obb_extent`
   trusts the mask, so noisy detector masks contaminate the fused cloud → inflated OBB extent →
   wrong scale → boxes miss the IoU gate. Add percentile/outlier-robust extent (e.g. drop the
   top/bottom k% per axis, or MCD/convex-hull-trim) and measure scale-err vs mask-IoU. Until
   fixed, detector-driven uses `object_track_icp`. See [[scale-fit-hurts-on-detector-masks]].
5. **Detector upgrades:** SAM 3 (AI-6, gated), Grounding-DINO+SAM2, YOLOE prompt-mode study
   (generic / prompt-free vs gt-vocab — still open from Phase 2).
6. **Association hardening (from SuperMap, RSS'26 — see `RELATED_WORK.md`).** Attacks the
   fragmentation half of the recall problem: noisy detector masks → wrong `centroid_world` →
   the isotropic 0.5 m world gate in `associate._pair_benefit` rejects a true match → a
   duplicate fragment track spawns (splits per-track recall). Two flagged, ablatable changes:
   - **6a. Reprojection-space, pose-aware gate (`assoc_reproj`, implement first).** Mask-median
     depth error is dominated by the **along-ray** component, so an isotropic world-space ball is
     the wrong gate shape. Project the track's fused-cloud centroid into the current frame (reuse
     `fusion.project_cloud_mask`'s intrinsics/front math on a single point) and gate
     **anisotropically**: pixel distance `< REPROJ_PIX_GATE` (tight ⊥-to-ray) **and** depth ratio
     `obs_z/pred_z ∈ [1/(1+τ), 1+τ]` (loose along-ray), plus a `W_REPROJ` benefit term. This is
     SuperMap's 3D-to-2D idea adapted — we already associate in the world frame (so ego-motion is
     handled by good poses), so the win is noise-shaped gating, NOT ego-motion. Default **off**
     until validated. **Accept:** on scene 200 detector-driven, fewer spawned/merged fragments
     and per-track recall moves toward per-frame recall; F1 narrows vs the 0.638 ceiling; no
     regression on GT-mask runs. Ablation: add `--assoc-reproj` to the step-1 `eval.run`
     command (registration modes share the mesh cache, so no SAM3D re-queue) →
     `object_track_icp` ± `assoc_reproj` as two rows.
     **Gate params are first-guess and do NOT need tuning yet.** `REPROJ_PIX_GATE=60px` and
     `DEPTH_RATIO_TOL=0.35` (60 reverse-engineered from a synthetic test) trade fragmentation
     (too tight) vs ID-swaps/false-merges (too loose). But the flag is off by default and the
     scene-200 finding (per-track ≈ per-frame recall) says association isn't the bottleneck
     there — so run the on/off ablation with the guessed values FIRST. Only sweep the knobs
     IF the flag is shown to help on a busier scene (137/428). They are overridable without a
     code edit via `--reproj-pix-gate` / `--reproj-depth-ratio-tol` (config keys
     `reproj_pix_gate` / `reproj_depth_ratio_tol`; knob-override unit-tested), so a sweep is a
     shell loop when/if it's warranted — and association-quality metrics (fragment count,
     per-track recall) come from tracks vs GT, no SAM3D worker needed for that part.
   - **6b. Bayesian confidence-weighted label fusion (`label_bayes`).** Replace the plain
     `label_votes` Counter (`+1` per detection, `most_common(1)`) with summed **confidence-weighted
     log-evidence** per class (logit(det_score) as a diagonal-dominant confusion stand-in; no true
     confusion matrix for open-vocab detectors), posterior = softmax. Correct merge = log-add (not
     count `update()`). Exposes `label_confidence()` → stamp into `scene_graph.json` node
     provenance; a confidence floor `REJECT`s low-confidence tracks **loudly** (matches the
     make-fallbacks-loud directive). **Accept:** label precision on detector-driven runs, effect of
     the confidence-reject on FP count. Second, after 6a.

Note: `object_track` passes a stable `job_key` (`{source}_{scene}_t{track_id}_{framing}`) and
honours `full_frame`. The depth-extent scale-fit is **done (2026-07-14)**: the track path reuses
`_fit_scale_to_extent` against the track's fused multi-view cloud. Registration modes: `none` |
`icp` (rigid pose) | `scale` (scale-fit only) | `scale_icp` (scale-fit + rigid pose), exposed
as named methods `object_track_{icp,scale,scale_icp}` (mirroring `sam3d_layout_*`) or via
`object_track --registration <mode>` (`--icp` is the legacy alias for `icp`).
