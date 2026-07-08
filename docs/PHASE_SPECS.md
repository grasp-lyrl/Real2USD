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
