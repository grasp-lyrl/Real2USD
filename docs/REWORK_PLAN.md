# Real2SAM3D Rework Plan (IROS 2026 rejection → resubmission)

**Paper:** Asset-Centric Metric-Semantic Maps of Indoor Environments (arXiv 2510.10778v2)
**Status:** Rejected IROS 2026. Target: ICRA 2027 / RA-L (or IROS 2027).
**Core thesis of the rework:** SAM 3D is a strong *shape* prior but a weak *metric localizer*.
The new system treats SAM 3D as a canonical-shape generator only, and owns localization
itself via multi-view evidence accumulation + certifiable Sim(3) registration + multi-view
refinement + scene-level reconciliation.

This document is written to be executed by a coding agent (Opus) with minimal additional
context. File references are to the current code so the executor can find everything.

---

## 1. Assessment of the current system

### 1.1 What is genuinely good (keep these)

1. **The asset-centric USD thesis.** Objects as retrievable/generatable assets with poses,
   in a USD scene graph that an LLM can parse, is still a differentiated position. The
   2025–26 field (SceneComplete, SimRecon, MetaScenes, MessyKitchens, EmbodiedGen) has
   moved *toward* this "simulation-ready compositional reconstruction" framing — the paper
   was early, not wrong. Reposition into that conversation instead of the "open-vocab
   mapping" conversation (Clio/ConceptGraphs), where the metrics don't favor us.
2. **Retrieval-or-generate hybrid.** CLIP/FAISS retrieval of known assets with SAM 3D
   generation as the open-set fallback (`sam3d_retrieval_node.py`) is a good architecture
   and nobody else does both at room scale. SceneComplete generates only; ACDC retrieves
   only.
3. **Registration against real depth instead of trusting generative layout.** The instinct
   in `sam3d_glb_registration_bridge_node.py` + `registration_node.py` (mask-filtered depth
   as ICP target, SAM3D pose as init) is exactly right — it's the *execution* that is weak
   (see 1.3).
4. **Physics settling as reconciliation.** Settling objects in Isaac Sim for physical
   plausibility is a real contribution; MessyKitchens (2026) is now publishing
   non-penetration as a headline feature. Quantify it (penetration depth, settle
   displacement) instead of using it qualitatively.
5. **Modular ROS2 + disk-queue worker split.** The job-dir handoff
   (`sam3d_job_writer_node.py` → `scripts_sam3d_worker/run_sam3d_worker.py` in a separate
   conda env) is the correct way to isolate the heavy model, enables offline replay, and
   gives per-slot debug artifacts (pose.json, per-slot GLBs). Keep the pattern.
6. **The timing instrumentation** (`pipeline_profiler_node.py`, `/pipeline/timings`) —
   reviewers want decomposed runtime; the plumbing already exists.
7. **The evaluation scaffolding** (`humble_ws/evaluations/`): greedy 3D-IoU matching,
   open-set P/R in Clio's style, FP breakdown, per-class CSVs. It's a real harness; it
   needs better metrics and public benchmarks, not a rewrite.

### 1.2 Does the logic make sense?

**The pipeline-level logic is sound**: detect → (retrieve|generate) canonical mesh →
register to observed depth → reconcile scene. That is the same skeleton as SceneComplete
(RA-L 2025) and Scan2CAD-lineage work, and it is the right skeleton.

**Two design decisions do NOT hold up:**

1. **Trusting SAM 3D's layout for anything metric.** SAM 3D's translation/scale are in
   normalized camera coordinates from a flow-matching layout head. Its own repo documents
   3× scale errors *with* calibrated depth pointmap conditioning
   (facebookresearch/sam-3d-objects issues #56, #57); ~10–20% scale error is the good
   case. The current code applies SAM3D scale blindly (`ply_frame_utils.py:100`), uses
   SAM3D translation as ICP init, and **falls back to that same init when ICP fitness is
   low** (`registration_node.py:474-488, 681-689`) — so exactly the objects where
   registration fails inherit the least trustworthy pose, silently, into the scene.
2. **Per-detection processing with no object persistence.** "Multi-view" today is only:
   (a) YOLOE track ids, used solely as a 60 s dedup timer
   (`sam3d_job_writer_node.py:92-127`), and (b) an accumulated global lidar cloud. There
   is no per-object evidence accumulation, no best-view selection (the *first* crop that
   passes the filter is the one SAM3D sees, even if it's the worst view), no fusion of
   per-view pose estimates, no merging when a track breaks and the same chair gets a new
   track id (→ duplicate objects in the scene). The paper's own failure cases (table seen
   only as legs from the Go2's viewpoint, chairs intersecting tables) are direct
   consequences.

### 1.3 Concrete weaknesses (from the code deep-read)

Placement/registration:
- ICP init from 4 hardcoded yaw hypotheses + DBSCAN(eps=0.2, min_samples=20)
  (`registration_node.py:606-626`); FGR at a 5 mm correspondence threshold on
  voxel-downsampled clouds; point-to-point (not point-to-plane) objective; fitness (an
  overlap ratio) used as the accept metric though it doesn't measure pose accuracy.
- `yaw_only_registration:=true` hard-projects to Z-up (`registration_node.py:507-514`) —
  correct prior, wrong mechanism: if the SAM3D mesh comes out tilted, yaw-only *preserves*
  the tilt.
- No Sim(3): scale is whatever SAM3D said; ICP then fights the scale error with
  translation. Complete-mesh-vs-partial-scan registration is a known ICP failure mode.
- Registration target is either the global accumulated cloud cropped by radius or a single
  frame's masked depth — never a fused, per-object, multi-view cloud.

Robustness/engineering:
- Camera extrinsics hardcoded as translation-only `[0.285, 0, 0.01]` with identity
  rotation in **three** places (`lidar_cam_node.py:126`, `realsense_cam_node.py:45`,
  `sam3d_glb_registration_bridge_node.py:37`) plus demo_go2 frame constants baked into
  `ply_frame_utils.py:64-71`. No TF2, no calibration file, not portable.
- Silent degradation everywhere: FAISS load failure silently disables retrieval
  (`sam3d_retrieval_node.py:75-102`); malformed pose.json silently falls back to
  clustering; low-fitness registrations enter the scene unmarked; SAM3D scale is never
  range-checked; depth is never validated for NaNs/holes.
- The frame chain (raw → PyTorch3D → camera → odom, row-vector convention, 5 hardcoded
  rotation constants) is correct-by-testing, not correct-by-construction. Any change
  breaks it invisibly.

Evaluation (why reviewers likely scored it down):
- Custom scenes only, manually annotated; no public benchmark → no comparability.
- No rotation error, no scale error, no Chamfer/F-score — for a paper whose claim is
  *precise object placement*, pose accuracy is never actually measured.
- Mixed headline numbers: hallway chairs mIoU 0.187 beats Clio (0.119) and SAM3D (0.112),
  but relaxed centroid accuracy 0.394 *loses* to SAM3D (0.727). 24/33 objects found.
  1.6 s/object vs Clio 0.025 s/object.
- SAM3D as a baseline reads as a strawman (single-image model evaluated at scene level);
  ConceptGraphs/HOV-SG absent; no per-module ablations; single runs, no variance.
- Naming hazard: 2026 reviewers read "SAM3D" as Meta's model — if any ambiguity with
  Yang et al. 2023 SegmentAnything3D exists in the draft, kill it.

---

## 2. Target system design ("R2S3D v2")

### 2.1 Design principle

> **SAM 3D provides shape. The robot provides metric truth.**
> Every object's pose, scale, and existence in the scene must be justified by fused
> multi-view sensor evidence, with an explicit confidence, or the object is flagged.

### 2.2 New core abstraction: the ObjectTrack

Replace the per-detection flow with a persistent per-object entity (this is the single
biggest architectural change; everything else hangs off it):

```
ObjectTrack {
  track_id            # stable id, survives detector track breaks
  label_votes         # multi-view label histogram (YOLOE labels across views)
  clip_embedding      # running mean of view embeddings (for re-ID + retrieval)
  observations[]      # per-view: {stamp, T_odom_cam, mask, crop, K, view_score}
  fused_cloud         # voxel-hashed, odom-frame, mask-filtered depth points ∪ over views
  canonical_mesh      # from retrieval OR SAM3D (generated once, from best view(s))
  pose  T_odom_obj    # Sim(3): R, t, per-axis scale s
  pose_confidence     # registration inliers/RMSE + multi-view reprojection IoU
  state               # TENTATIVE → ACTIVE → RECONSTRUCTED → REGISTERED → (MERGED|REJECTED)
}
```

**Association (the multi-view fix #1, and the duplicate-instance fix):** a new
`object_track_node` associates incoming detections to tracks by: detector track id when
available → else Hungarian matching on (3D centroid gate from mask-median depth,
projected-mask IoU, CLIP cosine). CLIP re-ID is what heals track breaks (occlusion,
exit/re-entry, detector flicker — the failure modes a live ROS2 stream has and curated
dataset frames don't). Duplicates die in three layers: (1) association at ingest;
(2) generation-once-per-mature-track, so a missed association costs a redundant
lightweight track, not a redundant mesh; (3) **late merge at reconciliation** — before
export, merge tracks whose fused clouds/meshes overlap (mesh-IoU > 0.3, compatible label
family), offline, with all evidence in the common odom frame. The hard streaming case is
revisits under odometry drift: use a drift-aware association gate (radius grows with
time-since-last-observation) and rely on CLIP re-ID + layer (3) as the backstop. Track
duplicate rate (predicted/GT count) as a headline metric — it was v1's main visible
failure.

**Detector choice:** primary = **Meta SAM 3** (concept-promptable open-vocab detection +
segmentation + *video tracking*, released with SAM 3D and natively paired with it) with
a broad noun-phrase prompt list — better track persistence on a jittery robot stream
directly reduces duplicates at the source, and mask conventions match SAM 3D. YOLOE
remains as the lightweight fallback and an ablation row ("detector choice"). If SAM 3 is
too heavy for the robot-side node, run it in the offline/near-online tier — the
disk-queue architecture is indifferent.

**Label strategy: decouple labeling from detection; never gate generation on a label**
(v1's silent label-substring drops were its worst semantic failure). The detector
proposes object regions (class-agnostic or broad-prompt); the *track* accumulates label
evidence — CLIP/SigLIP or small-VLM voting across its K best views (`label_votes`) —
which beats any single-frame detector label and feeds retrieval, open-set metrics, and
the LLM scene graph. Generate-everything (all masks → SAM3D) is ruled out: 10–20 s/object
and it reconstructs walls/floor; keep an objectness/size/background filter instead.

**View scoring & best-view selection (multi-view fix #2):** score every observation:
mask area, edge-contact ratio (reuse `tracking_pre_sam3d_filter.json` logic), blur
(variance of Laplacian), mask-depth coverage, and *novelty* of viewing angle vs already
kept views. Keep top-K (K≈4–6) diverse views per track. SAM3D runs **once per track** on
the best view (optionally with its pointmap), not on the first crop that shows up. This
alone should measurably improve mesh quality on the "table = legs only" failure case,
because a later, better view wins.

### 2.3 Localization stack (the precision fix)

Registration target is now the track's **fused multi-view cloud** (already
gravity-aligned in odom), not a single frame. Pipeline per track, replacing
`registration_node.py`'s internals (node interface can stay):

1. **Scale init — never from SAM3D.** s₀ = per-axis extent ratio between fused-cloud OBB
   and canonical-mesh OBB, with the vertical axis anchored by ground-plane contact when
   the object class is floor-standing. (ACDC/Any6D-style; robust to partial views by
   using visible-extent matching along observed axes only.)
2. **Coarse alignment — certifiable, correspondence-based.** FPFH correspondences →
   **TEASER++ with `estimate_scaling=true`** (Sim(3), robust to the ~90%+ outliers that
   partial scans of a complete mesh produce). Fallback ordering: TEASER++ → FGR → current
   yaw-sweep (keep as last resort + ablation baseline).
3. **Polish — point-to-plane ICP** at fixed scale, coarse-to-fine voxel schedule.
   Gravity as a *soft* prior: penalize roll/pitch deviation instead of hard yaw-only
   projection (fixes the tilted-mesh-stays-tilted bug while keeping the Z-up benefit).
4. **Multi-view render-and-compare refinement (multi-view fix #3, the paper's technical
   novelty).** Differentiable rendering (nvdiffrast, Diff-DOPE-style) of the posed
   canonical mesh into the K kept views jointly; minimize masked-depth L1 + silhouette
   IoU over (R, t, s) summed across views. Single-view silhouette alignment is
   scale-ambiguous; K posed views + depth pins all 9 DoF. ~1–3 s/object on GPU, offline.
   (Alternative if this stalls: per-view FoundationPose/Any6D + Barfoot–Furgale
   covariance-weighted SE(3) fusion with quaternion eigen-mean; keep as ablation arm.)
5. **Confidence + gating.** pose_confidence = f(inlier RMSE, inlier fraction, multi-view
   reprojection mask IoU). Below threshold → object stays in the scene graph flagged
   `low_confidence`, is excluded from metrics'
   "confident" tier, and is *never* silently placed at the SAM3D-predicted pose. This
   removes the current worst behavior.
6. **Scene-level reconciliation** (upgrade of `simple_scene_buffer_node.py`): duplicate
   merge (2.2), ground-plane snap for floor-standing classes, pairwise de-penetration,
   then Isaac Sim physics settle. Log per-object settle displacement — it becomes an
   evaluation metric (2.6 below) instead of an invisible fixup.

### 2.4 What SAM 3D is used for, explicitly

- Canonical textured mesh (its F1@0.01 = 0.234 vs ≤0.163 for all prior generators — it IS
  the strongest shape prior available; say so in the paper).
- Nothing else. Rotation/translation/scale outputs are used only as *one candidate init*
  for step 2, never as a fallback result. Paper narrative: "we measure SAM 3D's layout
  error at scene scale (Table X), showing xx cm / xx° / xx% scale error, motivating
  robot-side localization" — i.e., convert the weakness into the paper's motivating
  experiment. This experiment is cheap: run current `scene_sam3d_only` output through the
  new metrics.

### 2.5 Engineering fixes bundled with the rework

- One `frames.py` module: all transforms via named 4×4s with docstrings + round-trip unit
  tests; extrinsics from a calibration YAML / TF2, deleting the three hardcoded copies
  and the demo_go2 constants scatter.
- Validate SAM3D outputs (scale ∈ [0.05, 20], finite translation, watertight-ish mesh)
  and depth (NaN/hole fraction) at ingest; reject → log → count.
- Every fallback logs loudly and stamps the output record with which path produced it
  (needed for the ablation tables anyway).
- Atomic writes for scene.glb/scene_graph.json (temp + rename).

### 2.6 Evaluation & benchmarks (the resubmission-critical part)

Four tables, all on public data, plus the existing custom scenes as the "real robot"
table:

1. **Placement accuracy (headline, differentiating):** ScanNet + Scan2CAD (or MetaScenes'
   706 ScanNet scenes with sim-ready GT assets). Report the standard Scan2CAD criterion —
   **translation ≤20 cm ∧ rotation ≤20° ∧ scale ≤20%** — plus median translation /
   rotation / scale errors. No open-vocab mapping competitor reports this; it directly
   evidences the "precise placement" claim. Add per-object Chamfer + F-score@5cm/1cm
   against GT meshes.
2. **Clio-protocol parity:** Clio's public Office/Apartment/Cubicle/Building sequences,
   Clio's own osR/osP strict/relaxed + F1 + counts + s/frame. Makes the Clio comparison
   unimpeachable and reuses `evaluations/` mostly as-is.
3. **Open-vocab mapping context:** Replica (ConceptGraphs' 8 scenes), their metrics, with
   ConceptGraphs + HOV-SG numbers (published; rerun if feasible).
4. **Sim-usability (replaces qualitative Isaac demos):** watertight %, spawn penetration
   depth, physics-settle displacement, and one quantified downstream task (LLM/Nav2
   semantic-nav success, ≥3 seeds, mean±std) — ACDC/RialTo precedent.

**Baselines:** Clio, ConceptGraphs, (HOV-SG), Meta SAM 3D per-frame + `make_scene()`
(the *strong* version of the generative baseline), ACDC-style retrieval+bbox-fit, and
our own v1 pipeline. **Ablations (one row each):** view selection on/off; TEASER++ vs
FGR vs yaw-sweep; Sim(3) vs fixed scale; multi-view refinement on/off vs per-view-fusion
arm; retrieval vs generation vs hybrid; physics settle on/off; pose-noise sensitivity
(inject SE(3) perturbations on Replica GT poses — extends the existing
`evaluations/pose_sensitivity_eval.py` — to preempt the "you assume localization"
question; all target datasets ship posed RGB-D, so SequenceSource backends are thin
loaders: iMAP/NICE-SLAM Replica renders via ConceptGraphs' script, ScanNet .sens +
BundleFusion poses, Clio RealSense rosbags). **Metric additions to
`evaluations/`:** rotation geodesic error, per-axis scale error, Chamfer/F-score,
Hungarian (not greedy) matching, multi-run variance.

**Must-cite additions:** SAM 3D (2511.16624) with explicit disambiguation from
SegmentAnything3D (2306.03908) and SAM 3D Body; MV-SAM3D (2603.11633 — closest work,
differentiate: robot odometry + lidar vs offline pointmaps, room scale, USD output);
RecGen (2604.27106); SceneComplete; ACDC (2410.07408); MetaScenes (2505.02388); SimRecon
(2603.02133); MessyKitchens (2603.16868); URDFormer; DRAWER; FOUND-IT (2605.25371);
Scan2CAD lineage; TEASER++; FoundationPose/Any6D; mesh-quality-vs-pose study 2408.08234.

### 2.7 Novelty strategy (the rejection said "stitched-together modules" — it was fair)

Novelty must be constructed, not hoped for. Three compounding moves:

1. **One MAP objective as the paper's spine.** Joint estimation of a set of Sim(3)-posed
   assets from a posed RGB-D stream: per-object shape (retrieved|generated) + pose/scale;
   measurement terms = multi-view depth residuals + silhouette consistency; scene priors
   = gravity/upright, ground contact, pairwise non-penetration. Present pipeline stages
   as inference steps of this model (tracking = data association, registration = init,
   refinement = optimization, reconciliation = priors/projection). Confidence gating and
   merge criteria are terms of the objective, not heuristics bolted on.
2. **Algorithmic centerpiece: scene-level JOINT refinement.** Merge Phase 3's per-object
   multi-view refinement and Phase 4's de-penetration into one differentiable
   optimization over ALL objects simultaneously: per-object multi-view depth+silhouette
   residuals + cross-object SDF non-penetration + ground contact, optimizing every
   (R,t,s) together. Differentiation: Diff-DOPE is single-object/single-view; MV-SAM3D
   is offline collision-aware fusion from curated pointmaps; physics settling elsewhere
   (incl. our v1) is a non-differentiable afterthought. Ablation per-object vs joint
   directly measures it (targets the chairs-intersecting-tables failure). Isaac settle
   remains as final validation/projection, and as the ablation arm.
3. **Measurement contribution, claimed explicitly:** quantified generative-layout-error
   study at scene scale (Phase 0 GT-mask experiment), first Scan2CAD-grade placement
   evaluation for robot mapping, first asset-placement numbers on Replica.

Contribution list becomes: (i) formulation, (ii) joint refinement algorithm + ablations,
(iii) layout-error study + benchmark protocol, (iv) the system as vehicle. Phase 3/4
executors: build per-object refinement first (it's the joint optimizer with N=1 and no
cross terms), then add cross-object terms — same code path, flag-gated
(`refine.joint = true`).

### 2.8 Paper repositioning

- **Venue conversation:** "simulation-ready compositional scene reconstruction from a
  robot," not "open-vocab mapping." Contributions: (1) multi-view evidence-accumulating
  object tracks that turn a single-image generative model into a scene-scale mapper;
  (2) Sim(3) certifiable registration + multi-view differentiable refinement that fixes
  generative layout error (quantified); (3) physics-reconciled USD scenes evaluated with
  Scan2CAD-grade placement metrics + downstream robot task.
- Keep the honest speed framing (offline/near-online asset mapping; Clio-class systems
  are real-time but produce boxes, not sim-ready assets).
- **Representation positioning (defuses the "why USD over JSON?" complaint):** the
  contribution is the asset-centric scene graph (assets + Sim(3) poses + labels +
  confidence). JSON is its canonical serialization and the LLM-context format
  (token-lean); USD is the *simulation export* — Isaac-native loading for the physics
  settle + sim-usability eval, materials/physics schemas, instancing of repeated assets.
  Never argue USD-as-LLM-representation (v1's weakest claim; the README itself concedes
  buffer.json works as context). Consider renaming the system after the method, not the
  file format. Note: no prior work evaluates asset placement on Replica (asset lineage is
  ScanNet-anchored: Scan2CAD/ROCA/DiffCAD, MetaScenes, LiteReality arXiv 2507.02861 —
  add LiteReality to must-cites); reporting on Replica is a first, and ScanNet/Scan2CAD
  remains the differentiating table.
- **MetaScenes positioning (closest-looking work; get this paragraph right):**
  MetaScenes/Scan2Sim (CVPR 2025) shares our *output* (asset-replaced sim-ready scenes)
  but not our *problem*: it curates from complete ScanNet scans with human annotators
  ranking assets, and Scan2Sim is trained on those annotations. We are the autonomous,
  online, robot-embodied version — partial egocentric views, no human in the loop. Use
  MetaScenes three ways: (1) cite as problem validation; (2) its human-verified
  asset+pose selections as GT for the placement table (grades asset choice + pose +
  plausibility jointly); (3) Scan2Sim auto-retrieval as a baseline (it needs a complete
  scan; we don't). Anticipate and preempt the "MetaScenes but automated" read by leading
  with autonomy-from-partial-observation as the claim.

---

## 3. Execution plan for Opus (phased, with acceptance criteria)

**Phase 0 — Dataset + harness + naive baseline (1 day).** The evaluation substrate is
public data from day one; do NOT baseline against v1's outputs on the custom bags — that
inherits the rejected evaluation. Steps:
(a) **Data:** download Replica via ConceptGraphs' scripts (instant, posed RGB-D + GT
instance meshes/boxes). Same day, submit access requests for ScanNet + Scan2CAD (and
MetaScenes) — human action, longest pole in the campaign.
(b) **`SequenceSource` adapter (structural, load-bearing):** one interface yielding
posed RGB-D frames + intrinsics, with backends for Replica, ScanNet (later), and ros2
bags. This decouples the pipeline from ROS/Go2 topics and lets every benchmark and the
robot data flow through the same code.
(c) **Harness:** extend `evaluations/` with rotation geodesic error, per-axis scale
error, Chamfer + F-score@5cm/1cm, and Hungarian matching, computed against Replica GT.
(d) **Naive baseline:** SAM3D per object using its *own predicted layout*, composed via
`make_scene()` (plus a variant refined by the current v1 ICP path) on 1–2 Replica
scenes. This row is both the paper's motivating experiment (SAM3D layout error at scene
scale) and the number every later phase must beat.
*Accept: one table with SAM3D-layout (+ICP variant) rows on ≥1 Replica scene, all new
metrics computed end-to-end through the SequenceSource adapter.*
(The `v1-iros2026` tag/worktree stays frozen; a "vs v1" ablation row is optional —
regenerate it late by pointing v1 at the same data only if the comparison earns its
table row. Never a blocker.)

**Phase 1 — frames.py + validation + loud fallbacks (½–1 day).** 2.5 above. Structural
decision: v2 core logic (tracks, frames, registration, fusion) lives in a plain Python
library importable without ROS (e.g., `r2s3d_core`); ROS2 nodes become thin wrappers.
This is what makes dataset runs, unit tests, and the benchmark campaign cheap. *Accept:
round-trip transform tests pass; Phase 0 baseline numbers reproduce exactly through the
refactored code (regression test).*

**Phase 2 — ObjectTrack node (1–2 days).** `object_track_node` between camera nodes and
job writer: association (id → centroid/IoU/CLIP), fused per-track cloud (voxel hash),
view scoring, top-K view buffer; job writer becomes "reconstruct best view of mature
track." *Accept: on a Replica sequence (via SequenceSource) and a replayed bag,
duplicate-object count drops vs the Phase 0 baseline; per-track fused clouds visualized;
SAM3D invocations per scene decrease.*

**Phase 3 — Localization stack (2–3 days).** Scale init + TEASER++(Sim(3)) + pt-to-plane
ICP with soft gravity prior, inside `registration_node` interface; confidence gating; keep
old path behind a flag for ablation. Then multi-view nvdiffrast refinement as a separate
offline node/script. *Accept: Scan2CAD-criterion accuracy on ≥1 ScanNet scene beats v1
path and beats raw SAM3D layout; per-stage timing logged.*

**Phase 4 — Reconciliation upgrade (1 day).** Merge/de-penetrate/settle with logged
displacement; USD/GLB export with confidence attributes. *Accept: zero inter-mesh
penetration > 1 cm after settle; settle displacement reported per object.*

**Phase 5 — Benchmark campaign (2–4 days, parallelizable).** ScanNet+Scan2CAD (or
MetaScenes), Clio datasets, Replica; baselines (Clio + ConceptGraphs reruns, SAM3D
`make_scene`); ablation grid; real-robot scenes rerun with v2. *Accept: all four tables
filled with mean±std.*

**Phase 6 — Paper rewrite** per 2.7 (novelty strategy) + 2.8 (repositioning).

**Environments:** primary dev env is a **uv project** for `r2s3d_core` (pin Python 3.10
to match ROS Humble so the same package installs editable inside the docker container).
Phases 0–3 run there entirely — no ROS. The SAM3D worker stays in its existing
`sam3d-objects` conda env, untouched (disk-queue handoff; on datasets the queue is just
directories). ROS humble docker only for thin node wrappers, bag replay, and the Clio
rerun (Phases 2 wrapper + 5). Isaac Sim uses its own `python.sh` (Phase 4).

**Dependency notes for the executor:** Meta SAM 3 checkpoint (HF-gated like
sam-3d-objects — request access early, human step), TEASER++ (pip `teaserpp-python` or build), Open3D
≥0.18 (point-to-plane, FPFH), nvdiffrast + PyTorch (in the uv env — keep the SAM3D conda
env frozen), scipy `Rotation` (quaternion mean), ScanNet/Scan2CAD data agreements needed
early (start downloads Phase 0), Clio datasets public, Replica via ConceptGraphs'
scripts.

**Risk fallbacks:** TEASER++ integration stalls → scaled-Umeyama on FPFH
correspondences inside RANSAC (still Sim(3), still better than v1). nvdiffrast refinement
stalls → per-view FoundationPose + SE(3) fusion arm becomes the primary. ScanNet access
delays → MetaScenes or Replica-only for the placement table; keep Scan2CAD criterion
regardless (computable on any GT-box dataset with orientation + extents).
