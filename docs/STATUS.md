# STATUS — where we are in the v2 rework

**This is the living progress tracker. It is the first doc to read to know the current
state, and the last doc to update at the end of every work session / milestone.** Keep
it honest: "done" means *verified* (tests pass / numbers produced), not "code written".

- Strategy & rationale: `REWORK_PLAN.md` · Interfaces & resolved decisions: `PHASE_SPECS.md`
- **Things only the human can do: `ACTION_ITEMS.md`** (Claude adds to it on every gated dependency)
- Datasets & access: `DATASETS.md` · In-family systems & ideas to steal: `RELATED_WORK.md`

_Last updated: 2026-07-15 (first REAL detector-driven val-s200 run: `object_track_icp` on
gt-prompt YOLOE detections + full SAM3D worker → iou_f1@.25 **0.545**, centroid **4.5cm**,
class-free recall 1.0, scan2cad 0.15, chamfer 0.10. Corrects the earlier "box inflation"
scare (that was a SAM3D-free cloud-OBB proxy artifact; real placement is fine). GT mesh
orientation fixed: placed by AI2-THOR true rotation, not PCA (`_R_unity_euler`+`place_canonical_by_linmap`,
cache v2). Prior: split footgun fixed (`--split` on rescore/detect.run), CLIP-nearest-in-set
label mapping (`eval/label_map.py`, `--label-map clip`). NEXT = other variants + generic/pf
detector prompts through the real pipeline; extend to 137/428)._

### Detector-driven ProcTHOR — val scene 200 (measured 2026-07-14, no SAM3D; recall/placement only)

All on **val** split, 93 GT objects, 392 frames (stride 1). `--diagnose` recall is label-
agnostic mask-IoU>0.25 vs native seg. Tracker = `object_track` (reid+late_merge), SceneObjects
from each mature track's fused-cloud OBB.

- **Detection recall (YOLOE prompt study):** best-view→multi-view — `gt` 0.355→**0.710**,
  `generic` 0.269→0.602, `pf` 0.430→**0.710**. Mask IoU ~0.96 in all modes (mask quality is
  NOT the problem; recall is). `pf` finds the most per-view but ties `gt` at multi-view with
  438 junk labels → prompt vocab buys **label correctness, not recall**.
- **Recall ceiling decomposes:** trajectory coverage **0.796** (19/93 never visible) ×
  detector-given-visible **0.892** (gt/pf), 0.757 (generic). Once views are fused the detector
  is barely the bottleneck — coverage is.
- **Real tracker (gt):** one-to-one cd_recall 0.516, class-free 0.957 @1m; centroid err **13cm**.
  τ-sweep shows the 0.96→0.52 gap is the **coarse-1m-tolerance/clutter artifact** (vanishes at
  τ=0.1), NOT over-segmentation (only 9% of GT fragmented at 25cm). **`gt` is not over-merged/
  over-segmented; late-merge tuning won't help it.** `pf` IS over-segmented (185 tracks, 31%
  fragmented @25cm) → late-merge is a `pf`-only fix, and `pf` isn't our config.
- **REAL SAM3D+ICP placement (val s200, 2026-07-15, 72/93 placed): iou_f1@.25 0.545, iou_recall
  0.484, cd_f1@1m 0.727, class-free 1.0, centroid 4.5cm, rot 3.6°, scan2cad 0.151, chamfer 0.10,
  micro-F1 (CLIP) 0.388.** This CORRECTS the earlier "box inflation is the #1 lever" claim: that
  0.11 iou_f1 was a **SAM3D-free cloud-OBB proxy artifact** (raw depth-cloud boxes ~3×/side too
  big). The real pipeline (SAM3D shape + depth-extent + ICP) already produces well-sized,
  well-localized boxes — iou_f1 is a healthy 0.545, centroid 4.5cm. Robust extent estimation is
  NOT the bottleneck; the SAM3D+scale-fit design handles it. (The SAM3D-free tracker numbers
  above remain valid — they use track centroids, which are legit; only the cloud-OBB *extent* was
  the proxy artifact.)
- **GT mesh orientation fixed (2026-07-15):** GT asset meshes were placed by PCA-axis matching
  (`fit_canonical_to_obb`), whose axis-sign ambiguity flipped asymmetric assets (upside-down
  chairs). Now placed by AI2-THOR's true `rotation` (`_R_unity_euler` → `_M_WU @ R`, det -1 LH→RH;
  `place_canonical_by_linmap`), validated by native-mask silhouette IoU (0.497→0.528 mean; kettle
  0.21→0.77) and structurally upright. Moves scene chamfer only (0.104→0.0998; OBB metrics
  axis-invariant, unchanged). GT cache bumped to v2 (stores rotation).
- **Metrics critique (data-backed):** their 1 m centroid tolerance manufactures a 0.44 recall
  gap that is pure tolerance and lets over-segmentation inflate recall → **report tight-τ
  (0.25m) recall + 3D-IoU + scale/Chamfer as our columns** (their suite is blind to placement/
  extent, our differentiator). Label F1 needs CLIP-nearest-in-set (AI-7) to be fair to open-vocab.

## ⚠ AI-7 ANSWERED (2026-07-14) — current ProcTHOR numbers are NOT yet comparable

The coworker gave exact defs (full list in `ACTION_ITEMS.md` AI-7). **Two of them invalidate
every ProcTHOR number produced so far:** (1) their split is **`val`**, ours defaulted to
**`train`** — *different houses* per id, so our s200/etc. results are on the wrong scenes;
(2) their matching is **Hungarian on centroid distance ≤ τ (default 1 m)**, NOT our OBB
IoU@0.25 — a different matched set. Also: object F1 is **label-aware**; **Class-Free Geo
Recall = 1 m object-centroid** (our `geo_recall@tau` is a *different* surface metric,
mislabeled); GT must be **filtered to the shared ProcTHOR/DAAAM vocab** (≈74.7 obj/scene);
**10th id = 434** (we were missing it). Chamfer convention already matches (✓).

**Confirmed vs source + implemented (2026-07-14):** matching = **greedy 2D top-down (X,Y)
centroid ≤ 1 m** (NOT Hungarian/3D/IoU) — `eval/metrics.py` `cd_*` now greedy-XY, plus
`cd_micro_f1_many_to_one` (their over-seg-tolerant any-overlap F1) and XY `class_free_recall_1m`.
**Q3: the published harness does NO GT filtering → the DAAAM-lexicon blocker is REMOVED; no
hard blocker remains for the Objects column.** Remaining: regenerate on **val** (+ id 434)
with greedy-XY, decide whether to drop our `_THOR_EXCLUDE_TYPES` (they count doors/windows),
and trajectory parity is deferred (render val ourselves). The code/pipeline is fine; the
**numbers** just need the val re-run.

## ▶ NEXT SESSION — perception robustness (detector-driven ProcTHOR)

**The crux (Chris, 2026-07-14):** every ProcTHOR result so far uses GT (native THOR) masks =
perfect-perception ceiling. The benchmark methods use their OWN perception, so our GT-mask
column is **not a fair entry** — a valid column needs **detector-driven** numbers. Diagnosis:
recall-dominated (Phase-2: YOLOE found 35–70% of GT). Contribution thesis: multi-view tracking
makes per-track recall ≫ per-frame recall (objects missed in one frame seen in another) — the
asset-centric payoff. Strategy: `REWORK_PLAN.md §2.10`; how/commands: `PHASE_SPECS.md
§Perception robustness`. **Do in order:** (1) ✅ gap measured on scene 200 (below); extend to
137/428; (2) per-frame vs per-track recall (headline); (3) segment-everything→track→label;
(4) **robustify scale-fit to noisy masks — now HIGH priority (see finding)**; (5) detector
upgrades; (6) association hardening from SuperMap (RSS'26, see `RELATED_WORK.md`). First:
`uv sync --extra procthor --extra dev --extra detector --extra registration
--extra viz`, then `detect.run --source procthor` + `object_track_icp` (NB: icp, not scale_icp).

**Landed 2026-07-14 (unit-tested, NOT yet validated on scene 200):** step 6a — anisotropic
reprojection association gate (`config['assoc_reproj']`, default OFF) in
`tracks/associate.py`. Replaces the isotropic 0.5 m world gate (which rejects true re-IDs
when mask-median depth drifts along-ray → spawns duplicate fragments) with a pixel-tight /
depth-ratio-loose gate. New unit test `test_reproj_gate_heals_alongray_depth_noise` (fragments
3→2 mature). **TODO to call it done:** ablate `object_track_icp ± assoc_reproj` on scene 200,
confirm fewer fragments + per-track recall ↑ + no GT-mask regression. CLI: `--assoc-reproj`
(on/off) + `--reproj-pix-gate` / `--reproj-depth-ratio-tol` (tuning). **Gate params
(60 px / 0.35) are first-guess but do NOT need tuning yet** — the flag is off by default, and
the scene-200 finding above (per-track ≈ per-frame recall) says association isn't the
bottleneck there, so 6a may buy nothing until a busier scene (137/428). Order: run the on/off
ablation with the guessed values first; only sweep the knobs (overridable, no code edit;
knob-override unit-tested) IF the flag is shown to help. 6b (Bayesian label fusion) specced,
not implemented.

**Gap measured — scene 200 (2026-07-14).** Detector-driven vs native-GT-mask ceiling, n_gt=52:

| method | F1 | recall | precision | Scan2CAD | cent |
|---|---|---|---|---|---|
| ceiling `sam3d_layout_scale_icp` (GT masks) | **0.638** | 0.577 | 0.714 | 0.308 | 0.027 m |
| detector `object_track_icp` (**best**) | **0.538** | 0.481 | 0.610 | 0.000 | 0.086 m |
| detector `object_track_scale_icp` | 0.237 | 0.212 | 0.268 | 0.038 | 0.152 m |
| detector `object_track` (layout only) | 0.452 | — | — | 0.000 | 0.093 m |

Per-frame detector recall (diagnose) = 0.50. **Two findings:** (a) with `icp`, the gap to the
ceiling is modest (~0.10 F1) and recall-dominated (TP 30→25); per-track≈per-frame recall so
association isn't losing objects — the loss is objects never detected (scene 200 is a weak
multi-view showcase, few objects; test 137/428). (b) **The depth-extent scale-fit — the GT-mask
win — is a LIABILITY on detector masks:** it craters detector F1 (layout 0.452 → scale 0.151),
because noisy masks contaminate the fused cloud → inflated OBB extent → wrong scale → boxes miss
the IoU gate. So for detector-driven runs use `object_track_icp`, and step 4 (robustify the
scale-fit) is now the priority. See [[scale-fit-hurts-on-detector-masks]].

**Infra fixes that made the number valid (2026-07-14):** (1) `_diagnose` recall bug — it
compared detector masks to mesh-silhouettes (offset from ProcTHOR RGB), reporting 0.019; now
uses native masks → 0.50. (2) **ProcTHOR render cache** — AI2-THOR RGB is non-deterministic
(|Δrgb|≤205), which reshuffled tracker `track_id`s between the SAM3D queue and collect passes so
meshes bound to the wrong objects (bogus F1 0.10, TP 4). `ProcThorSource` now caches renders to
disk (default-on; `R2S3D_PROCTHOR_NOCACHE=1` to disable), replaying deterministically (RGB Δ=0,
stable ids, no controller on replay). See [[procthor-render-cache]].

**Scale-fit port (2026-07-14):** the scale-fit runs in the `object_track` path against the
track's fused cloud. Methods `object_track_{icp,scale,scale_icp}` mirror `sam3d_layout_*`;
equivalently `object_track --registration <mode>`. SAM3D job cache keyed by
`..._t{track_id}_{framing}` (registration-independent), so registration variants share one mesh
set — ablating them is an eval re-run, **not** a SAM3D re-queue.

## Phase dashboard

| Phase | Title | State | Notes |
|------|-------|-------|-------|
| 0 | Dataset + harness + naive baseline | 🟢 **done** | harness verified on 8 Replica scenes; SAM3D worker validated; `sam3d_layout` row on room0. **Full-frame >> crop input (F1 .58→.77, Scan2CAD 0→.16) — now the default.** all-scene aggregate + full-frame ICP pending |
| 1 | frames.py + validation + loud fallbacks | 🟢 **done** | `frames/` matches v1 `ply_frame_utils` to 1e-9; Phase 0 numbers reproduce exactly; ROS-node dedup deferred to the Phase 2 wrapper |
| 2 | ObjectTrack node | 🟡 **in progress** | ROS-free `tracks/` + `detect/` (YOLOE) built; tracker tests pass; **detector-in-sim fragmentation cleanup verified room0+room1** (SAM3D calls ↓~4×, tracks/GT 1.3→0.33). SAM3D-mesh placement-degradation (vs GT-mask ceiling) collecting; ROS wrapper deferred |
| 3 | Localization stack (TEASER++ / ICP / refine) | 🟢 **scale fix found** | **`scale+ICP` is the win**: depth-extent scale-fit cuts scale err 0.31→0.14 (2.6×) and Scan2CAD 0.12→0.31. ICP fixes pose; TEASER-vs-depth shelved (shrink-to-fit). See below. |
| 4 | Reconciliation + export | ⬜ not started | needs Isaac Sim |
| 5 | Benchmark campaign | 🟡 side-thread started | **ProcTHOR/MolmoSpaces scene-graph comparison adapter built + validated** (see below). Main campaign dataset access is the long pole — [AI-2..5](ACTION_ITEMS.md) started early |
| 6 | Paper rewrite | ⬜ not started | |

Legend: ⬜ not started · 🟡 in progress / partially blocked · 🟢 done · 🔴 blocked

## Phase 5 side-thread — ProcTHOR / MolmoSpaces scene-graph comparison (2026-07-13)

**Goal:** appear as a column in a coworker's holistic hierarchical scene-graph benchmark
(Objects/Rooms/Places/Building/Mesh/Trajectory/Grounding; vs Hydra, DAAAM, ConceptGraphs,
Clio, Khronos, HOV-SG, …) evaluated on a fixed slice of **ProcTHOR-10k** houses sourced
via Ai2 **MolmoSpaces** (ids `137,200,428,534,569,573,683,771,912` + 10th TBD). Real2USD is
object-centric → we honestly fill **only Objects + Mesh** rows (deep-but-narrow; Mesh is our
differentiator). Rooms/Places/Building/Trajectory/Grounding await those layers (future).
Design/interfaces: `PHASE_SPECS.md` Phase-5 side-thread. How-to: `DATASETS.md §5`.

**Built + VERIFIED (adapter, not yet the method rows):**
- `r2s3d_core/data/procthor.py` `ProcThorSource` (source keys `procthor`/`molmospaces`,
  `procthor` uv extra = `ai2thor` 5.0 + `prior`). AI2-THOR native RGB-D + instance-seg +
  exact GT boxes; deterministic reachable-position × yaw × horizon trajectory. Registered in
  `data/registry.py`.
- **Rendering works** on this desktop against **X `:1`** (RTX 5090 / driver 580); Unity build
  auto-downloaded.
- **Transform round-trip validated** (the load-bearing bit): Unity(LH,Y-up)→world(RH,Z-up)
  `(x,z,y)` permutation; masked depth back-projected into GT OBBs → **median containment 0.86**,
  vertical FOV confirmed over horizontal. `tests/test_procthor.py` (4 pure units +
  1 gated render test), **all 53 core tests pass** (`uv run pytest -q`; the 2 backprojection
  round-trips are the render-gated integration tests).
- **End-to-end pipeline validated:** `eval.run --source procthor --scene 137 --method oracle`
  → 93 objects, **F1=1.000 / S2C=1.000 / cent 0 / rot 0**; `oracle_noisy` degrades correctly
  (F1 0.58). `__iter__` frames flow (depth valid 1.0, camera height z=1.58 m — Z-up correct).
  Runs: `results/procthor_procthor_oracle{,_noisy}_137/`.

**Per-scene `scene_graph.json` (`results/<run>/<scene>/scene_graph.json`):** an object-centric
scene graph for downstream inference (navigation etc.). Header: `scene, source, method,
label_source` (`gt` for GT-detector `sam3d_layout*`, `detector` for `object_track`), `frame`
(Z-up world, meters), `sam3d_queue`. Each object: `id, label, center, extents, T_world_obj,
T_world_mesh` (raw `object.glb` verts → world, incl scale-fit + ICP), `mesh` (relative path),
`job_id`, per-object `scale_fit`/`icp`, plus the **best-view camera that generated the mesh**
(`cam_position` + `cam_quat_xyzw` ROS/TF-ready, full `T_world_cam`, `camera_K`) — also a good
nav viewpoint for "go observe object X" (stand at `cam_position`, face per the quat, look-at
= `center`). Any visualizer/metric rebuilds the posed prediction —
`trimesh.load(<queue>/<mesh>)`, apply `T_world_mesh` — **without re-running placement**
(round-trip verified). Off with `--no-placements`. Reduces (not removes) the AI-8 Mesh-rows
re-run: geometry vs real GT meshes still needs a pass; reviz/inspection no longer does.
NOTE: for `sam3d_layout*`, `label` is the **ground-truth** THOR objectType (GT-detector
ceiling); real predicted labels come from the `object_track` (YOLOE) path.

**Runner improvements (standing requirements, reused across phases):** (a) the SAM3D
disk-queue is now **per-experiment** (`<out_dir>/sam3d_queue`, override `--sam3d-queue`) so
objects from different runs never mix; same-experiment reruns still hit the input-hash
cache. (b) `eval.run` **always exports a viewable GLB** (`<out_dir>/<scene>/scene_{pred,gt,
compare}_lite.glb` via `recon/scene_glb.export_pred_vs_gt`) unless `--no-glb` — visual
inspection of placement is assumed for every run.

**GT-detector run (in flight):** `sam3d_layout` on ProcTHOR uses GT masks by rendering
`GTObject.mesh`; `ProcThorSource(gt_mesh="box")` attaches the OBB as a box mesh so this path
works before real asset meshes (AI-8) — a faithful GT-box-detector ceiling for the Objects
rows. Run those `--no-geometry` (box-vs-mesh Chamfer is meaningless). Native pixel-perfect
THOR instance masks are a later refinement.

**SAM3D job identity — content hash → stable logical key (2026-07-13, load-bearing fix):**
AI2-THOR RGB (and a few depth edge pixels) are NOT byte-reproducible across renders, so the
old pixel-content `_job_hash` made the collect pass miss every cached SAM3D output (all
generations orphaned). `run_sam3d` now takes a `job_key` (`{source}_{scene}_i{instance}_
{framing}`, object_track: `t{track_id}`) → readable, reproducible `job_id` (also makes the
queue browsable by scene). Content hash is now only the fallback for disk datasets. Verified
idempotent across re-renders (93 jobs, 0 dup). See [[sam3d-queue-design]].

**First campaign (scene-by-scene, per Chris):** queue is per-experiment
(`results/procthor_procthor_sam3d_layout_3scene/sam3d_queue`). Processing 137 first, then
200 (52) + 428 (197). SAM3D worker ~24 s/job on the 5090 (reloads pipeline per job).

**Native masks >> box masks (2026-07-13) — use native.** ProcTHOR scene 137, GT-detector
`sam3d_layout`, no ICP:

| | box-mask (proxy) | **native THOR mask** | Replica room0 ref |
|---|---|---|---|
| F1 @IoU.25 | 0.29 | **0.56** | 0.77 |
| recall@0.5 | 0.03 | **0.29** | 0.35 |
| Scan2CAD | 0.022 | **0.14** | 0.16 |
| centroid med | 0.088 m | **0.047 m** | 0.058 m |
| scale err med | 0.44 | **0.30** | 0.29 |
| placed / GT | 93/93 | 88/93 | 43/43 |

Native masks ~2× F1, 6× Scan2CAD. 137 now ≈ Replica (gap = 93 objects incl many small vs
43). **scale_err 0.30 == Replica** → this is SAM3D's inherent scale limit, not a ProcTHOR
artifact — motivates Phase-3 Sim(3). Runs: `results/procthor_procthor_137_{collect(box),
native}/`; GLB per scene under `<run>/137/scene_compare_lite.glb`.

**First full 3-scene ProcTHOR result (native GT-detector `sam3d_layout`, no ICP, 2026-07-13):**
`results/procthor_procthor_sam3d_layout_3scene/` (per-scene GLBs under `<scene>/`).

| scene | placed/GT | F1@.25 | recall@.5 | Scan2CAD | centroid | rot | scale err | dup |
|---|---|---|---|---|---|---|---|---|
| 137 | 88/93 | 0.56 | 0.28 | 0.13 | 4.7 cm | 10.0° | 0.30 | 0.01 |
| 200 | 42/52 | 0.62 | 0.21 | 0.14 | 7.0 cm | 7.8° | 0.33 | 0.00 |
| 428 | 175/197 | 0.64 | 0.24 | 0.10 | 6.0 cm | 8.3° | 0.30 | 0.02 |
| **agg** | 101.7/114 | **0.61** | **0.25** | **0.12** | **5.9 cm** | **8.7°** | **0.31** | **0.01** |

Consistent across scenes (F1 0.56–0.64), ~89% coverage, near-zero duplicates (GT detector).
scale err ~0.31 everywhere = the SAM3D ceiling → Phase-3 Sim(3) target. 4 jobs quarantined
to `input_failed/` (scene 200 i47–i50). **This is our Objects-rows column** (our metric
defs; reconcile to the coworker's — AI-7 — before publishing). Mesh rows still need AI-8.

**Phase-3 registration on the 3 scenes, agg (full ablation):**

| metric | layout | +ICP | +TEASER(gated) | +scale | **+scale+ICP** |
|---|---|---|---|---|---|
| F1@.25 | 0.61 | 0.71 | 0.52 | 0.51 | 0.66 |
| recall@0.5 | 0.25 | 0.33 | 0.20 | 0.26 | **0.38** |
| Scan2CAD | 0.12 | 0.13 | 0.09 | 0.26 | **0.31** |
| centroid | 5.9cm | 4.2cm | 5.0cm | 6.0cm | **2.6cm** |
| rotation | 8.7° | 4.2° | 9.2° | 9.8° | 4.7° |
| scale err | 0.31 | 0.33 | 0.31 | 0.12 | **0.14** |

**`scale+ICP` is the Phase-3 win** (`sam3d_layout_scale_icp`, `results/procthor_procthor_
sam3d_layout_scale_icp_3scene/`). Depth-extent scale-fit (`_fit_scale_to_extent`: rescale
the mesh so its OBB extent matches the **multi-view-fused masked-depth OBB extent**) cuts
scale err **0.31→0.14** and **Scan2CAD 0.12→0.31** (Scan2CAD gates on ≤20% scale, so fixing
scale unlocks it); ICP then fixes pose (centroid 2.6cm, rot 4.7°). **Validated by the
depth-extent check** (`scripts`/scratch: masked-depth extent == GT to ~1-2% on observed
axes, thin axis 0.83 single-view → 1.00 fused — so depth *does* give metric scale, and the
fused multi-view cloud recovers the unseen axis; retroactively justifies Phase-2 fusion).
Caveats: (a) **scale alone hurts F1** (0.51) — must pair with ICP for pose; (b) on the
*loose* F1@.25, ICP-only (0.71) edges scale+ICP (0.66) — a known loose-threshold artifact
(pose matters more than scale at IoU .25); the honest metrics (recall@.5, Scan2CAD, scale)
all favour scale+ICP. See [[teaser-registration-finding]] for the full arc.

**Prior negative result (kept for the record):**

| metric | layout | +ICP (rigid) | +TEASER (Sim3) |
|---|---|---|---|
| F1@.25 | 0.61 | **0.71** | 0.31 |
| scale err | 0.31 | 0.33 | 0.35 |

- **ICP (`sam3d_layout_icp`) works**: fixes pose (rotation −4.5°, F1 +0.10) but rigid → can't
  rescale (scale flat, Scan2CAD gated by the residual ~0.31 scale). Runs:
  `results/procthor_procthor_sam3d_layout_icp_3scene/`.
- **TEASER++ Sim(3) vs depth — TRIED, GATED, SHELVED (negative result).** `sam3d_layout_teaser`
  (`results/..._teaser_gated_3scene/`): even with a safety gate (accept only if the Sim(3)
  lowers mesh→depth residual vs layout) it still **loses to layout** (F1 0.52 < 0.61 < ICP
  0.71). Gate accepts ~37% of objects and those genuinely reduce residual (~1.8 cm), but the
  **median accepted scale is 0.67 — TEASER shrinks meshes.** Root cause is fundamental: the
  masked depth is a **partial one-sided** view, so a scale DOF fits it best by shrinking the
  mesh onto the visible sliver — lowers residual (gate accepts) but worsens true 3D extent/
  IoU. This is exactly why rigid **ICP helps** (no scale DOF → only fixes pose) and **TEASER
  hurts**. Ungated it was worse still (F1 0.31, 18° rot — bad FPFH corr on the hallucinated
  mesh). **Conclusion: registering to partial sensor depth cannot fix SAM3D's scale error**
  (target under-constrains the unseen extent). Scale needs a different lever: a genuinely full
  fused surface, upstream generation (SAM 3), or a scale prior — NOT TEASER-vs-depth.
  Solver + build verified fine (unit test recovers a known Sim3 to 3%, `tests/test_teaser.py`).
  Built from source (not on PyPI): local Eigen prefix + venv pybind11 → `_teaserpp.so` in
  `.venv/.../teaserpp_python/` (a `uv sync` drops it — rebuild from `~/build/TEASER-plusplus/`).
  See [[teaser-registration-finding]].

  **Phase-3 verdict: keep ICP (pose win); TEASER-Sim(3)-vs-depth shelved.**

**Two robustness bugs fixed while getting here:** (1) SAM3D worker infinite-retried a failed
job (reloading the pipeline each time → hung the whole queue on one bad job); now failures
quarantine to `input_failed/` (`run_sam3d_worker.py`). (2) degenerate masks (<2×2 bbox from
tiny/occluded objects) crashed SAM3D; `sam3d_layout` now skips masks below `min_mask_px`/
`min_mask_dim`. Also: SAM3D job identity switched from a pixel content-hash to a stable
logical `job_key` — AI2-THOR renders aren't byte-reproducible, so the content-hash cache
missed on collect (see [[sam3d-queue-design]]).

**Open / next:** (1) **AI-7** — get the coworker's exact metric defs + 10th id + split
(comparability-critical; our `evaluate()` uses our own matching defs for now). (2) Run the
real method rows — `sam3d_layout` (GT-mask, full-frame) + `object_track` (YOLOE) over the
9 scenes → the Objects rows. Needs the **SAM3D worker** running (AI-1 env is ready) +
`--extra detector` re-synced. This is the compute campaign; Claude-doable, not gated. (3) **AI-8
DONE (2026-07-14)** — MolmoSpaces `isaac/objects/thor` (~1 GB) downloaded + GT-mesh loader
wired (`data/thor_assets.py`: usd-core→trimesh, `assetId`→mesh, fit-to-OBB; `--gt-mesh asset`,
`--extra mesh`). Validated on scene 200 (46/52 real meshes, span↔OBB median ~1.00, oracle
geo_recall@5cm 0.987). Mesh **numbers** now just need the campaign run with `--gt-mesh asset`
against real predictions + a top-down Footprint IoU metric. **No re-run needed for already-run
scenes:** new `eval/rescore.py` (`python -m r2s3d_core.eval.rescore <result_dir> --source
procthor --scene 200 --gt-mesh asset --export-glb`) rebuilds predictions from the persisted
`scene_graph.json` (posed SAM3D GLBs) and re-evaluates + re-exports the compare GLB with real
GT meshes — reproduces the Objects column exactly (s200 icp f1 0.538 / recall 0.481 ✓) and
adds the first detector-driven Mesh-row numbers: **s200 `object_track_icp` scene_chamfer
0.148 m, geo_recall@5cm 0.475, micro_f1 0.387, macro_f1 0.288** (vs geom f1 0.538 — label
errors cost the micro/macro gap). Compare GLB now shows asset GT meshes, not boxes. (4) **Coworker-comparable named metrics — Option A DONE**
(2026-07-14): `eval/metrics.py` now emits label-aware `micro_f1`/`macro_f1`/`per_class`,
named per-scene counts (`matched/objects/predictions_per_scene`), the coworker "average
chamfer" (`chamfer_symmetric_mean_m` = our `chamfer_l1`/2), and a **scene-level, class-free
geometry** block (`scene_chamfer_mean_m`, `geo_recall/geo_precision/geo_fscore@tau` — pooled
pred vs GT surface points, no matching/labels; NaN until GT meshes land, AI-8). Headline
`f1` stays label-agnostic and is reported alongside micro/macro (not conflated). Tests in
`tests/test_metrics.py` (19 pass). Still **Option B** (pending, needs AI-7 confirm +
per-detection track provenance): many-to-one F1, fragmentation, merge rate, pairwise
P/R/F1 — the association-quality family (coworker's `compute_track_metrics`); wire from
`ObjectTrack` detection→track ids + GT instance ids. Footprint IoU still blocked on AI-8.
(5) **USD/Isaac-Sim scene export (follow-up, not eval-blocking).** The eval is format-
agnostic (normalises GLB preds + USDA GT to `trimesh` surface samples), so this is a
downstream *deliverable*, not a metric requirement. Deferred until Objects/Mesh numbers are
locked. When done: walk `scene_graph.json` (already carries per-object pose + mesh path) →
write one USD `Xform` per object referencing its mesh posed by `T_world_mesh`; `usd-core`
is now in the env (`--extra mesh`), so no new dep. Lets a reconstructed Real2USD scene load
into Isaac Sim (sim/nav demo, paper figure, robot deploy). Do NOT convert our GLB assets to
USDA on the eval path — zero metric gain.

## Phase 2 — detail (detector-in-sim ObjectTrack)

**Design (per Chris):** drive the ObjectTrack pipeline with a **real detector (YOLOE) over
the Replica RGB frames**, not GT masks — to measure pipeline degradation vs the perfect-mask
ceiling, study detector prompting, and show multi-view association + late-merge collapsing
detector fragmentation. See [[phase2-detector-in-sim]], `PHASE_SPECS.md §Phase 2`.

**Built (ROS-free, torch-free core):** `r2s3d_core/tracks/` (Observation/ObjectTrack lifecycle,
HSV-histogram appearance re-ID, 1 cm voxel-hash fused cloud, diverse top-6 view buffer,
association cascade + late-merge, `run_tracker`) and `r2s3d_core/detect/` (DetectionSet disk
cache + seeded corruptions torch-free; `yoloe.py` detector step behind the `detector` uv
extra). Methods `object_track` / `object_track_naive` reuse Phase-0 SAM3D placement; the ICP
variant registers against the track's fused multi-view cloud. **Tests: 50 pass** (`uv run
pytest -q`), incl. association/lifecycle/fusion/late-merge on a synthetic fixture + cache
round-trip + object_track wiring. Per-scene association/merge **debug HTML**
(`results/phase2_debug/room0_tracks.html`).

**Detector (YOLOE `yoloe-11l-seg.pt`, GT-vocab prompt, full detection recall diagnostic):**

| scene | GT obj | detection recall | median mask IoU |
|---|---|---|---|
| room0 | 43 | 0.35 | 0.82 |
| room1 | 27 | 0.70 | 0.66 |

So the detector *caps coverage* (finds 35–70% of GT objects), but masks are good when it fires
— the degradation is dominated by detector recall, not mask quality. (Prompting study: gt-vocab
only; generic/pf modes pending.)

**Fragmentation cleanup — the headline Phase-2 win (verified without the SAM3D worker; these are
track counts):** re-ID heals detector track breaks + late-merge folds overlapping tracks →
one track per truly-detected object.

| | room0 naive | room0 **full** | room1 naive | room1 **full** |
|---|---|---|---|---|
| SAM3D invocations | 56 | **15** | 37 | **9** |
| tracks / GT | 1.30 | **0.35** | 1.37 | **0.33** |
| tracks rejected (short) | 20 | 0 | — | — |
| late-merges | 0 | 5 | 0 | — |

room0: 76 detector track-ids → naive 56 mature (20 short fragments rejected) vs full **15
mature ≈ the ~15 GT objects actually detected** (0.35·43). tracks/GT collapses to ≈ detector
recall — association recovers the true object count from fragmentation. Runs:
`results/phase2_{room0_gt,room0_gt_naive,room1_gt,room1_gt_naive}/run.json`. Detections cached
at `results/detections/gt/<scene>/` (regenerate: `python -m r2s3d_core.detect.run`).

**Placement degradation — detector-driven vs the GT-mask ceiling (SAM3D meshes generated;
`object_track` layout, no ICP):**

| room0 | GT-mask ceiling (`sam3d_layout`) | naive (detector) | **full (detector)** |
|---|---|---|---|
| F1 @IoU.25 | 0.77 | 0.36 | 0.35 |
| Scan2CAD acc | 0.163 | 0.116 | 0.070 |
| centroid err med | 0.058 m | 0.077 m | 0.128 m |
| rotation err med | 14.1° | 11.1° | 12.5° |
| **duplicate rate** | 0 | **0.47** | **0.00** |
| **SAM3D calls** | 43 | 56 | **15** |

room1 (no GT-mask ceiling computed): full F1 0.44 / S2C 0.00 / cent 0.11 m / **dup 0.00 / 9 calls**
vs naive F1 0.47 / S2C 0.074 / cent 0.09 m / **dup 0.41 / 37 calls**.

**Read:** (1) the real detector roughly **halves F1/Scan2CAD vs the perfect-mask ceiling** — the
degradation is *recall-limited* (YOLOE finds only 35% of room0 GT), not mask quality. (2) full vs
naive: placement quality is ~flat, but full **kills the duplicate rate (0.47→0.00) and cuts SAM3D
calls ~4×** — that is the ObjectTrack contribution (v1's visible duplicate failure, fixed). (3)
Honest caveat: full's Scan2CAD dips slightly below naive (0.116→0.070) — consolidating to 15
objects gives fewer shots at the 20/20/20 gate; the lever for coverage is detector recall (better
prompt / SAM 3), **not** disabling association. This is the identical-instance precision/recall
tradeoff flagged in REWORK_PLAN §2.9; inspect in the debug HTML.

Runs: `results/phase2_{room0_gt,room0_gt_naive,room1_gt,room1_gt_naive}/run.json`; detections
`results/detections/gt/<scene>/`; debug HTML `results/phase2_debug/room0_tracks.html`.

**Still open:** prompt-mode study (generic/pf vs gt-vocab), split-corruption stress, `object_track
--icp` (fused-cloud registration) row, all-8-scene aggregate, and the (optional, non-blocking) ROS
wrapper — a bag `SequenceSource` backend covers recorded-robot eval without live ROS nodes; the
live wrapper is only for on-robot demo + repointing the 3 nodes + `ply_frame_utils` at `frames/`.

**▶ NEXT SESSION — continue Phase 2.** Env is set up (`uv sync --extra detector --extra dev`;
torch 2.8.0/torchvision 0.23.0 cu128 on the 5090). Detector caches + run.jsons for room0/room1
are committed as records but the binaries (detections.npz, SAM3D meshes) are gitignored/regenerable.
To reproduce or extend:
```
# detect (writes results/detections/<prompt>/<scene>/) — GPU
uv run python -m r2s3d_core.detect.run --source replica --scene room0 --prompt gt --diagnose
# track + place (queues SAM3D jobs; run the sam3d-objects worker; re-run to collect)
uv run python -m r2s3d_core.eval.run --phase phase2 --scene room0 --method object_track \
    --detections results/detections/gt --debug-html results/phase2_debug --name room0_gt
```
Next tasks, in order: (1) **prompt-mode study** — rerun detect with `--prompt generic` and
`--prompt pf`, compare detection recall + downstream F1 (does GT-vocab flatter the result?);
(2) `object_track --icp` (fused-cloud registration) row vs no-ICP; (3) **split-corruption stress**
(`--det-split-prob 0.5`) to show late-merge cleaning injected fragmentation; (4) extend to the
all-8-scene aggregate. SAM3D worker cmd is in any pending-jobs log line. Tests: `uv run pytest -q`
(50 pass). Design/decisions: `PHASE_SPECS.md §Phase 2`; rationale: [[phase2-detector-in-sim]].

## Current focus

Phase 0 done: SAM3D worker validated end-to-end on the 5090 and the real `sam3d_layout`
row produced on room0. A metric bug was found and fixed: rotation/scale are now
axis-labeling-invariant (min-volume OBB axes are unordered; naive R-vs-R comparison had
inflated rotation 110°→28°, scale 107%→54%). Placement composition validated independently
(posed mesh sits 3.8 cm from its own depth cloud).

**Full-frame vs crop finding (2026-07-08, changes the motivating story):** feeding SAM3D
the **full image + whole-scene depth pointmap** instead of a tight object crop roughly
halves layout error — the crop starves SAM3D's pointmap normalization of scene context.
Room0, 43 objects, same view, only framing differs:

| room0 | CROP (`--crop`) | FULL (default) |
|---|---|---|
| F1 @IoU.25 | 0.58 | **0.77** |
| recall@0.5 | 0.047 | **0.349** |
| Scan2CAD acc | **0.00** | **0.163** |
| centroid err med | 0.10 m | **0.058 m** |
| rotation err med | 28.5° | **14.1°** |
| scale err med | 0.54 | **0.29** |
| chamfer-L1 med | 0.14 m | **0.069 m** |

So the original "SAM3D layout is terrible (Scan2CAD 0)" result was substantially a
crop-feeding artifact on our side, not pure SAM3D badness. Full-frame is **now the
`sam3d_layout` default** (`config["full_frame"]`, `eval.run --crop` to compare). Results:
`phase0_replica_sam3d_layout` (full) vs `..._crop`. Ablation: `scripts/ablation_full_vs_crop.py`
(+ `results/ablation_full_vs_crop/compare.html`); input diagnostic: `scripts/viz_sam3d_inputs.py`.
Caveats: one scene / one view per object; 2D silhouette still ~1.5× the mask even when 3D
OBB scale is ~right → residual shape hallucination, not scale. See [[sam3d-full-frame-beats-crop]].

Variant B (`sam3d_layout_icp`) done. **2×2 ablation (framing × ICP), room0 — separates the
two error sources:**

| | crop | crop+ICP | FULL | FULL+ICP |
|---|---|---|---|---|
| Scan2CAD | 0.00 | 0.00 | 0.16 | **0.19** |
| scale err | 0.54 | 0.54 | 0.29 | 0.30 |
| rotation | 28.5° | 13.7° | 14.1° | **9.5°** |
| recall@0.5 | 0.05 | 0.12 | 0.35 | **0.47** |

ICP fixes **pose but not scale** (scale err unchanged 0.54→0.54 / 0.29→0.30; Scan2CAD stays 0
for both crop variants — rigid ICP can't rescale). Full-frame fixes **scale** (and shape:
chamfer 0.14→0.069). They are complementary → **FULL+ICP wins every scale-sensitive metric**.
This still motivates Phase 3 Sim(3) (scale err ~0.30 remains). ICP ran on the *cached*
full-frame SAM3D outputs — no regeneration. Runs: `phase0_replica_sam3d_layout{,_crop,_icp,_icp_crop}`.
Gotcha: don't headline F1@IoU.25 (0.25 is loose enough that crop+ICP's pose fix beats FULL+ICP
despite wrong scale); use recall@0.5 / Scan2CAD / F@5cm.

**ICP target is SINGLE-VIEW (a second lever, likely capping ICP).** `_masked_depth_cloud`
uses only the best-view frame → ~20.5k pts/object (a partial one-sided sliver). Fusing the
object's masked depth over all views gives ~568k pts/object (**27.8×**), much closer to the
full GT surface. Registering a full mesh to a partial target lets it slide along unobserved
directions. Fusing the target before ICP = Phase 2 ObjectTrack multi-view fusion (cf.
`demo_go2_wAccumPC.py`). **Tested** (`--icp-accumulate`, `config["icp_accumulate"]`,
run `..._icp_accum`): fusing over all views helps ICP **modestly** — F1 0.814→0.860,
F@5cm 0.786→0.811, centroid 3.8→3.2 cm, chamfer 5.5→5.1 cm — but does **nothing for scale
(0.303→0.303) or Scan2CAD (0.186)** since ICP is rigid, and rotation was already converged
(9.5°). Caveat / do-not-misread: this ran on *clean Replica GT depth + GT masks*, where single-view
already mostly works — so the "modest" accumulation gain **understates** multi-view's real
value. On a real robot (sparse/noisy depth, noisy per-frame masks) multi-view fusion is the
main lever for scale disambiguation + pose stability, and a track also enables best-view
selection for generation. That robustness — not this sim accumulation number — is the Phase-2
motivation (see kickoff note above). Residual scale ~0.30 still needs Phase-3 Sim(3). Clouds for inspection:
`results/ablation_full_vs_crop/icp_{target_singleview,target_accumulated[,_lite],source_posed,source_afterICP}.ply`
(script `scripts/export_icp_clouds.py`). Phase 1 done: `r2s3d_core/frames/`
is now the single source of frame transforms (SAM3D shape chain + Go2 body chain),
verified equal to v1 `ply_frame_utils` at 1e-9, with `config/go2_calibration.yaml`,
boundary validation (`frames/validate.py`), and round-trip/regression tests; Phase 0
numbers reproduce exactly. Deferred: repointing the 3 ROS nodes + `ply_frame_utils` at
`frames/` (do it when the Phase 2 ROS wrapper is built — touches the ROS package).

Next major work: (a) confirm the full-frame win on the **all-8-scene aggregate** (room0 only
so far); (b) **Phase 2** (ObjectTrack node). Full-frame + full-frame ICP already done on room0.

**▶ PHASE 2 — UNDERWAY (see "Phase 2 — detail" above for verified numbers).** Core ObjectTrack
library + YOLOE detector-in-sim built and tested; fragmentation cleanup verified on room0+room1.
**Why multi-view matters — the real point, NOT just point-cloud accumulation:**
1. **Best-view selection for generation** — SAM3D shape/scale quality depends heavily on the
   input view; a track lets us pick the best view(s) to feed it, not just whatever frame.
2. **Robustness on a real robot** — Phase 0/the ablation ran on *clean Replica GT depth + GT
   masks*, where single-view already mostly works (that's why the accumulation gain looked
   "modest" — do NOT read it as "multi-view barely helps"). On a real Go2, depth is sparse/noisy
   and per-frame masks are noisy/inconsistent; **multi-view fusion is the main lever that
   disambiguates scale and stabilizes pose under that noise.** Phase 2 is fundamentally about
   real-robot robustness, and that is where multi-view pays off most.

## Phase 0 — detail

**Package:** `humble_ws/src_Real2USD/r2s3d_core` (uv, Python 3.10, ROS-free).
`uv sync --extra dev && uv run pytest -q` → 23 passing.

Done & verified:
- SequenceSource + Replica backend; frame conventions verified against `room0_mesh.ply`
  (raw poses are OpenCV-optical; world already Z-up) and `oriented_bbox` decoded
  correctly (local-frame center; 100% GT mesh-vertex containment).
- Metrics: exact OBB-IoU, Hungarian matching, centroid/rotation(symmetry)/scale errors,
  Chamfer + F-score@5/2cm, Scan2CAD accuracy, scene P/R/F1, duplicate rate, count ratio.
- Runner (`python -m r2s3d_core.eval.run`) → `results/<phase>_<name>/run.json`; table
  script. Oracle scores perfectly on all 8 Replica eval scenes (323 GT objects);
  oracle_noisy degrades correctly.
- `sam3d_layout` (+ICP variant B) implemented against the disk-queue worker; placement
  math + best-view selection unit-tested.

Blocked:
- Actual `sam3d_layout` numbers (the paper's motivating layout-error experiment) — see
  [AI-1](ACTION_ITEMS.md). Code is ready and input-hash-cached.

Next step: Phase 1, or fill AI-1 to get the SAM3D baseline row.
