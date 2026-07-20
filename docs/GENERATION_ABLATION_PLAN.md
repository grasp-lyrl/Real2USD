# GENERATION ABLATION + BENCHMARK COLUMN — next-session plan

_Created 2026-07-19. Executable handoff. Read `STATUS.md` "Real-robot eval leg" + Phase-5
sections first for current numbers; this plan is the next focus._

## Why (the question to answer)

Does per-instance **generation** (SAM3D asset) make the scene graph geometrically *more
accurate*, or does it only add *richness/context*? We decomposed it (STATUS): metric
**location** comes from depth+registration (a cluster localizes as well — v1 cluster centroids
≈ v2 asset centroids); generation's unique value is **shape completion, canonical orientation,
and simulatable geometry** (the Mesh rows where every clustering method scores ~0: Footprint IoU
0.006–0.095, Chamfer 0.49–1.35 m in the coworker's ProcTHOR/MolmoSpaces table).

**Goal:** a confound-free internal ablation that isolates the generation step, + the missing
Footprint IoU metric, so we can (1) state rigorously whether generation improves geometry
accuracy and (2) fill our column (Objects + Mesh) in the coworker's benchmark. This is the
"is the extra compute worth it" proof (WORKSHOP_PAPER_PLAN C1+C3).

Run on **ProcTHOR val** (real GT asset meshes exist → can measure shape fidelity; the real Go2
scenes have cuboid-only GT so can't). Canonical 10 ids: 137,200,428,434,534,569,573,683,771,912.

## Benchmark targets — coworker's table (transcribed 2026-07-19, ProcTHOR/MolmoSpaces slice)

The per-method numbers we are landing next to (source screenshot in
`results/Screenshot 2026-07-09 at 3.08.29 PM (1).png`). Only our two rows — **Objects** and
**Mesh** — are reproduced; `-` = not reported by that method.

**Mesh — OUR DIFFERENTIATOR (every clustering method is near-floor here):**

| method | Chamfer (m) ↓ | Footprint IoU ↑ |
|---|---|---|
| Kimera-Semantics | **0.489** | **0.095** |
| Khronos | 0.499 | 0.090 |
| DAAAM | 0.670 | 0.055 |
| Hydra | 1.351 | 0.006 |

→ **Numbers to beat: Chamfer < 0.489 m, Footprint IoU > 0.095.** Chamfer's def is CONFIRMED
(AI-7 #5: scene-level pooled symmetric = our `scene_chamfer_mean_m` ✓) — Footprint IoU's exact
def is the ONE remaining gap (Q5, marked optional/non-blocking; spec a scene-level 5 cm occupancy
default and reconcile later). Our Phase-5 sim `sam3d_layout` scene_chamfer was ~0.10 m and detector
`object_track_icp` ~0.10 m — ~5× under Kimera's 0.489. That gap is large enough to warrant a visual
sanity-check that our scene_chamfer pools ALL objects' surfaces (not matched-only) before publishing,
but it's a magnitude check, not a definition question. This is the whole
"generation is worth it" case: a shaped mesh vs an observed cluster. The **cluster payload
(ablation B/A)** should land in the same 0.05–0.10 Footprint-IoU / 0.5–1.4 m Chamfer band as the
clustering methods — i.e. the ablation's cluster row *is* a same-front-end stand-in for Hydra/
Khronos on the Mesh rows, and asset − cluster is the generation delta stated cleanly.

**Objects — best-in-class to beat (matching def per AI-7: greedy-XY ≤1 m, label-aware):**

| metric | best reported | method | ours so far (val s200 object_track) |
|---|---|---|---|
| Micro F1 | 0.279 | Khronos | 0.388 (CLIP, 2026-07-15) |
| Many-to-one F1 | 0.384 | ConceptGraphs | TBD |
| Macro F1 | 0.195 | Khronos | TBD |
| Class-Free Geo Recall (1 m) | 0.873 | Clio | ~0.96–1.0 (gt tracker) |

Full field for context: Micro F1 — Hydra .074 / DAAAM .065 / ConceptGraphs .218 / Clio .101 /
Khronos .279 / HOV-SG .044 / OpenGraph .207. Objects/Scene ≈ 29–104 (ours ~75 unfiltered).
NOTE our Objects numbers are recall-bounded by detector coverage (~0.71 multi-view) and are *not*
the headline — Mesh is. Report Objects for completeness; lead with Mesh + the asset-vs-cluster delta.

## Open-vocab s200 set — COMPLETE (2026-07-19); ablation holds under open-vocab

All s200 val, same tracker; payload/prompt varied. (gt-asset footprint not shown — its run.json
predates the footprint metric; = 0.420 from the earlier rescore.)

| config | footprint | chamfer | f1@.25 | cd_micro_f1 | centroid | scale_err | n_pred |
|---|---|---|---|---|---|---|---|
| gt asset | (0.420) | 0.104 | **0.545** | 0.388 | 0.045 | 0.269 | 72 |
| gt cluster (denoise) | **0.464** | 0.120 | 0.386 | 0.398 | 0.078 | 0.311 | 73 |
| generic cluster | 0.445 | 0.131 | 0.367 | 0.291 | 0.074 | 0.229 | 65 |
| generic asset | 0.389 | 0.120 | **0.418** | 0.304 | 0.067 | 0.290 | 65 |
| pf cluster | 0.188 | 0.351 | 0.259 | 0.237 | 0.051 | 0.372 | 185 |
| pf asset | 0.126 | 0.494 | 0.304 | 0.232 | 0.059 | 0.215 | 157 |

**Findings:** (1) **Ablation holds under open-vocab** — generic asset f1@.25 **0.418 > 0.367** cluster
(asset wins strict 3D-IoU); cluster ties/wins footprint (0.445 > 0.389) — same pattern as gt-vocab.
(2) **gt→generic cost is recall (65 vs 73) + label F1** (cd_micro 0.398→0.291), geometry holds — so
gt-vocab is a fair-detector confound for the *Objects* rows only. (3) **`pf` is unusable** — 185
over-segmented tracks (dup 0.20–0.23, 438 junk labels) crater footprint (0.19)/chamfer (0.35) and F1;
NOT a paper row. **Working open-vocab config = `generic`.**

## ★ GENERATION-VALUE EXPERIMENT — shape completion of the unobserved surface (2026-07-19)

The strongest defensible "generation is worth it" claim (`scripts/gen_shape_completion.py`, s200,
22 large objects, max-extent ≥ 0.5 m). Isolates SHAPE completion from placement: oracle-place the
asset (map its OBB-local shape into the GT OBB, best of 24 cube rotations by fit — no deforming),
clean the cluster (clip observed points to the GT OBB), measure mean NN distance from the
**UNOBSERVED** GT surface to each payload.

| coverage bin | n | unseen % | asset recon | cluster recon | asset wins |
|---|---|---|---|---|---|
| low <0.3 | 5 | 91% | **0.085 m** | 0.172 m | 5/5 |
| mid 0.3–0.6 | 8 | 50% | **0.066 m** | 0.120 m | 7/8 |
| high >0.6 | 9 | 15% | 0.063 m | 0.080 m | 6/9 |

**18/22 objects: the asset reconstructs the unseen surface better; the gap DIVERGES at low coverage**
(2× at <0.3, near-tie at >0.6) — exactly the partial-view thesis. **Qualitative panel: track 128, a
chair at 2% coverage → asset completes it to 7.4 cm, cluster 24.5 cm (3.3×).**

**★ VAL-10 (n=10, DONE 2026-07-19) — STRONGER at scale. 212 objects across 10 scenes:**

| coverage bin | n | unseen | asset recon | cluster recon | asset wins |
|---|---|---|---|---|---|
| low <0.3 | 69 | 86% | **0.081 m** | 0.232 m | **68/69** |
| mid .3–.6 | 51 | 56% | 0.068 | 0.131 | 46/51 |
| high >0.6 | 92 | 18% | 0.074 | 0.083 | 52/92 |

**166/212 objects overall; at low coverage the asset completes the unseen surface to 8 cm vs the
cluster's 23 cm (2.9×), on 68/69 objects.** Gap collapses at high coverage (little unseen). This is
Fig 4 at n=10 — the strongest, most robust generation-value result. Qualitative panel: armchair
`results/paper/_figs/completion_panel_s200_t26.png`. Runner: `scripts/run_shape_completion_val10.sh`.

**Honesty guardrails (write these in):** (1) oracle placement uses GT pose → isolates the *shape
prior's* information, NOT an end-to-end capability; pair it with the end-to-end coverage diagnostic
(`gen_coverage_diag.py`) which shows placement is coverage-limited. Combined claim: *the shape prior
genuinely carries the unseen geometry, and realizing it end-to-end needs enough observation to
register.* (2) the 24-rotation search only picks discrete orientation (standard shape-retrieval eval),
does not fit shape to GT. (3) s200-only (asset is s200-only per the val-10 decision); n=22 large objects
is a solid single-scene figure — note val-N as future strengthening. This is the paper's Fig for "why
generate": divergence plot + the chair panel. See WORKSHOP_PAPER_PLAN.md.

## ★ C1 TABLE — SAM3D-native localization vs our registration (val-10, DONE 2026-07-19)

`scripts/run_paper_asset_val10.sh` (663 SAM3D meshes, 0 fail) → layout/icp/scaleicp share one gen pass.
`results/paper/sim/asset_{layout,icp,scaleicp}_gt_s*`, agg `scripts/agg_paper.py`:

| variant | IoU-F1 | recall@0.5 | centroid | scale_err | Scan2CAD | cd_f1@1m |
|---|---|---|---|---|---|---|
| **layout** (SAM 3D's OWN predicted scale/rot/translation) | 0.404 | 0.147 | 0.085 | 0.319 | 0.075 | 0.707 |
| **+ICP** (our pose fix) | **0.485** | **0.224** | 0.066 | 0.319 | 0.088 | 0.706 |
| **+scale+ICP** (our scale-fit + pose) | 0.449 | 0.190 | 0.066 | 0.335 | 0.043 | 0.708 |

**C1 confirmed: our registration beats SAM 3D's own localization** — +ICP gives +20% IoU-F1
(0.404→0.485), **+52% recall@0.5** (0.147→0.224), ~half the rotation error (per-scene ~11°→~6°).
**Honest:** on SIM, scale-fit does NOT beat plain ICP (SAM 3D's sim scale is already decent; scale_err
0.335 vs 0.319, Scan2CAD lower) — **scale-fit's win is the real-robot regime** (SAM 3D ~3× off →
0.47→0.24, STATUS real-robot leg). Combined C1/C3: ICP wins on sim, scale-fit wins on real. So `icp` is
the sim headline asset; `scale_icp` is the real-robot recipe.

## The ablation (hold everything fixed except the node payload)

Same ObjectTrack front-end, same tracks, same depth, same GT-mesh scoring. Vary ONLY the payload:

| payload | representation | what it isolates |
|---|---|---|
| **A. cluster** | the track's fused observed point cloud (`t.fused_cloud`); OBB for box metrics, raw points for surface metrics | what clustering methods output (partial observed surface) |
| **B. asset** (current) | SAM3D mesh + depth-extent scale-fit + ICP | generation + our placement |
| (optional C. box) | OBB-as-box mesh | crudest baseline |

**Fairness rule:** the cluster payload MUST use the same tracks and (for the box metrics) the
same scale-fit path as the asset — the only difference is "shaped mesh" vs "observed points."
Do NOT represent the cluster as a box for the *surface* metrics (that inflates the asset's win);
use the real observed points, which is what Hydra/ConceptGraphs/etc. actually have.

**Hypotheses (each outcome is a clean, publishable statement):**
- Asset ≫ cluster on **Chamfer / Footprint IoU** (completion of unobserved surface) → generation
  improves geometry fidelity, not just richness.
- Asset ≈ cluster on **centroid** (both from depth) → generation does NOT improve metric location
  (expected; consistent with v1-vs-v2).
- Asset > cluster on **3D-IoU / unobserved-axis extent** (shape prior completes the far axis the
  robot never sees) → the one place generation should improve metric extent. This is the crux; if
  it holds, it's the strongest "worth it" number.

### ✅ B DONE + FIRST RESULT (s200 val, 2026-07-19) — hypotheses REVISED by the data
`object_track_cluster` (+ `--node-payload cluster`) implemented (no SAM3D, no registration; cluster
box = fused-cloud OBB via `s3d._cloud_obb`, raw points as `surface_pts`). Same front-end/tracks as
the asset run (cluster 73 vs asset 72 objects placed / 93 GT → fair). Head-to-head:

| metric | asset (object_track_icp) | cluster | read |
|---|---|---|---|
| footprint_iou ↑ | 0.420 | 0.402 | **~TIE** |
| surf_fscore@5cm ↑ | 0.591 | 0.590 | **~TIE** |
| scene_chamfer (m) ↓ | 0.100 | 0.136 | asset +26% |
| centroid (m) ↓ | 0.045 | 0.132 | **asset 2.9×** |
| scale_err ↓ | 0.269 | 0.692 | **asset 2.6×** |
| 3D-IoU F1@.25 ↑ | 0.545 | 0.181 | **asset 3.0×** |
| recall@0.5 ↑ | 0.269 | 0.075 | **asset 3.6×** |
| Scan2CAD ↑ | 0.151 | 0.000 | **asset (cluster floors)** |

**The naive "generation completes surface" hypothesis is only WEAKLY true** (footprint/surf-coverage
~tie, Chamfer +26%): the multi-view fused cloud already covers the *observed* footprint, and a
top-down/surface metric doesn't reward the completed far side much. **The real, large win is METRIC
BOX QUALITY** — 3D-IoU 3×, scale 2.6×, centroid 2.9×, Scan2CAD 0→0.15. WHY: the raw cluster OBB is
**surface-biased** (center pulled toward observed faces → 13 cm off) and **oversized** (detector-mask
edge-bleed → scale_err 0.69, the known ~3× contamination); generation + scale-fit + ICP recovers a
well-centered, correctly-scaled, *simulatable* box. So the publishable statement is sharper than
planned: **generation's value is not more coverage — it is a clean, metric, well-posed asset where
the observed cluster gives a biased, oversized, un-simulatable blob.** Runs:
`results/phase0_s200_cluster/` (cluster), `results/procthor_procthor_object_track_icp_s200_val/` (asset).

**What this exposes:** the **generation delta** — asset vs cluster above = the C1+C3 win (well-posed
metric box + sim-ready mesh, not coverage). This is the ONLY controlled front-end/generation claim
(same front-end, same prompting, same metric).

**Cross-method front-end gap — open-vocab check (2026-07-19); a prior hard retraction was an
OVER-correction.** Ran the cluster with a genuinely OPEN-VOCAB detector (`generic` YOLOE + CLIP label
map, `results/phase0_s200_cluster_generic/`) to test whether gt-vocab prompting drove the cluster's
footprint ~0.44 vs the field's 0.006–0.095. It did NOT: open-vocab footprint **0.445** (vs gt-vocab
denoised 0.464), centroid/scale/chamfer ~unchanged; the gt→open-vocab cost is RECALL (65 vs 73 placed)
and LABEL F1 (cd_micro 0.398→0.291) — the label-agnostic geometry/footprint rows hold. So gt-vocab is
a real confound for the **Objects** rows but NOT for footprint/geometry, and the cross-method footprint
gap does not dissolve under open-vocab. **The REAL remaining uncertainty is the METRIC DEFINITION
(AI-7 Q5, unconfirmed):** the field matches very few objects/scene (Hydra 5.6, DAAAM 3.5) while our
cluster matches ~65 (cd_f1@1m 0.67) — if their Footprint IoU is per-object / recall-weighted (not
scene-level pooled like ours), their low numbers reflect coverage/match-count, not per-object surface
quality. **GATE: confirm the coworker's footprint/mesh def (Q5) before ANY cross-method footprint
claim — that, not gt-vocab prompting, is the blocker.** Architecturally our front-end is still
~ConceptGraphs/HOV-SG, so lead with the controlled asset-vs-cluster ablation regardless. See
WORKSHOP_PAPER_PLAN.md "front-end vs field".

**METHODOLOGY NOTE — what the cluster baseline is.** It is NOT a published system (not Hydra/
ConceptGraphs's code) — it is an **internal ablation** of our own pipeline (generation off, front-end
held fixed), which is the correct instrument for isolating our generation step (running another
system would re-introduce the front-end confound). The FIELD comparison is the coworker's benchmark
(their harness ran the real methods). The cluster bridges the two. For the cluster to faithfully
*represent* a clustering-method node, its construction matches how those methods build a node:
**denoise the accumulated cloud (SOR + largest-DBSCAN-cluster) then OBB** — ConceptGraphs / HOV-SG
convention. The first result above used the RAW cloud (edge-bleed-inflated → scale_err 0.69, unfairly
bad). Fair baseline = **denoised cluster** (implemented: `object_track_cluster` default
`cluster_denoise=on`; `--no-cluster-denoise` = raw ablation; `s3d._denoise_cloud`). Do NOT lend the
cluster our depth-extent scale-fit — that is our pipeline machinery, indefensible to give the
baseline; denoising is standard cloud cleaning any clustering method does.

**Step 1 DONE (2026-07-19): denoised cluster on s200 CHANGED THE CONCLUSION** (why the row mattered).
Three-way, same front-end, s200 val:

| metric | asset (icp) | cluster raw | cluster denoised (FAIR) |
|---|---|---|---|
| footprint_iou ↑ | 0.420 | 0.402 | **0.464** (cluster wins) |
| surf_fscore@5cm ↑ | 0.591 | 0.590 | 0.592 (tie) |
| scene_chamfer ↓ | 0.100 | 0.136 | 0.120 |
| cd_f1@1m ↑ | 0.727 | 0.578 | 0.723 (tie) |
| scale_err ↓ | **0.269** | 0.692 | 0.311 (near-tie) |
| centroid (m) ↓ | **0.045** | 0.132 | 0.078 |
| 3D-IoU F1@.25 ↑ | **0.545** | 0.181 | 0.386 |
| Scan2CAD ↑ | **0.151** | 0.000 | 0.097 |

**Denoising closes MOST of the raw-cluster gap.** The naive raw-cluster "asset 3×" numbers were an
artifact of edge-bleed contamination, NOT generation. Against a FAIR (denoised) cluster: footprint
the cluster WINS (0.464 > 0.420); surf-coverage / cd_f1@1m / scale are TIES; the asset's real,
remaining win is the **strict metric-box quality** — 3D-IoU 0.545 vs 0.386 (**1.4×**), Scan2CAD
0.151 vs 0.097 (**1.6×**), centroid 4.5 vs 7.8 cm — i.e. a *well-posed box* (correct center +
complete/canonical extent), plus the qualitative deliverable metrics can't score: a **simulatable
watertight mesh** vs a point cluster. **Corrected conclusion:** on clean sim val with a good detector,
the big robust win is the FRONT-END (C2; both clusters ≫ published clustering 0.006–0.095); the
GENERATION win (C1+C3) is real but MODEST on this regime, concentrated in strict 3D-IoU/Scan2CAD +
the sim-ready mesh. Generation should help most where depth is partial/noisy (real robot, occlusion)
— but real Go2 GT is cuboid-only so shape fidelity can't be measured there (state as a limitation).
Publishing the raw-cluster 3× would have been destroyed by any reviewer running a denoised cluster —
this row is why we gate on it. Runs: `results/phase0_s200_cluster_denoise/` (fair),
`results/phase0_s200_cluster/` (raw ablation).

**VAL-10 CLUSTER AGGREGATE DONE (2026-07-19, n=10):** `object_track_cluster` on all 10 val ids
(`results/paper/sim/cluster_gt_s*`, agg `scripts/agg_paper.py`): iou_f1 **0.405**, footprint **0.446**,
scale_err 0.306, cd_f1@1m **0.686** — closely tracks s200 (0.386), so the ablation's cluster side is
robust across scenes. This is **Claim B's n=10 breadth** (generation NOT needed for localization; the
observed cluster localizes/covers consistently). Asset stays s200 per the decision above (Claim C =
s200 ablation + shape-completion + real-robot scale-fit). Sim table done.

## Deliverables + implementation pointers

### A. Footprint IoU metric — `eval/metrics.py` ✅ DONE (2026-07-19)
**IMPLEMENTED + TESTED.** `geo.footprint_iou(pred_pts, gt_pts, cell=0.05)` (geometry.py) — projects
both pooled clouds to XY, floors to a shared 5 cm grid against the world origin, `|∩|/|∪|` of occupied
cells (Z-invariant by construction). Wired into `_scene_geometry` → new `footprint_iou` key (auto-
aggregated in run.json; added to `eval/table.py` as `footIoU`). `SceneObject` gained a `surface_pts`
field + `sample_surface()`/`has_surface()` so a mesh-less **cluster payload** is scored from its own
points (deliverable B relies on this); the per-matched Chamfer loop uses it too. 7 new unit tests
(identical→1, Z-invariance, disjoint→0, exact half-overlap→1/3, empty→nan, evaluate+cluster paths);
98 tests pass. **First real number — `object_track_icp` asset, s200 val (rescore, reproduces Objects
exactly): footprint_iou = 0.420, scene_chamfer = 0.100 m.** vs coworker Mesh best (Kimera 0.095 /
0.489): ~4.4× Footprint IoU, ~5× Chamfer. Strong asset-payload signal; the **cluster baseline (B) is
the missing half** to state the delta cleanly (+ confirm Q5 def; 0.42 at 5 cm is our default).

_Original spec (retained):_
Top-down (XY) occupancy IoU. Add to `_scene_geometry` (metrics.py:306) — it already pools pred vs
GT surface points via `geo.sample_surface(p.mesh, ...)` (metrics.py:315-324). Add: project pooled
pred pts and GT pts to XY, rasterize to a grid (start 5 cm cells), `IoU = |pred∩gt| / |pred∪gt|`
of occupied cells → `footprint_iou`. Emit both scene-level (default) and, if useful, per-matched
via `evaluate`'s matched loop (metrics.py:373-378, add a 2D variant beside the 3D IoU).
**OPEN: confirm the coworker's exact Footprint IoU definition** (scene-level occupancy vs
per-object; cell size; from mesh footprint vs OBB footprint) — AI item, add to ACTION_ITEMS.
Until confirmed, spec scene-level 5 cm occupancy from posed-mesh surface points.

### B. Cluster payload mode — `baselines/object_track.py` + `SceneObject`
- `_run` (object_track.py:86) per-track loop (~:112-145) currently calls `s3d.run_sam3d(...)` →
  mesh, then registration. Add `config["node_payload"]` ("asset" default | "cluster"). When
  "cluster": skip `run_sam3d`; build the SceneObject from `t.fused_cloud` — OBB (min-volume or the
  existing cloud-OBB helper) for `T_world_obj`/`extents`, and attach the raw points for surface
  metrics.
- `SceneObject` (metrics.py:53) samples `self.mesh` for Chamfer/Footprint. The cluster has no
  faces → add an optional `surface_pts: np.ndarray = None` field; in `_scene_geometry` (:316) and
  `evaluate` (:374) use `p.surface_pts` directly when present, else `geo.sample_surface(p.mesh)`.
  This keeps the cluster's real observed points as its surface (fair).
- Expose as `--node-payload {asset,cluster}` in `eval/run.py` (build_parser ~:203-300; thread into
  the config dict ~:53-81) and/or a method alias `object_track_cluster` in
  `baselines/__init__.py` (AVAILABLE, :42-45). Cluster runs need **no SAM3D** → fast, free.

### C. Runs (ProcTHOR val, `--gt-mesh asset --extra mesh`)
Start on **s200** (asset meshes already cached — `results/procthor_procthor_object_track_icp_s200_val/`
+ its sam3d_queue). Then val-10 aggregate.
- Asset already exists; re-score with Footprint IoU via `eval/rescore.py`
  (`python -m r2s3d_core.eval.rescore <asset_run_dir> --source procthor --scene 200 --gt-mesh asset`).
- Cluster: fresh tracker run (no SAM3D):
  `uv run python -m r2s3d_core.eval.run --source procthor --scene 200 --split val --method object_track
   --node-payload cluster --detections <procthor gt dets> --gt-mesh asset --name s200_cluster`
  (detections: reuse `results/detections/procthor/gt` if present, else `detect.run --source procthor`).
- Compare asset vs cluster on: iou_f1@.25, centroid, scale_err, scene_chamfer, geo_recall, **footprint_iou**.

### D. Fill the coworker's benchmark column (val-10 aggregate)
Objects rows: Micro/Many-to-one/Macro F1, Matched/Scene, Objects/Scene, Class-Free Geo Recall
(all already in `eval/metrics.py` — micro_f1/macro_f1/cd_micro_f1_many_to_one/class_free_recall_1m).
Mesh rows: scene_chamfer_mean_m + footprint_iou (new). Report **asset (our method)** and **cluster
(ablation baseline)** as two rows next to Hydra/ConceptGraphs/HOV-SG/Khronos/etc. Add a **runtime**
line (SAM3D ~24 s/obj) — "N× compute → M× geometry fidelity."
Reconcile metric defs with AI-7 (mostly done: greedy-XY ≤1 m, label-aware micro/macro; split=val).

## Current state (done this session — context for fresh start)
- Real-robot leg DONE: 4-scene v2 table (hallway/lounge/smalloffice-0/1) + v1 baseline (hallway,
  lounge; no v1 smalloffice outputs). See STATUS "Real-robot eval leg". Frame-reconciled v1 scorer:
  `scripts/score_v1_baseline.py`.
- Precision gate IMPLEMENTED (`--track-gate`, default 6-obs/0.40-conf, off by default): improves
  IoU-F1+precision on all 4 real scenes. 91 tests pass.
- ICP-drop probed → INCONCLUSIVE (do NOT drop ICP; scale_icp stays the recipe). Robust reg effect
  = scale-fit lowers scale_err only.
- Diagnostics: `scripts/rs_coverage_diag.py`, `scripts/rs_gate_sweep.py`.

## Open decisions to resolve next session
1. Footprint IoU exact definition — confirm with coworker (blocks the D column's comparability).
2. Cluster surface representation — raw fused points (recommended, fair) vs convex hull vs
   voxel-filled. Raw points = what clustering methods have; use that.
3. Cluster OBB: min-volume (trimesh bounding_box_oriented) vs axis-aligned. Match whatever the
   asset OBB uses for a fair extent comparison.
