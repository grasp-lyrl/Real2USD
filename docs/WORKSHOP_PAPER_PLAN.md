# WORKSHOP PAPER PLAN — SeMaNa @ IROS 2026 (extended abstract)

_Created 2026-07-18. This is the working plan for the 2–4 page workshop-paper rewrite.
Strategy source: `REWORK_PLAN.md`; numbers source: `STATUS.md` + `results/*/run.json`._

## Venue facts (these reshape the strategy)

- **Workshop:** SeMaNa — Semantic-Aware Mapping and Navigation (IROS 2026).
  CFP: https://federicorollo.github.io/SeMaNa-Workshop/sections/cfp.html
- **Format:** Extended Abstract, IEEE RAS double-column, **2–4 pages** (refs excluded),
  PDF ≤10 MB, figures encouraged.
- **Review:** double-blind, ≥2 reviewers. **Non-archival** — explicitly welcomes work
  "currently under review" and "prioritizes discussion and exchange of ongoing ideas
  rather than novel, previously unpublished research."
- **Deadline: 2026-07-29** (~11 days from creation). Notification 2026-08-27.
- **Consequence:** the twice-fatal "limited novelty / stitched modules" complaint is **not
  a rejection criterion here**. Goal shifts from *prove novelty* → *land the reframe + back
  it with the sim numbers we already have*.

## Locked decisions (2026-07-18)

1. **Name/title:** rename to **lead with the method** — drop "USD" from the headline;
   name after instance-forward asset mapping / metric object placement. USD demoted to
   sim-export serialization. Title candidates (pick during drafting):
   - "Making a Shape Generator a Scene Creator: Instance-Forward Asset Mapping for
     Metric-Semantic Scene Graphs"
   - "Beyond Shape: Placing Single-Image 3D Generations in a Metric Robot Scene Graph"
   - "Instance-Forward Asset-Centric Mapping: Turning SAM 3D into a Scene Creator"
2. **Primary story:** lead with **C1+C3** — "SAM 3D isn't enough" (shape ≠ scene) + our
   metric placement. C2 (instance-forward multi-view) is the *mechanism* that supports it.
3. **Real-robot demo:** **include**, compact + qualitative — one figure + short paragraph
   (the v1 Go2 + LLM-nav "deployed on a real robot" money shot). NOT a quantified nav
   metric (old `distance-navigated` was criticized).

## ⚠ Coworker-benchmark decision (2026-07-19, Chris) — LOAD-BEARING

The coworker will **not** be a co-author. Firm consequences:
- **Do NOT publish his computed baseline numbers** (Hydra/DAAAM/ConceptGraphs/Khronos/… from his
  ProcTHOR table) — his unpublished work product, and unverifiable by us.
- **We MAY use his metric *definitions*** (Chamfer confirmed == our `scene_chamfer_mean_m`; greedy-XY
  matching; class-free recall) — standard/published methods, fine to adopt for OUR own numbers.
- **The "appear as a column in his holistic benchmark" premise is retired** as a contribution. The
  ProcTHOR eval infra is repurposed for a **self-contained** evaluation.
- **Self-run field baseline — SOLVED with Clio (2026-07-19).** Clio was already run on the 4 Go2
  scenes (old machine); `/data/Clio/*.graphml` scored directly vs Supervisely GT + our metrics
  (`scripts/score_clio_baseline.py`; already in the GT frame, no reconciliation). Clio ≈ v2 on loose
  open-set recall (cd_f1@1m 0.535) but ≪ v2 on strict placement (IoU-F1 0.039 vs ~0.174, scale_err
  0.55) → the C3 differentiator, on real data, coworker-independent. **This is the workshop's field
  baseline — no need to install ConceptGraphs or run any method for the submission.** See STATUS
  "CLIO BASELINE".
- **Deferred to resubmission (NOT gating the workshop):** ConceptGraphs on our data (from-scratch
  Python + 5090 cu128 risk); **ScanNet/Scan2CAD** (strongest field home — public citable numbers +
  real meshes for the asset claim; access available per Chris but not downloaded, mesh GT needs
  Scan2CAD+ShapeNet + adapter).

## Thesis (one sentence)

> A single-image generative model (SAM 3D) makes excellent object *shapes* but cannot
> build a *scene*: turning it into a scene creator requires an instance-forward, multi-view
> front-end that decides *what objects exist* and a metric-localization back-end that
> decides *where and how big* they are — yielding an asset-centric scene graph whose nodes
> are Sim(3)-registered, simulation-ready meshes, not labeled point clusters.

### Three claims, each backed by numbers we already have

| # | Claim | Evidence (from STATUS.md / run.json) |
|---|-------|--------------------------------------|
| **C1** | SAM 3D is a shape prior, not a localizer | Scene-scale layout study: scale err **~0.31** consistent Replica room0 (0.29) + ProcTHOR (0.30–0.33). Full-frame≫crop (F1 0.58→0.77). |
| **C2** | Instance-forward multi-view tracking builds the object set robustly (vs. clustering) | Per-frame recall 0.35–0.50 → **per-track 0.71**. Fragmentation: tracks/GT 1.3→0.33, SAM3D calls ~4× fewer, **duplicate 0.47→0.00**. Ref bar: HOV-SG/ConceptGraphs instance mAP≈0 (SuperMap Tbl III). |
| **C3** | Owning localization gives precise placement (the axis the scene-graph line skips) | scale-fit+ICP: **scale 0.31→0.14, Scan2CAD 0.12→0.31, centroid 2.6–4.5 cm, rot 3.6–4.7°**; detector-driven `object_track_icp` iou_f1@.25 **0.545**. |

**Node-payload ablation — controlled support for C1+C3 (added 2026-07-19, `GENERATION_ABLATION_PLAN.md`).**
Same front-end / tracks / depth / scoring; vary ONLY the node payload: **asset** (SAM 3D mesh +
scale-fit + ICP) vs **cluster** (the track's fused observed cloud — what clustering methods output).
The cluster is built the faithful way (SOR+DBSCAN denoise → OBB; ConceptGraphs/HOV-SG convention);
the raw un-denoised cloud is an unfair strawman (edge-bleed inflates its box) and is kept only as an
ablation. **Result (s200 val), stated honestly:**
- **Supports C1** (SAM 3D is not a localizer): the depth-only cluster already localizes / covers well
  — footprint IoU the cluster even *wins* (0.464 vs 0.420), cd_f1@1m + scale ~tie. Location/coverage
  come from depth+front-end, NOT generation.
- **Supports C3** (generation's value = a *well-posed, simulatable* metric asset): the asset's real
  remaining win is **strict metric-box quality** — 3D-IoU 0.545 vs **0.386**, Scan2CAD 0.151 vs
  **0.097**, centroid 4.5 vs 7.8 cm — plus the qualitative deliverable metrics can't score (a
  watertight, canonically-oriented, **sim-ready mesh** vs a point cluster).
- **Honest caveat to write in:** on clean sim with a strong detector the geometry delta is MODEST
  (~1.4–1.6×), not dramatic. Generation should help most under partial/noisy depth (real robot,
  occlusion), but real Go2 GT is cuboid-only → shape fidelity unmeasurable there. Lead C1+C3 on the
  *well-posed simulatable asset* framing, NOT "generation dramatically improves geometry accuracy."

**Positioning:** instance-forward / tracking-by-detection (the side SuperMap RSS'26 won on)
vs. clustering scene graphs (**Hydra, Khronos** — these cluster a labeled map). Differentiator =
**node payload is a registered, simulation-ready mesh with a well-posed metric box** — a row the
ID/seg-only line can't fill. vs. MetaScenes: autonomous-from-partial-egocentric-views, no
human-in-the-loop curation.

**⚠ Front-end vs field — a "~5× win" claim is NOT yet defensible; the blocker is the metric def, not
gt-vocab (2026-07-19).** Tempting number: our same-front-end cluster footprint ~0.44 vs the coworker
table's 0.006–0.095. Two confounds were suspected; open-vocab runs resolved one and elevated the other:
- **gt-vocab prompting — checked, NOT the driver for footprint/geometry.** Re-ran the cluster with a
  genuinely open-vocab detector (`generic` YOLOE + CLIP labels): footprint held at **0.445** (vs
  gt-vocab 0.464); the gt→open-vocab cost is RECALL (65 vs 73 objects) and LABEL F1 (cd_micro
  0.398→0.291), i.e. the **Objects** rows, not footprint/geometry. So the footprint gap does NOT
  dissolve under open-vocab.
- **Metric definition (AI-7 Q5) — UNCONFIRMED, now the primary uncertainty.** The field matches very
  few objects/scene (Hydra 5.6, DAAAM 3.5) vs our cluster's ~65 (cd_f1@1m 0.67). If their Footprint
  IoU is per-object / recall-weighted rather than scene-level pooled (ours), their low numbers reflect
  coverage/match-count, not per-object surface quality — a metric-axis difference, not a front-end one.
Also our front-end is **architecturally ~ConceptGraphs/HOV-SG** (instance-forward; the clustering
distinction is real only vs Hydra/Khronos). **GATE before any cross-method footprint claim: confirm
Q5 with the coworker (one question, not a compute campaign).** Regardless, **lead with the controlled
asset-vs-cluster ABLATION** (same front-end/prompting/metric). For a fair FIELD entry, report the
open-vocab (`generic`/`pf`) rows, not the gt-vocab ones. See GENERATION_ABLATION_PLAN.md
"Cross-method front-end gap".

## Section skeleton (target 4 pp)

1. **Intro + positioning** (¾ col) — scene-creator thesis; clustering-vs-instance-forward;
   curated-vs-autonomous; 3 claims as contributions.
2. **Method** (1–1.25 pp) — ObjectTrack front-end (assoc + fused cloud + best-view +
   late-merge); retrieve-or-generate node (SAM 3D shape-only, once per track); localization
   back-end (depth-extent Sim(3) scale-fit + ICP + confidence gate); **USD export = one
   sentence, serialization only**.
3. **Experiments** (1.5–1.75 pp) — TWO quantitative legs:
   (a) **Sim (ProcTHOR)** — controlled node-payload + registration ablation (Table 1), GT meshes
   measure shape fidelity; (b) **Real robot (Go2, 4 scenes)** — **v2 vs Clio** metric-placement
   comparison + v2-internal contribution ablation (Table 2); Clio is the coworker-independent field
   baseline, and the internal ablation (not v1) isolates what our method buys. Plus perception
   robustness (Table 3) + Figs 2–3.
4. **Real-robot deployment** (¼ col + 1 fig) — v1 Go2 + LLM-nav money shot, qualitative (the
   *deployed* story on top of the Table-2 *quantitative* real-robot result).
5. **Conclusion / limitations** (¼ col) — offline/near-online; asset maps approximate
   surfaces; static single-session; recall@0.5 ceiling on real depth (shared with baselines);
   sim generation delta modest on clean depth.

## Figures (3) + Tables (3) — the full experimental plan (near-final; numbers in hand)

**Fig 1 — teaser/pipeline:** robot → object tracks → registered assets → USD in Isaac Sim (refresh v1
fig; read as instance-forward).
**Fig 2 — C1:** SAM 3D `make_scene()` vs ours vs GT, with the scale-err number.
**Fig 3 — C2:** per-frame vs per-track recall bar (0.35–0.50 → 0.71).
**Fig 4 — ★ "why generate" (the generation-value figure; n=10, 212 objects):** unobserved-surface
reconstruction vs observation coverage — oracle-placed asset vs clean cluster. Two curves that DIVERGE
as coverage drops: low-coverage (86% unseen) asset **0.081 m** vs cluster **0.232 m** (2.9×, 68/69
objects); near-tie at high coverage. 166/212 objects favor the asset. Generation's value concentrates
in the partial-view regime real robots face. Inset qualitative panel: armchair at 44% observed →
generation fills the unobserved base (red=seen, blue=generated) vs GT.
`scripts/gen_shape_completion.py` + `run_shape_completion_val10.sh`. Honesty: oracle placement isolates
the shape prior (pair with the end-to-end coverage note that placement is observation-limited).

### Table 1 — SIM placement + node-payload ablation (ProcTHOR val; the controlled C1+C3 experiment)
Everything varies ONE factor at a time on the same front-end/tracks/depth/metrics. Cols: 3D-IoU F1@.25
/ Scan2CAD / centroid / scale / Chamfer / Footprint-IoU. Rows: GT-mask ceiling; detector-driven
`object_track` under registration ablation (layout / +ICP / +scale+ICP); and the **node-payload
ablation** — denoised **cluster** (our sim clustering baseline) vs **asset**. Numbers in hand (s200):
asset 3D-IoU **0.545** vs denoised-cluster **0.386**, Scan2CAD 0.151 vs 0.097, footprint 0.42 vs 0.46
(cluster ties/wins — reported honestly). Aggregate over the 10 val ids is the only remaining run.

### Table 2 — REAL-ROBOT (4 Go2 scenes): v2 vs Clio (external baseline) + v2-internal ablation
Same Supervisely cuboid GT, same v2 `evaluate()`, all in the GT frame. Per-scene TP is small (1–6) so
we report the mean + the loose/strict split, and trust cross-method consistency, not single-scene medians.

**Headline — two-way, ours vs a real independent method:**

| method (mean over 4 scenes) | IoU-F1@.25 | recall@0.5 | centroid (m) | scale-err | cd-F1@1m | class-free@1m |
|---|---|---|---|---|---|---|
| **Clio** (Maggio RA-L'24, real method) | 0.039 | 0.00 | 0.324 | 0.551 | 0.535 | 0.687 |
| **v2 (ours: scale-fit+ICP+gate)** | **0.174** | 0.00 | **0.19** | **0.35** | **0.63** | **0.75** |

Story: on **loose open-set recall** (cd-F1@1m / class-free) the two are comparable — objects get found
and roughly localized. On **strict metric placement** (IoU-F1, scale-err) v2 pulls clearly ahead (4–5×
Clio on IoU-F1, lowest scale-err). That is the whole thesis in one table: *the field already finds
objects; we place them metrically.* recall@0.5 = 0 for both is the honest shared ceiling (real depth +
partial views) → future work, not hidden. (Clio boxes CLIP segments → report both the loose column,
fair to Clio, and the strict column, our differentiator.)

**Contribution ablation — v2-INTERNAL, single-variable on the SAME front-end** (this, NOT v1, carries
"what our method buys"): ±depth-extent scale-fit → the C3 scale claim (e.g. hallway scale-err
0.375→**0.240**); ±precision gate → IoU-F1 + precision up on **all 4** scenes (0.089→0.104, 0.156→0.182,
0.190→0.267, 0.111→0.143). Clean because detector/labels/tracks are held fixed.

**v1 is NOT a comparison row (decision 2026-07-19, Chris).** The v1→v2 delta bundles several changes at
once (multi-view consolidation + scale-fit + gate) AND is confounded (different detector, open-vocab vs
gt-vocab labels, only 2/4 scenes), so it cannot cleanly attribute the difference to any one change — the
v2-internal ablations do that far better. v1 appears only as a **one-line provenance note** ("v2
supersedes our IROS'26 pipeline; the scale-fit + multi-view consolidation are the changes"), not a
numeric row. (v1 numbers stay in STATUS for the record, not the paper.)

### Table 3 (or inline) — perception robustness (C2, internal)
Per-frame → per-track recall (0.35–0.50 → 0.71), duplicate/fragmentation cleanup (dup 0.47→0.00,
tracks/GT 1.3→0.33), SAM3D-call reduction (~4×). Measured on our own pipeline (a mechanism claim, NOT
a cross-method superiority claim — see the front-end caveat above).

## Why the experiments are defensible (write this into the paper explicitly)

The reviewer worry is "stitched modules / unfair comparison." Each experiment is built to pre-empt it:

1. **The sim comparison is a CONTROLLED ABLATION, not a system-vs-system bake-off.** We do NOT drop an
   external clustering system into sim — that would confound front-end differences (detector,
   association) with the thing we want to measure. Instead we hold the *entire* front-end / tracks /
   depth / metric fixed and swap ONLY the node payload (asset ↔ denoised observed cluster). This is
   strictly more rigorous than an external baseline: the delta is *attributable to generation alone*.
   **This is the answer to "why not run someone else's clustering on sim" — the cluster payload IS our
   clustering baseline, and making it internal is what makes the claim clean.** The cluster is built
   the way clustering scene-graph methods build nodes (SOR+DBSCAN denoise → OBB; ConceptGraphs/HOV-SG
   convention), so it faithfully represents that class of method without importing its confounds.
2. **The real-robot comparison uses a real, independent method (Clio) — coworker-independent.** We do
   not reuse anyone's private benchmark numbers; we ran Clio ourselves and scored it with the same GT
   and metric code as our own method. Same frame, same `evaluate()`, no per-method tuning.
3. **Metrics are honest and dual-reported.** We report BOTH loose (1 m centroid, where baselines are
   competitive) and strict (3D-IoU / Scan2CAD / scale, our differentiator), and we flag the shared
   ceilings (recall@0.5=0 on real; modest sim generation delta) rather than hiding them. Chamfer is
   verified identical to the community definition; we lead the mesh claim on it.
4. **Two regimes, each playing to what it can measure.** Sim (ProcTHOR, real GT meshes) measures shape
   fidelity the asset node adds; real Go2 (cuboid GT) measures metric placement where the scale-fit
   earns its keep (SAM 3D scale ~3× off on real depth). Neither over-reaches: sim can't claim real-world
   robustness, real can't claim shape fidelity — stated as such.

## Completion status — this reads as near-finished work

- **DONE / numbers in hand:** sim payload ablation (s200), sim registration ablation (3-scene + s200),
  real-robot v2 4-scene table, v1 baseline, **Clio baseline**, **★ generation-value shape-completion
  experiment (Fig 4: asset completes the unseen surface 2× better than the cluster at low coverage,
  18/22 objects)**, Footprint-IoU + cluster-payload code (103 tests pass), open-vocab robustness
  (generic). Real-robot deployment figure assets exist (v1).
- **REMAINING (small, bounded):** (a) **cluster val-10** (running, free) → gives Claim B ("generation
  does NOT localize; the observed cluster localizes fine across 10 scenes") at n=10; **asset stays
  s200** (decision — Claim C is carried by s200 ablation + 4-scene real-robot scale-fit + the sim-ready
  mesh point, so val-10 asset SAM3D is not needed); (b) draft prose + build Figs 1–3 from `run.json`
  (aggregate via `scripts/agg_paper.py`); (c) runtime line (SAM3D ~24 s/obj).
- **Claim split (frames the whole eval):** A instance-first = perception recall (not the payload
  ablation); B generation≠localization = cluster ≈ asset on coarse metrics, n=10 via free cluster; C
  generation adds well-posed box + sim-ready mesh = s200 ablation + real-robot scale-fit + qualitative.
- **NOT blocking (resubmission):** ScanNet/Scan2CAD, ConceptGraphs-on-our-data, coworker's ProcTHOR
  column (dropped — see coworker-benchmark decision).

## Answers to prior reviews (both rounds) baked into the design

| Prior complaint | Answer |
|---|---|
| Limited novelty / stitched modules | Non-archival venue; novelty = instance-forward composition + placement-eval axis |
| USD claim unjustified | Demoted to sim-export; lead with placement |
| Insufficient baselines / 3 categories | ProcTHOR full-vocab (~75 obj/scene), ceiling-vs-detector, registration ablations, clustering ref bar |
| No runtime | SAM3D ~24 s/job + per-stage timing; offline/near-online honesty |
| No occlusion/clutter | Cluttered multi-room ProcTHOR under a real detector |
| Writing/figure clutter | Extended-abstract discipline, 3 figs max |

## Dataset & eval matrix (proposed 2026-07-18 — sim breadth + real-robot)

**De-risk insight:** SAM3D meshes cache per `(scene, track, framing)` — *registration-
independent*. One SAM3D pass per scene → every registration ablation row (layout / +ICP /
+scale+ICP / reproj-scale) is free as an eval re-run. So the only thing to lock before
fanning out is **detector + prompt mode + framing**, NOT the registration method.

| Tier | Data | Status on disk | Effort | Role in paper |
|---|---|---|---|---|
| **1** | **ProcTHOR ×10 canonical ids (val)** — 137,200,428,434,534,569,573,683,771,912 | only **s200** run so far; GT assets at `~/Data/datasets/molmospaces/isaac` | Low (background compute, no adapter) | **Headline Table 1** (n=1→n=10; kills "narrow eval") |
| **2** | **Own Go2 bags + v2 (4 labeled scenes: smalloffice-0/1, hallway-1, lounge-0)** | bags at `/data/go2/lidar/*`; **3D-cuboid GT** at `evaluations/supervisely/*.pcd.json` (`position`+`rotation`+`dimensions`) + parser `real2usd_eval/parse_supervisely_bbox_json.py` (**USDA files are v1 outputs, NOT GT — ignore**) | Medium (bag `SequenceSource` backend; port Supervisely→GTObject OBB; Go2 frame chain fragile) | **Quantitative real-robot placement table** (centroid / 3D-IoU / rot / scale / label F1 — box GT, no mesh Chamfer); this is where the **scale-fit contribution pays off** (SAM3D scale ~3× off on real depth) |
| **3** | **Clio apartment head-to-head** | `/data/clio_datasets/apartment` + `Clio-Eval` (ROS1 Noetic) | High (ROS1 adapter + rerun Clio + osR/osP reconcile) | **Stretch / "in progress"** comparability table; non-archival welcomes preliminary; never a submission blocker |

**Promoted config to lock before fan-out (recommendation):** detector = YOLOE **`gt`-vocab,
full-frame** (strongest validated: iou_f1@.25 0.545, multi-view recall 0.71); `generic`/`pf`
as an open-vocab robustness row if they finish. **Promoted registration = ObjectTrack +
best-view SAM3D + depth-extent Sim(3) scale-fit + ICP**, with `object_track_icp` (no
scale-fit) as the ablation row. Story: sim shows placement precision (scale-fit ≈ ICP on
clean depth); **own bags show the scale-fit earns its keep** (SAM3D scale ~3× off on real).

**Resolved 2026-07-18:** own-scene GT = **Supervisely 3D cuboids** (4 scenes), NOT the USDA
files (those are v1 outputs). **To verify (D3–4):** does the Clio-Eval ROS1 harness run on
this box at all?

**RealSense is the PRIMARY real-robot source (2026-07-18, corrected).** All 4 Go2
scenes have proper RealSense bags under `/data/go2/rs/` with **rgb8 color + dense
hardware-aligned depth (~65–90% valid) + `/utlidar/robot_pose` in the GT odom frame** —
strictly better than the lidar path. `data/realsense.py` `RealSenseSource` (source
`realsense`/`rs`): depth-triggered RGB-D, `depth_mm/1000`, pose via `frames.T_odom_cam_go2`,
no cloud/projection/undistort needed. Scene map: lounge-0, smalloffice-0/1, hallway-1.
Validated: lounge-0 len 1014, depth 65–72% valid, pose in GT frame; hallway GT-box overlay
lands on objects. **Correction:** an earlier note here claimed the lounge/smalloffice RS
color was "broken (==depth)" — that was a topic-selector bug (`endswith("color/image_raw")`
also matched `aligned_depth_to_color/image_raw`); the RS color is genuine rgb8. The lidar
path below is retained as a secondary sensor / LiDAR-vs-RealSense depth ablation.

**Lidar-path adapters BUILT + validated (2026-07-18).** Secondary real-robot source:
- `data/rosbag.py` `RosbagSource` — pure-Python bag read (`rosbag` uv extra = `rosbags`,
  no ROS), accumulates `/point_cloud2` (already odom-frame) into one cloud, poses each
  `/camera/image_raw` via `frames.T_odom_cam_go2` (extrinsic verified == v1 `ProjectionUtils`
  to 1e-16), projects the cloud → metric depth (`lidar_depth.project_cloud_to_depth`),
  **undistorts RGB** (plumb_bob k1≈−0.34) into the pinhole-K frame so RGB↔depth align.
- `data/supervisely.py` `load_supervisely_gt` — 3D cuboids → `GTObject` (radians xyz-euler,
  center, full extents, label aliases → chair/table/door). Registered as source `rosbag`
  with a scene map (smalloffice-0/1, hallway-1, lounge-0).
- Validated: lounge-0 cloud 331k pts, GT XY ⊂ cloud XY ⊂ camera-traj, depth median 5.5 m,
  RGB↔depth overlay checked (`results/rosbag_debug/`). 89 core tests pass (+9 new).
- Known limitation: `__iter__` streams all image blobs sequentially (no random access) —
  fine for a stride-1 detector pass, wasteful for sampling.

## Detector-choice ablation — "why SAM 3 and not YOLOE?" (anticipated reviewer Q)

**Disambiguate loudly (CLAUDE.md gotcha):** SAM 3 = 2D open-vocab detect/seg/**video-track**;
SAM 3D = single-image mesh generator (shape). They are different models; reviewers will
conflate them.

**Positioning (prose, essential, free):** the detector is a swappable front-end; the
contribution is the detector-agnostic **3D metric back-end** (fused multi-view cloud,
best-view-for-generation, 3D re-ID, Sim(3) placement). "Just use SAM 3" gives better 2D
tracks but no metric 3D pose/scale/asset placement — you still need the whole back-end. SAM 3
is a complementary better front-end, not a replacement. YOLOE chosen = open, light,
robot-deployable.

**Experiment (upside, gated):** only YOLOE is integrated (`detect/yoloe.py`); SAM 3 is
HF-gated (AI-6, open) + unintegrated → risky in 11 days. Plan:
- **Pragmatic cross-detector row:** Grounding-DINO + SAM2 (ungated; *SuperMap's exact stack*)
  → proves detector-agnosticism + parks next to SuperMap's front-end.
- **Real SAM 3 row:** only if HF gated access is requested **today** (long pole); frame
  "in progress" (non-archival allows).
- **Free partial evidence already in hand:** YOLOE prompt-mode study (`gt` 0.71 / `generic`
  0.60 / `pf` 0.71 recall) shows placement holds as the front-end varies.

## Writing checklist additions

- [ ] Disambiguate **SAM 3 (detector) vs SAM 3D (shape generator)** explicitly on first use.
- [ ] State the detector-agnostic framing before any detector-choice discussion.

## Open tasks before/while drafting

- [ ] Confirm headline number source: **val aggregate across the canonical 10 ids** vs
      **s200 alone** (generic/pf prompt rows were "RUNNING 2026-07-17" — lock or fall back).
- [ ] **Node-payload ablation val-10 aggregate** (framing now locked → C1+C3 lead + ablation support):
      run `object_track_cluster` (denoised, free — no SAM3D) on the 9 remaining ids + confirm asset
      meshes cached for all 10, aggregate asset vs denoised-cluster for the Table-1 payload row.
- [ ] Produce a **runtime line** (SAM3D 24 s/job + per-stage) — answers every prior review.
- [ ] Build Fig 2 (SAM3D layout vs ours vs GT) and Fig 3 (recall bar) from run.json.
- [ ] Refresh Fig 1 teaser from v1 assets; pick a real-robot frame for the deployment fig.
- [ ] Related-work paragraph: SuperMap, Hydra, ConceptGraphs, HOV-SG, MetaScenes,
      SceneComplete, SAM 3D (disambiguate from SegmentAnything3D).
- [ ] Anonymize; IEEE RAS double-column template; ≤10 MB; IEEE-style refs.

## Action item (human)

- arXiv preprint **2510.10778** is non-anonymous. Double-blind + existing preprint is
  normally acceptable at a non-archival workshop given the changed title/framing — just do
  not cite it as "our prior work." Surfaced here for provenance; see `ACTION_ITEMS.md` if it
  needs tracking.

## 11-day timeline

- D1–2: lock framing/numbers; outline + claim sentences.
- D3–4: Figs 2–3 + Table 1; confirmatory runs (prompt rows, runtime).
- D5–7: full draft.
- D8–9: internal read; cut to 4 pp; related work; polish.
- D10: anonymize, format, refs.
- D11: buffer + submit before 2026-07-29.
