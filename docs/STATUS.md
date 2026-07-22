# STATUS — where we are in the v2 rework

**This is the living progress tracker. It is the first doc to read to know the current
state, and the last doc to update at the end of every work session / milestone.** Keep
it honest: "done" means *verified* (tests pass / numbers produced), not "code written".

- Strategy & rationale: `REWORK_PLAN.md` · Interfaces & resolved decisions: `PHASE_SPECS.md`
- **Things only the human can do: `ACTION_ITEMS.md`** (Claude adds to it on every gated dependency)
- Datasets & access: `DATASETS.md` · In-family systems & ideas to steal: `RELATED_WORK.md`

_**★ SESSION 2026-07-22 — VAL-10 ORACLE UPPER BOUND + SAM 3 DETECTOR ABLATION + PAPER TIGHTENED.**
Committed `e33d0d3` (v2-rework; SAM3D queue meshes gitignored, only run.json/CSV provenance tracked).
- **Val-10 oracle campaign DONE** (744 GT-mask SAM3D meshes, 0 fail; `scripts/run_paper_oracle_val10.sh`
  queue+collect, **stride 1** — fixes the old s200 stride-20 confound). Numbers `results/paper/_tables/
  oracle_val10_mean.csv` (regen `scripts/agg_oracle.py`). **Perception ceiling is now val-10**: oracle
  +icp IoU-F1 **0.744** vs detector 0.485; oracle +scale+icp **scale 0.130** vs 0.335 (scale-fit shines
  on clean masks); **oracle layout 0.62 → +icp 0.74** ⇒ the registration back-end is needed EVEN with
  perfect masks (reinforces C1). `tab:oracle` in `paper/main.tex`.
- **SAM 3 detector ablation DONE** (s200, stride 10). New `detect/sam3.py` (HF transformers `Sam3`
  open-vocab backend, `--backend sam3`, batched concept prompts), `scripts/run_s200_e2e.sh` +
  `probe_mask_cleanliness.py`, `results/paper/_tables/s200_frontend_e2e.csv`. **Verdict:** SAM 3 raises
  open-set recall (cd-F1 0.82 w/ gate, best detector-driven) + cleaner masks (leak_frac 0.56→0.34) but
  NOT strict placement — it over-detects (116 tracks/93 GT → precision 0.41; ungated IoU-F1 0.45 <
  YOLOE 0.52), at **~55× detector compute** (O(vocab); no multi-concept single pass). **Default YOLOE.**
  Framed as the "strong" middle of the front-end axis (Appendix `tab:detector`); see [[sam3-detector-cost]],
  [[sam3d-worker-launch]].
- **Paper experiments restructured** (`paper/main.tex`): perception-ceiling subsection reframed to
  "The Front-End Is the Bottleneck" (weak YOLOE → strong SAM 3 → perfect oracle); the two real-robot
  tables combined into one `tab:real` — this **surfaced + fixed a mixed-config bug** in the old vs-Clio
  "v2" row (strict from +gate, loose from +scale_icp). Dropped redundant `fig_c1_sim`; moved the scene
  figure into the ceiling subsection; **all result tables standardized to 2dp** (rotation 1dp).
- **Honest-read correction:** the gated "v2" loose recall (cd-F1 0.50) is BELOW Clio (0.54); the
  loose-recall tie belongs to **+scale_icp (0.63)**, not the gated config. C3 lead stays the paired scale test.
- **NEXT:** write the section PROSE from the `\todo`/POINTS-TO-HIT bullets (Intro/C1–C3/ablations still
  scaffolded, not written); build Fig 1 (pipeline teaser — only remaining placeholder). Paper is
  self-consistent (compiles; 5 tables/3 figures + appendix)._

_**▶ NEXT SESSION FOCUS (2026-07-19): `docs/GENERATION_ABLATION_PLAN.md`** — the confound-free
internal ablation (cluster-cloud vs SAM3D-asset, same tracks) + a new Footprint IoU metric, on
ProcTHOR val, to prove whether generation improves geometry ACCURACY (not just richness) and to
fill our column (Objects+Mesh) in the coworker's benchmark. **Deliverable A (Footprint IoU metric)
DONE + tested (98 tests pass):** `geo.footprint_iou` + `footprint_iou` metric key + `SceneObject.
surface_pts` (for the cluster payload). First real number — `object_track_icp` asset, s200 val:
**footprint_iou 0.420 / chamfer 0.100 m** vs coworker Mesh best (Kimera 0.095 / 0.489) = ~4.4× / ~5×.
Coworker's per-method targets are now transcribed into the plan's "Benchmark targets" section.
**★ GENERATION-VALUE WIN (2026-07-19): shape-completion of the unobserved surface.** The strong,
defensible "why generate" result (`scripts/gen_shape_completion.py`, s200, 22 large objects): oracle-
placed asset reconstructs the UNSEEN surface **2× better** than the clean cluster at low coverage
(0.085 vs 0.172 m), 18/22 objects, gap diverges as coverage drops. Qualitative panel: armchair 44%
observed → generation fills the unobserved base (`results/paper/_figs/completion_panel_s200_t26.png`).
This is paper Fig 4. Honesty: oracle placement isolates the shape prior; pair with the end-to-end
coverage diagnostic (placement is observation-limited). Also this session: **Clio real-robot baseline**
(4 scenes, coworker-independent — v2 ≫ Clio on strict placement), **v1 dropped as a comparison row**
(confounded; internal ±scale-fit/±gate ablations carry contributions), paper structure built out
(`WORKSHOP_PAPER_PLAN.md`), experiment index + naming + aggregator (`docs/EXPERIMENT_MATRIX.md`,
`scripts/agg_paper.py`). **★ ASSET VAL-10 CAMPAIGN DONE (2026-07-19):** 663 SAM3D meshes (0 fail) → C1 table + shape-completion
at n=10. **C1 (SAM3D-native localization vs our registration):** +ICP beats layout +20% IoU-F1
(0.404→0.485), +52% recall@0.5 (0.147→0.224), ~half rotation err; scale-fit neutral on sim (its win is
real-robot). **Shape-completion n=10 (212 objs): asset reconstructs unseen surface 2.9× better than
cluster at low coverage (0.081 vs 0.232 m, 68/69 objs); 166/212 overall** — the strongest generation
claim, now robust across scenes. Runs `results/paper/sim/asset_{layout,icp,scaleicp}_gt_s*`.

**AI-9 RESOLVED (Wayland→Xorg; render on `:0`+gdm auth). VAL-10 CLUSTER DONE (n=10):** iou_f1 0.405,
footprint 0.446, scale_err 0.306, cd_f1@1m 0.686 — tracks s200 (0.386) → **Claim B has n=10 breadth**
("observed cluster localizes/covers consistently across 10 scenes; generation not needed for
localization"). Asset stays s200 per decision (0.545/0.420/0.233/0.727). Runs `results/paper/sim/`,
CSVs `results/paper/_tables/sim_*.csv` (regen via `scripts/agg_paper.py`).

**Deliverable B (cluster payload) + FAIRNESS ROW DONE (s200 val) — conclusion CORRECTED.**
`object_track_cluster` / `--node-payload cluster` (denoise default-on = ConceptGraphs/HOV-SG node
convention; `--no-cluster-denoise` = raw ablation). Three-way asset vs raw vs DENOISED cluster: the
raw-cluster "asset 3×" was edge-bleed contamination, not generation. Against a FAIR denoised cluster:
footprint the cluster WINS (0.464>0.420), scale/cd_f1@1m/surf-coverage TIE, and the asset's real
remaining win is strict metric-box quality only — **3D-IoU 0.545 vs 0.386 (1.4×), Scan2CAD 0.151 vs
0.097 (1.6×), centroid 4.5 vs 7.8 cm** + the simulatable mesh. **Generation (C1+C3) is real but MODEST
on clean sim.** ⚠ **RETRACTED the "front-end ~5× vs the field" idea** — the cluster-vs-published
footprint gap (0.40 vs 0.006–0.095) is CONFOUNDED (our gt-vocab near-oracle prompting vs their
open-vocab; scene-level-pooled vs their unconfirmed footprint def; architecture ~ConceptGraphs). Only
the asset-vs-cluster ABLATION is a controlled claim. See GENERATION_ABLATION_PLAN.md "Step 1 DONE" +
WORKSHOP_PAPER_PLAN "DO NOT claim a front-end ~5× win". NEXT: val-10 aggregate (asset + denoised
cluster); for a fair FIELD entry, produce open-vocab (`generic`/`pf`) rows or cite gt-vocab caveat. This session also finished the
real-robot leg: 4-scene v2 table + v1 baseline + precision gate (`--track-gate`, helps all 4 scenes);
ICP-drop probed and RETRACTED (do not drop ICP). Details below._

_Last updated: 2026-07-22 (val-10 oracle upper bound + SAM 3 detector ablation done; paper
experiments restructured/tightened, committed e33d0d3. **Next: write section prose + Fig 1 teaser.**
Earlier 2026-07-19: real-robot leg + v1 baseline + precision gate; GENERATION_ABLATION_PLAN done.)_

_Prior 2026-07-18 (**FIRST REAL-ROBOT PLACEMENT NUMBERS** — full detect→track→SAM3D→register
pipeline on Go2 hallway-1 via `RealSenseSource`, stride-2 gt masks: 33/57 tracks, scale-fit (reproj)
cuts real-depth scale_err **0.375→0.240** (C3 confirmed on real data), best centroid 0.260 m +
label_acc 0.75; but IoU/Scan2CAD floored — centroid ~0.26–0.31 m is scatter-dominated (systematic
XY bias only ~0.24 m). Pose-only ICP hurts rotation on detector-mask clouds (5.9°→18°). Full table +
next levers in the "Real-robot eval leg" section below.)_

_Prior (2026-07-17): val-s200 **SAM3D layout baseline** produced on the correct houses:
`sam3d_layout{,_icp,_scale,_scale_icp}` on oracle GT masks → Object-F1 **0.94** / iou_f1@.25
**0.66–0.73**, vs object_track's **0.39** / **0.47–0.55** on YOLOE masks. The .39→.94 jump is
the **perception gap** (detection recall ~0.48 + CLIP labels ~0.55), not asset geometry. Scale-fit
HELPS on oracle masks (layout_scale_icp S2C .17→.33) but craters on the YOLOE fused cloud — regime-
dependent. **generic/pf detector prompts through the real pipeline: RUNNING (2026-07-17).**_

**📊 Running metrics visualization (keep adding to it):** a published Artifact compiles the
ProcTHOR s200 experiment metrics into a coworker-facing table (mask-source column, coworker
metric-name mapping, metric dictionary, provenance notes) — regenerated from `run.json`
aggregates, never hand-edited. **URL: https://claude.ai/code/artifact/5896089c-86ab-4a45-9fe5-41f068be0fe3**
As new experiments land (variants, prompts, other scenes), add rows here rather than making
one-off tables. Source HTML lives in the session scratchpad; rebuild + republish the same file
path to keep the URL. See [[procthor-mask-provenance-and-val-ceiling]].

_Prior (2026-07-15): first REAL detector-driven val-s200 run (`object_track_icp` on gt-prompt
YOLOE + SAM3D worker → iou_f1@.25 0.545, centroid 4.5cm, class-free recall 1.0, scan2cad 0.15,
chamfer 0.10). Corrects the earlier "box inflation" scare (SAM3D-free cloud-OBB proxy artifact).
GT mesh orientation fixed: placed by AI2-THOR true rotation not PCA (`_R_unity_euler`+
`place_canonical_by_linmap`, cache v2). Split footgun fixed (`--split` on rescore/detect.run),
CLIP-nearest-in-set label mapping (`eval/label_map.py`, `--label-map clip`)._

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
  big). The real pipeline (SAM3D shape + ICP; SAM3D's native scale, no scale-fit in this run) already
  produces well-sized, well-localized boxes — iou_f1 is a healthy 0.545, centroid 4.5cm. Robust extent estimation is
  NOT the bottleneck; the SAM3D+scale-fit design handles it. (The SAM3D-free tracker numbers
  above remain valid — they use track centroids, which are legit; only the cloud-OBB *extent* was
  the proxy artifact.)
- **GT mesh orientation fixed (2026-07-15):** GT asset meshes were placed by PCA-axis matching
  (`fit_canonical_to_obb`), whose axis-sign ambiguity flipped asymmetric assets (upside-down
  chairs). Now placed by AI2-THOR's true `rotation` (`_R_unity_euler` → `_M_WU @ R`, det -1 LH→RH;
  `place_canonical_by_linmap`), validated by native-mask silhouette IoU (0.497→0.528 mean; kettle
  0.21→0.77) and structurally upright. Moves scene chamfer only (0.104→0.0998; OBB metrics
  axis-invariant, unchanged). GT cache bumped to v2 (stores rotation).
- **Registration/scale-source ablation (val s200, same tracks+meshes, iou_f1 / chamfer / scale_err):**
  `icp` (no scale-fit) **0.545 / 0.104 / 0.269** = current best on iou; `scale_icp` w/ raw fused
  cloud **0.158 / 0.315 / 0.692** (cratered — detector-mask cloud contaminated, OBB ~3× too big);
  **Method A** `scale_source=fused_robust` (SOR + largest-DBSCAN-cluster) **0.376 / 0.132 / 0.513**;
  **Method B** `scale_source=reproj` (clean 2D mask span + median depth, SAM3D aspect for the
  along-ray axis) **0.509 / 0.093 / 0.320** — best chamfer of ANY variant, ~ties icp on iou/centroid.
  So scale-fit no longer HURTS: read scale off the mask, not the cloud. On sim val SAM3D's native
  scale is already decent (icp scale_err 0.27) so reproj can't beat icp on iou here; the payoff is
  real-robot data (SAM3D scale ~3× off). `scale_icp` also now iterates scale↔ICP to convergence
  (`--scale-icp-iters`, default 5; improves pose: rot 1.9°→1.2°). **B2 tested & rejected:**
  `scale_source=reproj_mv` (3rd axis from a 2nd ~orthogonal view instead of SAM3D aspect) is
  slightly WORSE (iou_f1 0.509→0.461) — a 2nd-view silhouette span is a noisier along-ray estimate
  than SAM3D's shape prior. **`reproj` (SAM3D 3rd axis) stays the recommended scale source.**
  Knobs: `--scale-source {fused,fused_robust,reproj,reproj_mv}`.
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
multi-view showcase, few objects; test 137/428). (b) **The depth-CLOUD-extent (fused) scale-fit — the GT-mask
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
| 3 | Localization stack (TEASER++ / ICP / refine) | 🟢 **scale fix found** | **`scale+ICP` is the win**: scale-fit cuts scale err 0.31→0.14 (2.6×) and Scan2CAD 0.12→0.31 (GT-mask, via fused masked-depth extent; on DETECTOR masks the recipe switched to **RGB-mask reprojection** — depth-cloud extent craters there). ICP fixes pose; TEASER-vs-depth shelved. See below. |
| 4 | Reconciliation + export | ⬜ not started | needs Isaac Sim |
| 5 | Benchmark campaign | 🟡 side-thread started | **ProcTHOR/MolmoSpaces scene-graph comparison adapter built + validated** (see below). Main campaign dataset access is the long pole — [AI-2..5](ACTION_ITEMS.md) started early |
| 6 | Paper rewrite | 🟡 planning | **Target: SeMaNa @ IROS 2026 workshop (non-archival, 2–4 pp, deadline 2026-07-29).** Plan + locked framing in `WORKSHOP_PAPER_PLAN.md`: rename to lead with method (USD demoted to sim-export), lead with "SAM 3D shape≠scene + our placement" (C1+C3), include compact real-robot Go2 fig. |

Legend: ⬜ not started · 🟡 in progress / partially blocked · 🟢 done · 🔴 blocked

## Real-robot eval leg — Go2 adapters BUILT + validated (2026-07-18)

For the workshop-paper real-robot placement table (`WORKSHOP_PAPER_PLAN.md` Tier 2), the
Go2 bags now flow through the same `SequenceSource` interface as Replica/ProcTHOR. **Two
sensor sources:**

- **`data/realsense.py` `RealSenseSource` (source `realsense`/`rs`) — PRIMARY.** All 4
  scenes have proper RealSense bags under `/data/go2/rs/` with **rgb8 color + dense
  hardware-aligned depth (~65–90% valid) + `/utlidar/robot_pose` in the GT odom frame**.
  Depth-triggered RGB-D; `depth_mm/1000`; pose via `frames.T_odom_cam_go2`; no
  cloud/projection/undistort. Scenes: lounge-0, smalloffice-0/1, hallway-1. Validated
  lounge-0 (len 1014, 65–72% depth) + hallway GT-box overlay on objects. NOTE: an earlier
  claim that lounge/smalloffice RS color was depth was a **topic-selector bug**
  (`endswith("color/image_raw")` also matched the aligned-depth topic) — RS color is real
  rgb8. Overlays in `r2s3d_core/results/rosbag_debug/`.
  **TIME-SYNC GOTCHA (fixed):** `/utlidar/robot_pose` stamps its `header.stamp` on the
  robot's internal clock — offset from the camera wall clock by **~months** though both
  cover the same recording window — so pairing pose↔depth by header stamp pins every frame
  to one pose (camera never moves → all tracks collapse to one corner, recall ~0). Fixed:
  `RealSenseSource` syncs by **bag record time** (`tsn` from `r.messages`), a single clock
  across topics. (Lidar `/odom` IS on the camera clock — RosbagSource unaffected.)
  **First real-data tracker diagnostic (hallway-1, YOLOE gt-vocab, stride 5, 450 det → 22
  mature tracks, no SAM3D):** per-track centroid recall vs 57 cuboid GT — full (reid+merge)
  **@0.25m 0.07 / @0.5m 0.19 / @1.0m 0.32**, over-seg only 3/57. Association is NOT the
  bottleneck (full≈naive track count, reid/merge lift recall 0.14→0.19@0.5m). Bottlenecks:
  detection recall/coverage + centroid precision (RS extrinsic is the uncalibrated front-cam
  `[0.285,0,0.01]`; masked-depth gives surface- not box-centroid — SAM3D+registration fixes
  the latter). NEXT lever: calibrate the RS→odom extrinsic (v1 had a `realsense_to_lidar_transform`).
  **Detector prompt probe (hallway-1 RS, STRIDE 2, multi-view DETECTION recall vs 57 GT):**
  `gt` 1113 det/3 labels **@0.5m 0.65 / @1.0m 0.88**; `pf` 5779 det/**217 labels** @0.5m 0.89 /
  @1.0m 0.96 (best recall but junk labels → needs CLIP relabel+filter before tracking);
  `generic` 882 det/18 labels 0.60/0.82 (worst — drop). **Stride is the dominant lever**
  (gt @1m 0.32→0.88 from stride 5→2). Recommended working config: **`gt` @ stride 2** (clean);
  `pf` is the recall ceiling if CLIP relabeling is added. GT-box overlays (fixed pose) in
  `results/rosbag_debug/RS_hallway_gtbox_fixed_*.png` show boxes in correct regions, ~0.4m off.
  **Frame reconciliation (2026-07-18):** the v1 `/data/sam3d/*-{lidar,rs}/accumulated_points.ply`
  clouds are in a **first-pose-relative** odom origin (lounge cloud X[-3,13] = GT X[33,43] minus
  the start pose), NOT the absolute-odom GT frame — so they can't directly calibrate the RS
  extrinsic to GT. `RealSenseSource` (absolute odom) IS in the GT frame — that choice is correct.
  If extrinsic calibration is wanted, ICP **my** RS cloud ↔ **my** lidar-bag cloud (both GT frame).
  **FIRST REAL PLACEMENT NUMBERS — hallway-1 (2026-07-18, full detect→track→SAM3D→register
  pipeline, stride 2 gt masks, 33 mature tracks / 57 GT, SAM3D worker drained 33/33 on the 5090).**
  The SAM3D mesh cache is registration-independent, so the 3-row ablation below is one SAM3D pass +
  free eval re-runs (`--sam3d-queue results/phase0_hallway1_rs_icp/sam3d_queue` to reuse):

  | metric | layout | +ICP (pose) | +scale+ICP (reproj) |
  |---|---|---|---|
  | iou_f1@.25 | 0.067 | 0.111 | 0.089 |
  | Scan2CAD | 0.000 | 0.000 | 0.000 |
  | centroid med | 0.292 m | 0.308 m | **0.260 m** |
  | rotation med | **5.9°** | 18.2° | 19.8° |
  | scale_err med | 0.375 | 0.357 | **0.240** |
  | centroid recall@.5m | 0.211 | 0.228 | **0.246** |
  | label_acc | 0.33 | 0.60 | **0.75** |

  **Findings:** (1) **the reprojection scale-fit (RGB-mask span + median depth + SAM3D aspect, NOT the
  depth-cloud OBB) earns its keep on REAL depth** — scale_err
  0.375→**0.240** (~36%) with `scale_source=reproj`, plus best centroid + label_acc. This is the
  paper C3 thesis confirmed in the real regime (on sim val, scale-fit ≈ ICP because SAM3D scale is
  already decent; on real Go2 depth it clearly helps). (2) **[RETRACTED 2026-07-19] The earlier
  "pose-only ICP HURTS rotation (5.9°→18°)" claim was a small-matched-set artifact.** The full
  registration ablation (layout/icp/scale/scale_icp, both scenes) shows only TP=3–6 matched objects,
  so rotation/centroid *medians* are high-variance and not comparable across variants (layout's low
  rotation is over the 3 easiest objects; ICP matches 5–6 incl. harder-rotated ones). On the headline
  **IoU-F1 ICP actually helps/ties** (hallway icp 0.111 > scale_icp 0.089 > layout 0.067; lounge
  icp/scale_icp 0.156 > layout 0.130) — do NOT drop ICP. The robust effect is scale-fit ↓ scale_err;
  registration mode otherwise barely moves iou_f1/cd_f1/coverage. (3) **Everything IoU/Scan2CAD-gated stays floored** because
  the centroid is stuck at ~0.26–0.31 m (vs ~5 cm on sim val), far above what IoU@.25 / Scan2CAD's
  20%-scale gate need. **Centroid residual is SCATTER-dominated** (matched-pred diagnostic: mean
  signed XY offset ≈(−0.16,−0.18) m, |mean|=0.24 vs per-axis std 0.33–0.37, |std|=0.55; Z well-
  aligned, median |Z| 0.16 m). So the uncalibrated front-cam RS extrinsic contributes a modest
  ~0.24 m systematic XY bias, but the DOMINANT error is per-object scatter (depth/mask/surface-
  centroid noise, partial views) — **extrinsic calibration will only partially help** (expect
  median XY ~0.38→~0.30, not sim-level). Runs: `results/phase0_hallway1_rs_{layout,icp,
  scale_icp_reproj}/`. Minor bug: compare-GLB export fails ("Can't export empty scenes") on all 3
  runs — scene_graph.json writes fine (33 objects); visual-inspection GLB needs a fix.
  **C2 COVERAGE DECOMPOSITION (2026-07-18, `scripts/rs_coverage_diag.py`).** WHY does detection
  recall 0.88@1m collapse to mature-track recall 0.37@1m? Decomposed the 57 GT (τ=1.0m): recalled
  **21 (0.37)**, never-detected **15 (0.26)**, localization-lost **12 (0.21)** (a mature track sits
  ≤1m away but greedy 1-to-1 gave it to a neighboring GT — clutter contention + scatter), associable
  fragmentation **5 (0.09)**, sparsity **5 (0.09)**. Mature-track disposition: **33 mature → 19
  distinct GT, 4 duplicate, 10 (30%) FALSE-POSITIVE** (no GT within 1m). **Surprise result: the
  multi-view association is NOT the leak** — fragmentation is only ~9% and late-merge already folds
  18 tracks; the maturation gate is effectively `MIN_ACTIVE_OBS=3` (batch finalize matures every
  ACTIVE track, so `MIN_MATURE_VIEWS=6` is dead code in eval). The real limiters are **(1) track
  PRECISION** — 30% of mature tracks are FP spurious detections surviving ≥3 frames (drives eval
  precision 0.15); **(2) LOCALIZATION scatter** — the same 0.55m scatter that floors placement also
  breaks GT matching (12 GT lost to contention, and obs centroids sit 0.5–1m off box centers:
  never-detected drops 24→15 as τ 0.5→1.0); **(3) detector COVERAGE** — 26% of GT never detected
  within 1m (all 4 outlets, some chairs/tables). Also 166/1113 detections (15%) never became track
  observations (likely no valid masked depth). [[perception-robustness-crux]] said multi-view
  recall recovery was the thesis; on THIS real scene the mechanism works but is bottlenecked by
  precision + localization, not association.
  **Next levers, re-ranked by this evidence:** (a) **track precision/confidence gate** — reject the
  10 FP mature tracks (persistence + label-vote/appearance consistency + fused-cloud density gate);
  lifts precision AND placement F1; (b) **localization scatter** — box-centroid not surface-centroid,
  tighter fused-cloud depth gating (helps placement AND matching/coverage — shared root cause);
  (c) **detector coverage** — `pf` prompt + CLIP relabel, or stride 1 (raises the 0.74/0.88 ceiling);
  (d) RS extrinsic calibration (partial, ~0.24 m systematic only); (e) extend to lounge-0/
  smalloffice-0/1. Association tuning (assoc_reproj, lower gate) is LOW payoff here (~5 GT).

  **TRACK PRECISION GATE — IMPLEMENTED (2026-07-18, lever a).** New opt-in `--track-gate`
  (`tracks/tracker.py` `_passes_precision_gate`, applied AFTER late_merge so merged-in fragments
  count toward persistence): demotes a MATURE track to REJECTED unless `n_obs >= gate_min_obs`
  AND `mean(det_score) >= gate_min_score`. Defaults **6 / 0.40** (tuned by `scripts/rs_gate_sweep.py`
  + full eval; knobs `--gate-min-obs`/`--gate-min-score`). Motivated by the FP-vs-TP characterization:
  FP mature tracks are short-lived + low-confidence (median n_obs 6 vs 17, mean det_score 0.40 vs
  0.50; 9/10 are hallucinated "door"). **Result on hallway-1 `object_track_scale_icp` (reproj):
  cuts FP tracks 29→16, IoU-precision 0.121→0.200, IoU-F1 0.089→0.104, IoU-recall FLAT (0.070,
  same 4 TP), centroid/scale/label unchanged.** Stricter 10/0.45 over-prunes (drops a TP, F1 back
  to 0.088). On the loose class-free-1m metric the gate trades recall for precision (cd_recall
  0.37→0.26) — expected; the honest IoU/Scan2CAD columns (the paper differentiator) improve.
  Default OFF (prior runs reproduce). 91 tests pass (+2 gate unit tests in `test_tracks.py`).
  Runs: `results/phase0_hallway1_rs_scaleicp_gate_{gentle,default}/`.

  **CLIO BASELINE — scored with v2 metrics (2026-07-19, `scripts/score_clio_baseline.py`).**
  Clio (Maggio RA-L'24) was run on the 4 Go2 scenes on an old machine; processed open-set object
  graphs are at `/data/Clio/<scene>.graphml` (object nodes: open-set `name`, `bbox_pos` center,
  `bbox_dim` extents, `bbox_orientation`). **Coordinates are ALREADY in the absolute-odom GT frame**
  (per-scene X/Y ranges coincide with Supervisely GT → NO reconciliation, unlike v1). Scored directly
  vs the SAME Supervisely GT + v2 `evaluate()`. A real published-method baseline on OUR real data,
  **coworker-independent** (we do NOT reuse the coworker's benchmark numbers — Chris 2026-07-19).

  | metric | Clio hallway | Clio lounge | Clio so-0 | Clio so-1 | **Clio mean** | **v2 mean (scale_icp+gate)** |
  |---|---|---|---|---|---|---|
  | n_pred / gt | 46/57 | 36/39 | 14/11 | 14/10 | — | — |
  | IoU@.25 F1 | 0.019 | 0.053 | 0.000 | 0.083 | **0.039** | **~0.174** |
  | recall@0.5 | 0.000 | 0.000 | 0.000 | 0.000 | **0.000** | ~0 (both floored) |
  | centroid med | 0.358 | 0.343 | — | 0.270 | 0.324 | ~0.19 |
  | scale_err med | 0.524 | 0.317 | — | 0.811 | 0.551 | ~0.35 |
  | cd_f1@1m | 0.485 | 0.427 | 0.560 | 0.667 | **0.535** | ~0.63 |
  | class-free@1m | 0.579 | 0.462 | 0.909 | 0.800 | **0.687** | ~0.75 |

  **Read (honest + favorable):** Clio is COMPETITIVE on loose open-set recall (cd_f1@1m 0.535,
  class-free 0.687 — occasionally ties v2, e.g. so-0 class-free 0.909) but FAILS strict metric
  placement (IoU-F1 0.039, scale_err 0.55, recall@0.5=0 on all scenes). It boxes CLIP *segments* with
  no metric-extent step → localizes objects roughly but can't size/pose them. **This is exactly the
  C3 differentiator** (asset+registration → well-sized, well-placed boxes). Caveat: Clio boxes are
  CLIP-segment extents (some tiny fragments), so the strict-IoU gap partly reflects that Clio isn't
  built for metric boxes — report BOTH the loose (Clio ≈ v2) and strict (v2 ≫ Clio) columns; lead the
  differentiator on the strict/scale metrics. Label_acc is noise here (matched sets 0–3 objects);
  `--label-map clip` is an IDENTITY no-op for Clio (its stripped labels chair/table/door are already
  in the GT vocab → don't re-run expecting movement; Clio emits only 3 coarse categories, misses outlet).
  **This is the workshop's field baseline — no need to install/run Clio or ConceptGraphs for the
  submission; ConceptGraphs-on-our-data + ScanNet are resubmission investments.**

  **[PAPER DECISION 2026-07-19, Chris] v1 is NOT a comparison row in the paper.** The v1→v2 delta
  bundles multiple changes (consolidation + scale-fit + gate) AND is confounded (different detector,
  open-vocab vs gt-vocab labels, 2/4 scenes), so it can't cleanly attribute any single change. The
  v2-INTERNAL ablations (±scale-fit, ±gate — same front-end) carry "what our method buys"; the field
  comparison is v2 vs Clio. v1 stays here for the record + a one-line provenance mention in the paper.

  **V1 (OLD real2sam3d) BASELINE — scored with v2 metrics (2026-07-18, `scripts/score_v1_baseline.py`).**
  To know whether v2 actually improves on the paper's v1 method, scored the committed v1 outputs at
  `/data/sam3d/<scene>/scene_graph.json` (+ `glbs_world/` posed meshes) against the SAME Supervisely
  GT + v2 `evaluate()`. **Frame reconciliation (load-bearing):** v1's world frame is absolute-odom in
  orientation+Z but XY-shifted by the first robot pose (`demo_go2.py` subtracts `init_odom["t"][:2]`) —
  so the map to the GT frame is a **pure XY translation**, read authoritatively from the v1-saved
  `step_init/odom_rs.json` (validated: ADD → 15/57 matches on hallway, SUB/raw → 0). Boxes recomputed
  as oriented bboxes from the posed `glbs_world` meshes (v1's `scene_graph` only stores inflated AABBs).
  Old `humble_ws/evaluations` harness CANNOT score these as-is (reads a flat `objects` list, not the
  step-wise on-disk schema; does no frame reconciliation; **no v1 numbers were ever saved**).

  | metric | v1 hallway | v2 hallway (best) | v1 lounge (raw) | v1 lounge (dedup .5) |
  |---|---|---|---|---|
  | n_pred / gt | 33 / 57 | 20 / 57 | 69 / 39 | 39 / 39 |
  | IoU@.25 F1 | 0.067 | **0.104** | 0.204 | 0.154 |
  | IoU precision | 0.091 | **0.200** | 0.159 | 0.154 |
  | centroid med | **0.239 m** | 0.260 | 0.234 | 0.199 |
  | rotation med | **10.8°** | 19.8° | 8.1° | 6.8° |
  | scale err med | 0.471 | **0.240** | 0.591 | 0.804 |
  | cd-F1 @1m | 0.311 | **0.489** | 0.556 | 0.641 |
  | class-free recall@1m | 0.404 | **0.579** | 0.923 | 0.897 |
  | label acc | 0.33* | 0.75 | 0.27* | 0.33* |

  *v1 uses OPEN-VOCAB labels (prompt-free SAM3D), v2 YOLOE gt-vocab — label_acc not comparable, and
  the coverage/recall gap is partly a detector difference, not just the pipeline. **Read:** on
  *matched* objects v1's centroid (0.20–0.24 m) and rotation (7–11°) are already as good as / better
  than v2 — v2's ICP even hurts rotation on real detector-mask clouds. **v2's real wins are (1) SCALE**
  (scale-fit 0.47/0.59→0.24, the C3 contribution), **(2) consolidation/precision** (v1 duplicate_rate
  0.18 on lounge vs v2 ~0; ObjectTrack collapses per-step dups), **(3) closed-vocab labels.** But v1's
  per-step coverage is strong on dense scenes (lounge class-free recall 0.92 raw). Both are floored on
  IoU@.25/Scan2CAD by scale (recall@0.5 = 0 for both). The Feb-2026 `_out_2112026` v1 runs have 158/92
  objects (per-step, little filtering → heavy dups) + `timing_metrics.json` (~20–100 s/frame — v1 runtime
  line for the paper). **Implication for next steps:** v2's differentiators on real data are scale +
  consolidation, NOT centroid/rotation; and coverage is detector-bound — reinforces the localization
  (box-centroid) + detector-coverage levers over more registration tuning.

  **V2 ON LOUNGE-0 — head-to-head with v1 (2026-07-18).** Ran the full v2 pipeline on lounge-0
  (stride-2 gt YOLOE, 643 det → **38 mature tracks / 39 GT = tracks/gt 0.97**, SAM3D 38/38 drained).
  Ablation reusing the cached meshes (`results/phase0_lounge0_rs_{layout,icp,scaleicp,scaleicp_gate}/`):

  | metric | v1 raw (69) | v1 dedup (39) | v2 layout (38) | v2 scale+ICP (38) | v2 +gate (16) |
  |---|---|---|---|---|---|
  | IoU@.25 F1 | **0.204** | 0.154 | 0.130 | 0.156 | 0.182 |
  | IoU precision | 0.159 | 0.154 | 0.132 | 0.158 | **0.312** |
  | IoU recall | **0.282** | 0.154 | 0.128 | 0.154 | 0.128 |
  | centroid med | 0.234 | **0.199** | 0.283 | 0.244 | 0.279 |
  | rotation med | 8.1° | 6.8° | **4.8°** | 15.3° | 12.4° |
  | scale err | 0.591 | 0.804 | 0.653 | **0.477** | 0.500 |
  | cd-F1 @1m | 0.556 | 0.641 | **0.675** | 0.623 | 0.509 |
  | class-free recall@1m | **0.923** | 0.897 | 0.872 | 0.795 | 0.564 |
  | label acc | 0.27* | 0.33* | **1.00** | 1.00 | 1.00 |
  | duplicate rate | 0.179 | 0 | 0 | 0.026 | 0 |

  *open-vocab (v1) vs YOLOE gt-vocab (v2) — not comparable, and confounds coverage. **Honest read
  (lounge is v1's strongest scene):** v1 raw actually WINS on IoU-F1 (0.204) — on a dense room the loose
  IoU@.25 metric rewards its flood of 69 per-step detections (recall 0.28) despite dups (0.18). v2's
  consolidation (69→38) TRADES that recall for a clean map: **label acc 1.00 vs 0.27, scale 0.48 vs 0.59,
  cd-F1 0.675 vs 0.556, rotation 4.8° (layout) vs 8.1°, dup 0, and precision 0.31 with the gate.** So the
  v1-vs-v2 verdict is **metric-dependent**: v1 on raw recall-driven IoU-F1 + coverage; v2 on precision,
  labels, scale, non-duplication — a *usable* asset map vs a flood of boxes. **[CORRECTED 2026-07-19]
  ICP-DROP ABLATION — inconclusive, do NOT drop ICP.** Full reg ablation on both scenes (see hallway
  block): matched sets are tiny (TP 3–6), so rotation medians are unreliable; on IoU-F1 ICP helps/ties
  everywhere (lounge icp/scale_icp 0.156 > layout/scale 0.130). The only robust registration effect is
  scale-fit lowering scale_err. So the recommended real recipe stays **scale_icp (scale-fit + ICP)**,
  optionally + precision gate; layout-only is not better. The gate is more aggressive on lounge (38→16);
  its 6-obs floor is scene-density-dependent → may want adaptive/per-scene tuning. Both scenes:
  recall@0.5=0, Scan2CAD=0 → scale still floors the strict metrics for v1 AND v2.

  **REGISTRATION ABLATION COMPLETED w/ LAYOUT BASELINE (2026-07-20) — corrected the real-robot story.**
  Added `layout` (SAM3D-native) + `icp` runs for smalloffice-0/1 (reused drained queues) so all 4 scenes
  have layout/icp/scale_icp/+gate. 4-scene means: IoU-F1 layout 0.172 / icp 0.190 / scale_icp 0.137 /
  +gate 0.174; scale_err 0.495 / 0.490 / **0.346** / 0.351; centroid 0.219/0.224/**0.190**/0.199.
  **CORRECTED + SAVED finding: scale-fit's real win is SCALE, proven by a PAIRED per-object test**
  (`scripts/paired_scale_test.py`, 61 objects): layout scale-err median 0.739 (mean 1.14 — some objs
  2–4× off) → scale-fit 0.592, **38/61 objs better, Wilcoxon p=9.5e-5**. NOT centroid (paired unchanged,
  p=0.33 — aggregate 0.19 was an IoU-matched-set selection artifact) and NOT IoU-F1 (noise-dominated on
  TP 1–6/scene; scale-fit hurts it, gate recovers via precision; floored by ~0.5 m centroid scatter from
  the uncalibrated extrinsic). ICP hurts rotation (small-set artifact). v2-vs-Clio IoU gap is
  pipeline-level (even layout 0.172 ≫ Clio 0.039). Lead the real C3 claim on the PAIRED SCALE test, not
  IoU-F1. Full table in WORKSHOP_PAPER_PLAN Table 2. Runs: `results/phase0_*_rs_{layout,icp}`.

  **FULL 4-SCENE v2 REAL-ROBOT TABLE (2026-07-19, scale_icp reproj ± precision gate).** smalloffice-0/1
  added (v2-only — no v1 outputs in /data/sam3d). Detect stride-2 gt; SAM3D drained (SO0 10/10, SO1 8/8).

  | scene (GT) | scale_icp F1 | +gate F1 | +gate prec | centroid | scale_err | cd-F1@1m | class-free@1m |
  |---|---|---|---|---|---|---|---|
  | hallway-1 (57) | 0.089 | 0.104 | 0.20 | 0.260 | 0.240 | 0.47 | 0.58 |
  | lounge-0 (39) | 0.156 | 0.182 | 0.31 | 0.244 | 0.477 | 0.62 | 0.80 |
  | smalloffice-0 (11) | 0.190 | 0.267 | 0.50 | 0.206 | 0.262 | 0.67 | 0.91 |
  | smalloffice-1 (10) | 0.111 | 0.143 | 0.25 | 0.051 | 0.404 | 0.78 | 0.70 |

  **The precision gate improves IoU-F1 AND precision on ALL 4 scenes** (0.089→0.104, 0.156→0.182,
  0.190→0.267, 0.111→0.143) — the most consistent v2 finding on real data (unlike the noisy registration
  medians). smalloffice-1 reaches **5 cm centroid / 5.6° rot** (small room, dense coverage → near sim
  quality); smalloffice tracks/gt 0.80–0.91. **Caveat: TP=1–6 per scene → per-scene IoU-F1/rotation are
  noisy; trust the cross-scene consistency (gate) and the pooled aggregate, not single-scene medians.**
  Runs: `results/phase0_smalloffice{0,1}_rs_scaleicp{,_gate}/`, `results/phase0_hallway1_rs_scaleicp_gate_gentle/`.
  Re-cache stride-2 gt detections first (`detect.run --source realsense --scene hallway-1
  --prompt gt --stride 2`); **run eval at `--stride 2` to match the detection cache** (default
  stride 20 would visit only 1/10 of the detected frames and gut multi-view coverage).
  Diagnostic scripts persisted in `r2s3d_core/scripts/`:
  `rs_detect_probe.py` (prompt-mode recall probe + GT overlays), `rs_tracker_diag.py`
  (naive-vs-full tracker recall), `rs_loc_diag.py` (centroid localization vs GT). **SAM3D
  worker:** `conda activate sam3d-objects && python humble_ws/src_Real2USD/real2sam3d/
  scripts_sam3d_worker/run_sam3d_worker.py --queue-dir <out>/sam3d_queue --sam3d-repo
  humble_ws/src_Real2USD/real2sam3d/sam-3d-objects --use-depth` (add `--once` to drain then exit); then re-run
  `eval.run` to collect. See `real2sam3d/config/USER_NEXT_STEPS.md`.
- **`data/rosbag.py` `RosbagSource` (source `rosbag`/`lidar`) — SECONDARY** (LiDAR-vs-RS
  depth ablation): front-cam RGB + projected sparse lidar depth. Details below.

The lidar path (below) flows the same way:
- **`data/rosbag.py` `RosbagSource`** (source `rosbag`/`go2`) — pure-Python bag read via
  `rosbags` (new `rosbag` uv extra; NO ROS install; only standard sensor/nav/tf topics
  deserialized, never the custom `go2_interfaces` msgs). The Go2 has **no depth camera**, so
  it accumulates `/point_cloud2` (already odom-frame) into one cloud and **projects it into
  each `/camera/image_raw` pose to synthesize metric depth** (`lidar_depth.py`, ROS-free port
  of v1 `ProjectionUtils.lidar2depth`; meters + nearest-wins z-buffer). Poses via
  `frames.T_odom_cam_go2` (extrinsic verified **==** v1 to 1e-16). RGB is **undistorted**
  (plumb_bob k1≈−0.34) into the pinhole-K frame so RGB↔projected-depth align. Cloud cached
  to `$R2S3D_DATA/rosbag_cache`.
- **`data/supervisely.py` `load_supervisely_gt`** — the v1 hand-labeled 3D cuboids
  (`evaluations/supervisely/<scene>.pcd.json`, copied into the repo) → `GTObject` (radians
  xyz-euler, position=center, dimensions=full extents, label aliases → chair/table/door).
  GT is **absolute odom Z-up** = the same frame the cloud+poses use, so preds align directly.
- **4 scenes** wired: smalloffice-0/1, hallway-1, lounge-0 (bags under `/data/go2/lidar/`).
  Validated on lounge-0: cloud 331k pts, GT XY ⊂ cloud XY ⊂ camera-traj, depth median 5.5 m,
  RGB↔depth overlay eyeballed (`r2s3d_core/results/rosbag_debug/`). **89 tests pass** (+9 new:
  `test_lidar_depth.py`, `test_supervisely.py`). Metrics available: centroid / 3D-IoU / rot /
  scale / label F1 (box GT, no mesh Chamfer). NEXT: run detector→track→SAM3D→register on the
  4 scenes and score vs the cuboid GT (the real-robot column).

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
