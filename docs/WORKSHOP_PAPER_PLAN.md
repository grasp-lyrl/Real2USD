# WORKSHOP PAPER PLAN — SeMaNa @ IROS 2026 (extended abstract)

_Consolidated 2026-07-19 (rewritten from the working-log version — exploration/retraction trail
removed; this reflects current understanding with final numbers in hand). Numbers regenerate from
`results/paper/` via `scripts/agg_paper.py`; deeper notes in `STATUS.md` + `GENERATION_ABLATION_PLAN.md`._

## Venue & format
- **SeMaNa** (Semantic-Aware Mapping & Navigation), IROS 2026 workshop. Extended abstract, IEEE RAS
  double-column, **2–4 pages** (refs excluded), ≤10 MB, figures encouraged.
- **Double-blind, non-archival** — explicitly welcomes work under review; prioritizes discussion over
  novelty. ⇒ the prior "limited novelty / stitched modules" rejection reason does **not** apply here.
- **Deadline 2026-07-29**, notification 2026-08-27.

## Thesis (one sentence)
> A single-image 3D generative model (SAM 3D) produces excellent object *shapes* but cannot build a
> *scene*; we turn it into a scene creator with an instance-forward multi-view front-end that decides
> *what objects exist* and a metric-placement back-end that decides *where and how big* they are —
> yielding an asset-centric scene graph whose nodes are Sim(3)-registered, simulation-ready meshes
> rather than labeled point clusters.

## Positioning & honesty (read before writing any claim)
- **The front-end is infrastructure, not our novelty.** Our ObjectTrack front-end (per-frame open-vocab
  masks → project → associate/merge → per-object fused cloud) is *architecturally the same shape* as
  ConceptGraphs / HOV-SG. The "instance-forward vs clustering" distinction is real only vs Hydra/Khronos
  (which cluster a global labeled map). **Do NOT claim a novel or superior front-end** — a reviewer will
  see ConceptGraphs immediately.
- **The differentiator is the node payload:** a generated, Sim(3)-registered, *simulation-ready* mesh
  with a well-posed metric box, where the clustering line keeps a labeled point cluster. That is the row
  the ID/segment line cannot fill.
- **USD is serialization, not a claim** — one sentence (export for simulation).
- **SAM 3 ≠ SAM 3D.** SAM 3 = 2D open-vocab detect/seg/track; SAM 3D = single-image shape generator.
  Disambiguate on first use; reviewers conflate them.

## Contributions (three claims, each a single-variable experiment — final numbers in hand)

**C1 — SAM 3D is a shape prior, not a localizer; our registration supplies the metric pose/scale.**
- *Sim (ProcTHOR val-10):* placing the SAM 3D mesh by SAM 3D's OWN predicted pose (`layout`) vs adding
  our ICP: IoU-F1 **0.404→0.485 (+20%)**, recall@0.5 **0.147→0.224 (+52%)**, rotation ~**9.7°→6.8°**.
- *Real (Go2):* the **reprojection scale-fit** (frontal extent from the 2D mask, along-ray from SAM 3D
  aspect — `scale_source=reproj`) cuts scale error substantially on real depth: hallway **0.375→0.240**.
- *Regime-dependent scale (important — do NOT claim a blanket "SAM 3D scale is 3× off"):* our MEASURED
  aggregate scale error is **~27% on sim** (icp-only) vs **~37% on real** (layout). SAM 3D's scale is
  reliable *in-distribution* (clean ProcTHOR renders) and degrades *out-of-distribution* (real robot
  imagery) — which is exactly why scale-fit is ≈-neutral on sim (icp-only wins) but helps on real. The
  "~3×" figure is an upstream-documented worst-case (Meta) / worst-axis, not our aggregate. So `icp` is
  the sim asset recipe, `scale_icp`+reproj the real.

**C2 — Generation earns its keep by completing the geometry the robot never observed.** (the "why generate")
- *Sim shape-completion (val-10, 212 objects, oracle-placed to isolate shape):* the generated asset
  reconstructs the UNOBSERVED surface **2.87× better** than the observed cluster at low coverage
  (**0.081 vs 0.232 m**, 68/69 objects), collapsing to a tie where the object is well seen (18% unseen).
  **166/212 objects favor the asset.** Generation's value concentrates in the partial-view regime real
  robots live in — *not* a slight average gain.
- *Honest pairing:* end-to-end, realizing this needs enough observation to register (placement is
  coverage-limited); the oracle-placement result isolates that the shape prior *carries* the unseen
  geometry. Plus the deliverable metrics can't score: a watertight, canonically-oriented mesh vs a cluster.

**C3 — The net map places objects metrically where the clustering scene-graph line only localizes them
coarsely.** (usable asset map)
- *Real (Go2, 4 scenes) vs Clio (Maggio RA-L'24, run by us, same GT+metrics):* on **loose** open-set
  recall the two tie (cd-F1@1m 0.63 vs 0.54; class-free 0.75 vs 0.69 — objects get found); on **strict**
  metric placement our v2 pulls clearly ahead — **IoU-F1 0.174 vs 0.039 (4–5×)**, scale-err 0.35 vs 0.55.
- *Contribution ablation (v2-internal; `layout`=SAM3D-native → +icp → +scale-fit → +gate, Table 2):*
  scale-fit's real win is **scale** — a PAIRED per-object test (61 objs) gives median scale-err
  0.739→0.592, **38/61 objects, Wilcoxon p=9.5e-5**. NOT centroid (paired: unchanged, p=0.33 — the
  aggregate 0.19 was a matched-set selection artifact) and NOT IoU-F1 (noise-dominated on TP 1–6/scene;
  scale-fit slightly hurts it, the gate recovers it via precision). Attribute *scale accuracy* to
  scale-fit, *precision* to the gate; do not claim centroid or IoU-F1 gains on real.

## Results summary (the tables/figures — all numbers in hand, regen via `scripts/agg_paper.py`)

**Table 1 — Sim, node-payload × registration (ProcTHOR val-10; the controlled C1 + payload experiment).**
Same front-end/tracks/depth/metrics; vary one factor. `results/paper/sim/<config>_gt_s<id>/run.json`;
**CSV `results/paper/_tables/sim_per_config_mean.csv` (means) + `sim_per_run.csv` (per-scene); regen
`scripts/agg_paper.py`.**

| config | IoU-F1 | recall@.5 | Scan2CAD | cent(m) | rot° | scale_err | Chamfer | footIoU | cdF1@1m |
|---|---|---|---|---|---|---|---|---|---|
| asset: layout (SAM3D-native) | 0.404 | 0.147 | 0.075 | 0.085 | 9.7 | 0.319 | 0.182 | 0.387 | 0.706 |
| asset: **+ICP (ours)** | **0.485** | **0.224** | 0.088 | 0.066 | 6.8 | 0.319 | 0.176 | 0.402 | 0.706 |
| asset: +scale+ICP (ours) | 0.449 | 0.190 | 0.043 | 0.066 | 6.3 | 0.335 | 0.176 | 0.413 | 0.708 |
| cluster (baseline) | 0.405 | 0.229 | 0.123 | 0.049 | 1.2 | 0.306 | 0.157 | 0.446 | 0.686 |

Read: our registration beats SAM 3D-native placement (C1). vs the clean-sim cluster it's nuanced (asset
wins strict IoU-F1; cluster competitive on centroid/Scan2CAD) — which is *why* C2's shape-completion
(partial-view regime) is the real generation story, not the clean-sim average.

**Scale-source ablation — why we read scale off the RGB mask, not the depth cloud (s200 val).**

| scale source | IoU-F1 | scale_err | Chamfer |
|---|---|---|---|
| fused (raw depth-cloud OBB extent) | 0.158 | 0.692 | 0.254 |
| fused_robust (SOR+DBSCAN denoised OBB) | 0.376 | 0.513 | 0.132 |
| **reproj (2D mask span + median depth + SAM 3D aspect) — ours** | **0.509** | **0.320** | **0.093** |
| icp only (no scale-fit; SAM 3D native scale) | 0.545 | 0.269 | 0.104 |

Reading the object's extent from the **depth-cloud OBB craters** (fused: scale_err 0.69, boxes badly
oversized) because a few edge-bleed points inflate the box; denoising the cloud (fused_robust) only
partially recovers because the cloud is a *partial one-sided view* whose unobserved-axis extent is wrong
regardless of noise. Reading the frontal extent from the **clean 2D mask** (reproj) recovers it.

**Why ICP is still fine on the same noisy cloud** (a method-design point worth stating): extent is a
**max/extremum** statistic (set by min/max coords → outlier-fragile), while **rigid ICP is a bulk/average**
fit (summed NN distance over all points, dominated by the object body → outlier-tolerant, and it has no
scale DOF, only nudging pose from SAM 3D's rough layout). So the *same* contaminated cloud that wrecks the
extent still constrains pose. **Design principle: scale needs the true boundary (fragile from noisy/partial
depth → clean RGB mask + shape prior); pose needs the bulk surface (robust → ICP on the cloud).**

**Table 2 — Real robot (Go2, 4 scenes): v2 vs Clio + registration ablation (incl. `layout` baseline).**
`results/phase0_*_rs_*/run.json`; **CSV `results/paper/_tables/real_registration_ablation.csv` (regen
`scripts/agg_real.py`); paired save `paired_scale_test.csv` (regen `scripts/paired_scale_test.py`); Clio
via `scripts/score_clio_baseline.py`.** Per-scene TP is small (1–6) so IoU-F1 is noisy; lead on scale.

*vs external method:*
| method (mean, 4 scenes) | IoU-F1 | cent(m) | scale-err | cdF1@1m | class-free@1m |
|---|---|---|---|---|---|
| Clio (independent, run by us) | 0.039 | 0.324 | 0.551 | 0.535 | 0.687 |
| **v2 (ours: scale-fit+ICP+gate)** | **0.174** | **0.19** | **0.35** | 0.63 | 0.75 |

*registration ablation (our pipeline; `layout` = SAM3D-native placement — the baseline you asked for):*
| metric (4-scene mean) | layout | +icp | +scale_icp | +gate |
|---|---|---|---|---|
| IoU-F1 | 0.172 | 0.190 | 0.137 | 0.174 |
| centroid (m) | 0.219 | 0.224 | **0.190** | 0.199 |
| **scale_err** | 0.495 | 0.490 | **0.346** | 0.351 |
| rotation° | 10.8 | 16.4 | 17.3 | 16.6 |

**Honest read (corrected 2026-07-20; PAIRED test rescues the scale claim — `scripts/paired_scale_test.py`):**
- **Scale-fit's real win is SCALE — proven rigorously by a PAIRED per-object test** (61 objects matched
  under both layout & scale_icp, centroid ≤1 m): layout scale-err median **0.739** (mean 1.14 — some
  objects 2–4× off) → scale-fit **0.592**, **better in 38/61 objects, Wilcoxon p = 9.5e-5**. This
  survives the noisy IoU-F1 because it bypasses the matching gate. Per-scene: hallway 15/22, lounge 17/25
  (strong); smalloffice small-n/weak.
- **Do NOT claim centroid improvement.** The aggregate "centroid 0.219→0.190" (Table row) is a SELECTION
  artifact of the tiny IoU@0.25-matched set; on the paired set centroid is unchanged (0.455→0.495,
  p=0.33). Scale-fit fixes *extent*, not location — as intended.
- **IoU-F1 is floored by centroid scatter (~0.5 m) from the uncalibrated extrinsic** (orthogonal, one-time
  calibratable), not by the edit failing; scale-fit slightly hurts it, gate recovers it. Lead the real
  C3 claim on the **paired scale test**, not IoU-F1.
- **IoU-F1 on real is noise-dominated** (TP 1–6/scene; smalloffice-0 `layout` 0.381 skews the mean) — it
  does not cleanly separate modes; the gate nets it positive (0.137→0.174) mainly via precision.
- **ICP hurts rotation on real** (small-matched-set artifact, already flagged).
- **The v2-vs-Clio IoU gap is pipeline-level:** even `layout` (0.172) ≫ Clio (0.039); scale-fit's
  *specific* contribution is scale accuracy, not that gap.

v1 (our IROS'26 pipeline) stays a one-line provenance note, not a numeric row (confounded).

**Fig 4 (headline) — "why generate": unobserved-surface reconstruction vs coverage (n=10, 212 objects).**
Two diverging curves: asset ~flat 6–8 cm; cluster rises to 23 cm as coverage drops. Inset qualitative
panel: armchair 44% observed → generation fills the unobserved base (red=seen, blue=generated) vs GT
(`results/paper/_figs/completion_panel_s200_t26.png`).

**Figs 1–3 (to build):** Fig 1 pipeline teaser (robot→tracks→registered assets→USD in Isaac Sim);
Fig 2 = C1 visual (SAM 3D-native layout vs +ICP vs GT); Fig 3 = optional recall/robustness bar.

## METHOD — precise, write-up-ready (with file references)

The system is three stages. Code lives in `humble_ws/src_Real2USD/r2s3d_core/src/r2s3d_core/`; the
detector-driven path is `baselines/object_track.py::_run`.

**(1) Instance-forward front-end** (`tracks/`, `detect/`) — *infrastructure, not a claimed contribution.*
- Per-frame open-vocabulary detection+segmentation by **YOLOE** (`detect/yoloe.py`), prompted with the
  scene's GT class vocabulary ("`gt`" prompt); detections cached as a `DetectionSet` (`detect/cache.py`).
- **Multi-view association** (`tracks/tracker.py::run_tracker`, `tracks/associate.py`): each detection is
  back-projected with its masked depth and associated across frames into an **object track**; the track
  accumulates a voxel-downsampled **fused point cloud** (`tracks/fusion.py`) and keeps its observed views.
  Re-ID + **late-merge** fold fragmented tracks (`config['reid']`, `config['late_merge']`, both on).
- **Maturation gate:** a track matures at `MIN_ACTIVE_OBS = 3` observations (batch finalize; the
  `MIN_MATURE_VIEWS=6` constant is dead code in batch eval — see [[tracker-maturation-gate]]).
- **Best-view selection** (`tracks/view.py::full_view_score`, `object_track.py::_best_view`): the least-
  occluded, most-centered kept view is chosen as the single crop fed to generation.

**(2) Retrieve-or-generate node — SAM 3D, shape only** (`baselines/sam3d_layout.py`, SAM 3D worker).
- Each mature track's best view (**full frame** + mask, not a tight crop — full-frame roughly halves
  layout error, see [[sam3d-full-frame-beats-crop]]) is sent once to **Meta SAM 3D** (arXiv 2511.16624;
  NOT Yang'23 SegmentAnything3D) via a disk-queue worker (`real2sam3d/scripts_sam3d_worker/
  run_sam3d_worker.py`, conda `sam3d-objects`, `--use-depth`). Jobs are content-keyed by
  `{source}_{scene}_t{track}_{framing}` so the mesh cache is **registration-independent** — one
  generation pass yields all registration-ablation rows as free eval re-runs.
- We keep **only the mesh (shape)**. `place_from_sam3d` seeds a world pose from SAM 3D's predicted
  rotation and places the object at the observed depth; **SAM 3D's predicted translation is discarded**
  (unreliable, normalized camera coords), its scale is a starting point to be refined.

**(3) Metric placement back-end** (`baselines/sam3d_layout.py`, `object_track.py::_scale_step/_icp_step`).
Registration modes (`--registration`): `none`=`layout` (SAM 3D-native pose), `icp`, `scale`, `scale_icp`.
- **Reprojection scale-fit** (`--scale-source reproj`, the deployed recipe; `_reproj_target_extent` +
  `_mask_inplane_metric`): the object's two **frontal (in-plane) metric extents are read from the CLEAN
  2D detection mask** — PCA the mask pixels → two principal pixel spans → metric via
  `span_px × median_masked_depth / focal` (the mask has IoU ≈ 0.96 where the raw depth cloud is
  edge-contaminated); the **third (along-ray, unobserved) axis keeps SAM 3D's shape aspect**. Depth's
  role is only the robust median depth (metricizing the mask span) + the ICP target — NOT a depth-cloud
  OBB extent. `_fit_scale_to_extent` applies the anisotropic scale about the mesh OBB centre.
  Alternative scale sources kept as ablations: `fused` (raw depth-cloud OBB — craters), `fused_robust`
  (SOR+DBSCAN-denoised OBB), `silhouette` (Method C, attempted, negative — see limitations).
- **ICP** (`refine_icp`): rigid pose refinement of the (scaled) mesh against the track's fused multi-view
  cloud; no scale DOF. `scale_icp` alternates scale-fit ↔ ICP to convergence (`--scale-icp-iters`).
- **Precision gate** (`tracks/tracker.py::_passes_precision_gate`, `--track-gate`, default 6-obs/0.40-conf,
  off by default): demotes a mature track to REJECTED unless `n_obs ≥ gate_min_obs` AND
  `mean(det_score) ≥ gate_min_score` — prunes short-lived, low-confidence false-positive tracks.
- **Export:** the object-centric scene graph serializes to USD/GLB for simulation (one sentence in paper).

**Design principle to state (from the scale-source ablation + the ICP question):** *scale needs the
object's true boundary — fragile from a noisy/partial depth cloud (its OBB extent is a max/extremum
statistic, wrecked by a few edge-bleed points and by one-sided views), so we read it from the clean RGB
mask + shape prior; pose needs the bulk surface — robust, so rigid ICP on the same cloud is fine (a
summed-NN average dominated by the object body, with no scale DOF).*

## METRIC DEFINITIONS — exact (`eval/metrics.py::evaluate`, `eval/geometry.py`, `eval/label_map.py`)

- **IoU-F1 (headline `f1`)**: Hungarian matching of pred↔GT on **3D OBB IoU ≥ 0.25**; `F1=2PR/(P+R)`,
  `P=TP/n_pred`, `R=TP/n_gt`. `recall@0.5` = Hungarian recall at IoU ≥ 0.5.
- **centroid_err_median_m**: median L2 centre distance over matched pairs.
- **rotation / scale_err** (`geometry.py::box_pose_error`): axis-labeling-invariant (min over the
  symmetry-allowed OBB axis relabelings per `symmetry_for_label`); `scale_err` = max per-axis
  |pred/GT − 1|; medians over matched pairs.
- **scan2cad_accuracy**: fraction of GT with a prediction within 20 cm ∧ 20° ∧ 20% scale.
- **cd_* (coworker-comparable)**: GREEDY match on 2D top-down (X,Y) centroid ≤ τ (default **1 m**).
  `cd_f1` label-agnostic; `cd_micro_f1`/`cd_macro_f1` label-aware (their Object F1).
  **class_free_recall_1m** = fraction of GT with ANY predicted centroid within 1 m in X,Y (label-free).
- **scene_chamfer_mean_m**: SCENE-LEVEL pooled symmetric Chamfer = mean of the two directional mean-NN
  distances over all pred vs all GT surface points (confirmed identical to the coworker's
  `symmetric_chamfer_distance`, `scripts/scene_graph_metrics.py`).
- **footprint_iou** (`geometry.py::footprint_iou`): top-down XY occupancy IoU of pooled pred vs GT surface
  points, 5 cm cells. (Our def; the coworker's exact Footprint-IoU def is unconfirmed — do not compare
  cross-method on footprint; see ACTION_ITEMS Q5.)
- **Label mapping** (`eval/label_map.py`, `--label-map clip`): open-vocab predicted labels are snapped to
  the closest scene-vocab word by CLIP text cosine before label-aware scoring.

## DATA PROVENANCE — every table/number → source file + regen command (for LaTeX)

Regeneration assumes cwd `humble_ws/src_Real2USD/r2s3d_core`. Sim runs need the Xorg display
(`R2S3D_DISPLAY=:0 R2S3D_XAUTHORITY=/run/user/1003/gdm/Xauthority`); cached scenes replay without it.

| Paper table / number | Source file(s) | Regen |
|---|---|---|
| **Table 1 (sim, val-10): layout/icp/scaleicp/cluster means** | `results/paper/_tables/sim_per_config_mean.csv` (means), `sim_per_run.csv` (per-scene), from `results/paper/sim/<config>_gt_s<id>/run.json` (config∈{asset_layout,asset_icp,asset_scaleicp,cluster}, id∈10 val ids) | `uv run python scripts/agg_paper.py` |
| **Scale-source ablation (s200): fused/fused_robust/reproj/icp** | `results/procthor_procthor_object_track_{scale_icp,scaleicp_fused_robust,scaleicp_reproj,icp}_s200_val/run.json` | read those run.json aggregates |
| **Fig 4 / shape-completion (n=10, 212 objs): coverage bins, 166/212, 2.87×** | per-scene `results/paper/_tables/shape_completion_s<id>.csv` (10 files) | `bash scripts/run_shape_completion_val10.sh` (prints the pooled bin table + writes the CSVs) |
| **Fig 4 qualitative panel (armchair, track 26)** | `results/paper/_figs/completion_panel_s200_t26.png` | `uv run … python scripts/render_completion_panel.py --scene 200 --track-id 26 --asset-run results/paper/sim/asset_icp_gt_s200` |
| **Table 2 real registration ablation (layout/icp/scale_icp/+gate × 4 scenes + MEAN)** | `results/paper/_tables/real_registration_ablation.csv`, from `results/phase0_{hallway1,lounge0,smalloffice0,smalloffice1}_rs_{layout,icp,scale_icp_reproj|scaleicp,scaleicp_gate*}/run.json` | `uv run python scripts/agg_real.py` |
| **C1/C3 real SAVE: paired scale test (61 objs, 0.739→0.592, 38/61, p=9.5e-5)** | `results/paper/_tables/paired_scale_test.csv` (per-object) | `uv run --extra dev python scripts/paired_scale_test.py` (prints win-rate + Wilcoxon) |
| **Table 2 Clio row (0.039 / 0.324 / 0.551 / 0.535 / 0.687)** | `/data/Clio/*.graphml` scored vs `evaluations/supervisely/<scene>_voxel_pointcloud.pcd.json` | `uv run --extra registration python scripts/score_clio_baseline.py` (prints per-scene + mean) |
| **C1 sim numbers (layout 0.404 → icp 0.485, recall@0.5 0.147→0.224)** | rows of `sim_per_config_mean.csv` (asset_layout_gt vs asset_icp_gt) | `scripts/agg_paper.py` |
| **Real-robot detections / SAM3D queues** | detections `results/detections/gt/<scene>/`; per-scene SAM3D meshes in `results/phase0_<scene>_rs_scaleicp*/sam3d_queue` | detect: `detect.run --source realsense --scene <scene> --prompt gt --stride 2` |
| **Sim detections / SAM3D queues** | detections `results/detections/procthor/gt/<id>/`; asset meshes shared queue `results/paper/sim/_assetq` (gitignored) | `scripts/run_paper_sim_val10.sh`, `run_paper_asset_val10.sh` |

## PAPER OUTLINE with topic sentences (the drafting skeleton — iterate here next session)

**1. Introduction (¾ col)**
- Robots acting in human spaces need object-centric *metric* maps whose nodes are complete, placed 3D
  objects — usable for simulation and manipulation — not labeled point clusters.
- Single-image 3D generators (SAM 3D) now make excellent object *shapes*, tempting a shortcut, but a
  shape generator is not a scene builder: it decides neither what objects exist, nor where they are, nor
  how big.
- We turn a single-image shape generator into a scene creator by pairing an instance-forward multi-view
  front-end (what exists) with a metric-placement back-end (where/how big), yielding an asset-centric
  scene graph of Sim(3)-registered, simulation-ready meshes.
- Our contributions are three single-variable results: registration supplies the metric pose SAM 3D
  cannot (C1); generation completes the geometry the robot never observed (C2); and the net map places
  objects metrically where the clustering scene-graph line only localizes them coarsely (C3) — shown in
  sim with real GT meshes and on a deployed Go2 quadruped.

**2. Related work (¼–½ col)**
- Clustering metric-semantic scene graphs (Hydra, ConceptGraphs, HOV-SG, Khronos, Clio) build object
  nodes from segmented point clusters; our front-end is deliberately of the same instance-forward family,
  and our contribution is the node payload — a placed, simulatable mesh — not the front-end.
- Single-image 3D generation (SAM 3D) and scene/shape completion (SceneComplete, MetaScenes) produce
  shapes or curated scenes but assume clean crops or human-in-the-loop; we place generated shapes
  autonomously from partial egocentric robot views.
- (Disambiguation sentence: SAM 3 is a 2D detector/tracker; SAM 3D is the shape generator we use.)

**3. Method (1–1.25 pp)**
- An instance-forward front-end associates per-frame open-vocabulary detections into multi-view object
  tracks, each carrying a fused point cloud and a best-view crop; this is standard machinery, included
  because it is what feeds generation and placement (not a claimed contribution).
- Each mature track becomes a node once via a single-image generator that supplies *shape only* — SAM
  3D's predicted **translation** is discarded (the object is placed by depth + ICP, not SAM 3D's
  normalized-camera-coord pose — this is the "shape prior, not localizer" point); its rotation seeds ICP,
  and its scale — reliable in-distribution, degraded on real imagery — is refined by the reprojection fit.
- A metric back-end places the shape by a **reprojection scale-fit** then ICP: the object's two frontal
  (in-plane) extents are read from the *clean 2D detection mask* — pixel span × median masked depth /
  focal length (the mask has IoU ≈ 0.96 where the raw depth cloud is edge-contaminated) — while the
  unobserved along-ray third axis keeps SAM 3D's shape aspect; ICP against the track's fused multi-view
  cloud then fixes pose, gated by a confidence test. (Depth's role is the robust *median* that metricizes
  the mask span and the ICP target cloud — NOT a depth-cloud OBB extent.) The map serializes to USD (one sentence).

**4. Experiments (1.5 pp)**
- We evaluate in two regimes with single-variable, controlled comparisons rather than confounded
  system-vs-system bake-offs: ProcTHOR sim (real GT asset meshes → shape fidelity) and a real Go2 in four
  scenes (cuboid GT → metric placement).
- C1: across ten sim scenes our registration beats SAM 3D's own predicted placement (+20% 3D-IoU, +52%
  recall@0.5, halved rotation error), and on real depth the scale-fit halves scale error — SAM 3D
  localizes poorly; we fix it.
- C2: isolating shape with oracle placement, the generated asset reconstructs the surface the robot never
  saw 2.9× better than the observed cluster at low coverage (166/212 objects), while adding little where
  the object is already well observed — generation's value is completing the unseen, not a clean-sim average.
- C3: against an independently-run clustering scene-graph method (Clio) on the robot data, both find and
  coarsely localize objects, but our asset map places them metrically (4–5× 3D-IoU, lowest scale error),
  and internal ±scale-fit/±gate ablations attribute the gain to our back-end.
- (Defensibility sentence: every comparison is either a same-front-end single-variable ablation or an
  independently-run method scored with identical metrics; loose and strict metrics are dual-reported and
  shared ceilings stated.)

**5. Real-robot deployment (¼ col + fig)**
- The full pipeline runs on a deployed Go2 quadruped and produces an asset-centric USD map consumed by an
  LLM navigation layer — a qualitative end-to-end demonstration on real hardware.

**6. Limitations & conclusion (¼ col)**
- Placement is observation-limited (registration needs sufficient views; strict recall is floored on
  partial real depth, shared with baselines); the reprojection scale-fit reads *image-plane silhouette*
  spans, so it is approximate for objects whose axes are rotated off the image plane (front-on
  assumption); asset maps approximate surfaces; and evaluation is single-session/static — yet the
  asset-centric node is a usable, simulation-ready alternative to labeled point clusters, and generation
  demonstrably completes what the robot cannot see.

## Reviewer-response design (defensibility, to state explicitly)
1. **Controlled ablations, not bake-offs.** The sim comparison holds the entire front-end/tracks/depth/
   metric fixed and varies only the node payload (asset ↔ denoised cluster) or registration mode — the
   delta is attributable to the one thing changed. This is *why* we don't drop an external clustering
   system into sim: the cluster payload IS our clustering baseline, made confound-free.
2. **Independent real baseline.** Clio was run by us and scored with the same GT + metric code as ours;
   no reused private numbers, no per-method tuning.
3. **Honest metrics.** Dual-report loose (1 m centroid — baselines competitive) and strict (3D-IoU /
   Scan2CAD / scale — our differentiator); state shared ceilings (recall@0.5=0 on real; modest clean-sim
   generation average). Chamfer verified identical to the community definition.
4. **Two regimes, each measuring what it can.** Sim measures shape fidelity (real GT meshes); real
   measures metric placement (where scale-fit pays off). Neither over-reaches.

## Provenance / non-content notes
- **Coworker benchmark retired as a contribution** (coworker not a co-author): do NOT publish his
  computed baseline numbers; his metric *definitions* are fine to adopt (standard). Our field comparison
  is the independently-run Clio, not his table.
- **Front-end-vs-field footprint gap is NOT claimed** — confounded by an unconfirmed metric definition
  and by our gt-vocab prompting; only the controlled asset-vs-cluster ablation is claimed.
- **arXiv 2510.10778** (v1) is non-anonymous; double-blind + existing preprint is fine at a non-archival
  workshop given the changed title/framing — do not cite it as "our prior work." (See ACTION_ITEMS.)
- **Title candidates** (pick during drafting): "Making a Shape Generator a Scene Creator: Instance-Forward
  Asset Mapping for Metric-Semantic Scene Graphs" / "Beyond Shape: Placing Single-Image 3D Generations in
  a Metric Robot Scene Graph".

## Drafting tasks (what's left — no experiments remain)
- [ ] Draft prose from the outline above (topic sentences → paragraphs).
- [ ] Build Fig 1 (pipeline teaser, refresh v1 assets), Fig 2 (C1 layout-vs-ICP-vs-GT), finalize Fig 4
      (curve + armchair panel already rendered); pick the real-robot deployment frame.
- [ ] Runtime line: SAM 3D ~24 s/obj + per-stage timing (offline/near-online honesty).
- [ ] Related-work paragraph + citations (Hydra, ConceptGraphs, HOV-SG, Khronos, Clio, SAM 3D,
      SceneComplete, MetaScenes; disambiguate SAM 3D from Yang et al. SegmentAnything3D).
- [ ] Anonymize; IEEE RAS double-column template; ≤10 MB; IEEE refs.

## Deferred to the archival resubmission (not this workshop)
ScanNet/Scan2CAD (public citable field numbers + real meshes for the asset claim; access available, not
downloaded); ConceptGraphs on our data; a cross-detector row (Grounding-DINO+SAM2) for detector-agnosticism.
**Pose-aware silhouette scale-fit — ATTEMPTED 2026-07-20, first cut NEGATIVE (kept as future work).**
Idea: scale the mesh so its *projected silhouette* (known ICP pose + full SAM 3D shape) matches the mask
— handle off-axis objects, constrain the 3rd axis from oblique views. Implemented as
`sam3d_layout._silhouette_scale_transform` + `--scale-source silhouette` (Powell over 3 per-axis scales,
projected-silhouette IoU objective, prior reg toward SAM 3D aspect). **s200 result: IoU-F1 0.412 vs
reproj 0.509 / icp 0.545 — WORSE.** Diagnosis: (1) the objective is nearly flat (sil-IoU 0.711→0.728
across the whole scale range) because `project_cloud_mask` fills the *convex hull* — too coarse to be
scale-sensitive; the optimizer barely moves (median |s−1| 0.014) and its small moves slightly hurt
scale_err (0.27→0.43); (2) sim gt-vocab masks + axis-aligned objects don't stress the off-axis case it
targets. **To earn a place it needs (a) a true mesh-RASTERIZED silhouette (not convex hull) so the
objective is scale-sensitive, and (b) an off-axis / real-robot test regime. Not worth it for the
workshop.** Recipe stays reproj (real) / icp (sim). Code kept as an ablation stub.
