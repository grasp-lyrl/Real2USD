# STATUS — where we are in the v2 rework

**This is the living progress tracker. It is the first doc to read to know the current
state, and the last doc to update at the end of every work session / milestone.** Keep
it honest: "done" means *verified* (tests pass / numbers produced), not "code written".

- Strategy & rationale: `REWORK_PLAN.md` · Interfaces & resolved decisions: `PHASE_SPECS.md`
- **Things only the human can do: `ACTION_ITEMS.md`** (Claude adds to it on every gated dependency)
- Datasets & access: `DATASETS.md`

_Last updated: 2026-07-08 (full-frame ablation — new sam3d_layout default)._

## Phase dashboard

| Phase | Title | State | Notes |
|------|-------|-------|-------|
| 0 | Dataset + harness + naive baseline | 🟢 **done** | harness verified on 8 Replica scenes; SAM3D worker validated; `sam3d_layout` row on room0. **Full-frame >> crop input (F1 .58→.77, Scan2CAD 0→.16) — now the default.** all-scene aggregate + full-frame ICP pending |
| 1 | frames.py + validation + loud fallbacks | 🟢 **done** | `frames/` matches v1 `ply_frame_utils` to 1e-9; Phase 0 numbers reproduce exactly; ROS-node dedup deferred to the Phase 2 wrapper |
| 2 | ObjectTrack node | ⬜ not started | SAM 3 detector access ([AI-6](ACTION_ITEMS.md)) helps but YOLOE fallback exists |
| 3 | Localization stack (TEASER++ / ICP / refine) | ⬜ not started | go/no-go gate: must beat sam3d_layout |
| 4 | Reconciliation + export | ⬜ not started | needs Isaac Sim |
| 5 | Benchmark campaign | ⬜ not started | dataset access is the long pole — [AI-2..5](ACTION_ITEMS.md) started early |
| 6 | Paper rewrite | ⬜ not started | |

Legend: ⬜ not started · 🟡 in progress / partially blocked · 🟢 done · 🔴 blocked

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
(9.5°). Caveat: this uses GT masks + GT poses → an *optimistic upper bound* on Phase-2 fusion
(real fusion is noisier). So: full-frame is the big lever (scale), accumulation a small
pose/coverage polish, residual scale ~0.30 still needs Phase-3 Sim(3). Clouds for inspection:
`results/ablation_full_vs_crop/icp_{target_singleview,target_accumulated[,_lite],source_posed,source_afterICP}.ply`
(script `scripts/export_icp_clouds.py`). Phase 1 done: `r2s3d_core/frames/`
is now the single source of frame transforms (SAM3D shape chain + Go2 body chain),
verified equal to v1 `ply_frame_utils` at 1e-9, with `config/go2_calibration.yaml`,
boundary validation (`frames/validate.py`), and round-trip/regression tests; Phase 0
numbers reproduce exactly. Deferred: repointing the 3 ROS nodes + `ply_frame_utils` at
`frames/` (do it when the Phase 2 ROS wrapper is built — touches the ROS package).

Next major work: (a) confirm the full-frame win on the **all-8-scene aggregate** (room0 only
so far); (b) **Phase 2** (ObjectTrack node). Full-frame + full-frame ICP already done on room0.

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
