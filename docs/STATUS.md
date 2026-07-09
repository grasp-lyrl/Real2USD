# STATUS — where we are in the v2 rework

**This is the living progress tracker. It is the first doc to read to know the current
state, and the last doc to update at the end of every work session / milestone.** Keep
it honest: "done" means *verified* (tests pass / numbers produced), not "code written".

- Strategy & rationale: `REWORK_PLAN.md` · Interfaces & resolved decisions: `PHASE_SPECS.md`
- **Things only the human can do: `ACTION_ITEMS.md`** (Claude adds to it on every gated dependency)
- Datasets & access: `DATASETS.md`

_Last updated: 2026-07-08 (Phase 2 ObjectTrack — detector-in-sim; fragmentation cleanup verified on room0+room1)._

## Phase dashboard

| Phase | Title | State | Notes |
|------|-------|-------|-------|
| 0 | Dataset + harness + naive baseline | 🟢 **done** | harness verified on 8 Replica scenes; SAM3D worker validated; `sam3d_layout` row on room0. **Full-frame >> crop input (F1 .58→.77, Scan2CAD 0→.16) — now the default.** all-scene aggregate + full-frame ICP pending |
| 1 | frames.py + validation + loud fallbacks | 🟢 **done** | `frames/` matches v1 `ply_frame_utils` to 1e-9; Phase 0 numbers reproduce exactly; ROS-node dedup deferred to the Phase 2 wrapper |
| 2 | ObjectTrack node | 🟡 **in progress** | ROS-free `tracks/` + `detect/` (YOLOE) built; tracker tests pass; **detector-in-sim fragmentation cleanup verified room0+room1** (SAM3D calls ↓~4×, tracks/GT 1.3→0.33). SAM3D-mesh placement-degradation (vs GT-mask ceiling) collecting; ROS wrapper deferred |
| 3 | Localization stack (TEASER++ / ICP / refine) | ⬜ not started | go/no-go gate: must beat sam3d_layout |
| 4 | Reconciliation + export | ⬜ not started | needs Isaac Sim |
| 5 | Benchmark campaign | ⬜ not started | dataset access is the long pole — [AI-2..5](ACTION_ITEMS.md) started early |
| 6 | Paper rewrite | ⬜ not started | |

Legend: ⬜ not started · 🟡 in progress / partially blocked · 🟢 done · 🔴 blocked

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
