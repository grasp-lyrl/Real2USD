# STATUS — where we are in the v2 rework

**This is the living progress tracker. It is the first doc to read to know the current
state, and the last doc to update at the end of every work session / milestone.** Keep
it honest: "done" means *verified* (tests pass / numbers produced), not "code written".

- Strategy & rationale: `REWORK_PLAN.md` · Interfaces & resolved decisions: `PHASE_SPECS.md`
- **Things only the human can do: `ACTION_ITEMS.md`** (Claude adds to it on every gated dependency)
- Datasets & access: `DATASETS.md`

_Last updated: 2026-07-07 (end of Phase 0)._

## Phase dashboard

| Phase | Title | State | Notes |
|------|-------|-------|-------|
| 0 | Dataset + harness + naive baseline | 🟢 **done** | harness verified on 8 Replica scenes; SAM3D worker validated + real `sam3d_layout` row on room0 (variant B / all-scene aggregate pending) |
| 1 | frames.py + validation + loud fallbacks | ⬜ not started | fully unblocked — recommended next |
| 2 | ObjectTrack node | ⬜ not started | SAM 3 detector access ([AI-6](ACTION_ITEMS.md)) helps but YOLOE fallback exists |
| 3 | Localization stack (TEASER++ / ICP / refine) | ⬜ not started | go/no-go gate: must beat sam3d_layout |
| 4 | Reconciliation + export | ⬜ not started | needs Isaac Sim |
| 5 | Benchmark campaign | ⬜ not started | dataset access is the long pole — [AI-2..5](ACTION_ITEMS.md) started early |
| 6 | Paper rewrite | ⬜ not started | |

Legend: ⬜ not started · 🟡 in progress / partially blocked · 🟢 done · 🔴 blocked

## Current focus

Phase 0 done: SAM3D worker validated end-to-end on the 5090 and the real `sam3d_layout`
row produced on room0 (F1@.25 0.58, **Scan2CAD acc 0.0**, centroid 10 cm, rotation ~28°,
scale ~54%) — the motivating layout-error result. A metric bug was found and fixed:
rotation/scale are now axis-labeling-invariant (min-volume OBB axes are unordered;
naive R-vs-R comparison had inflated rotation 110°→28°, scale 107%→54%). Placement
composition validated independently (posed mesh sits 3.8 cm from its own depth cloud).

Remaining Phase-0 niceties (optional): `sam3d_layout_icp` (variant B, needs open3d) and
the all-8-scene aggregate. Next major work: **Phase 1** (`frames.py`).

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
