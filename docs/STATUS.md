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
| 0 | Dataset + harness + naive baseline | 🟡 **substantially done** | harness verified on all 8 Replica scenes; SAM3D baseline numbers blocked on [AI-1](ACTION_ITEMS.md) |
| 1 | frames.py + validation + loud fallbacks | ⬜ not started | fully unblocked — recommended next |
| 2 | ObjectTrack node | ⬜ not started | SAM 3 detector access ([AI-6](ACTION_ITEMS.md)) helps but YOLOE fallback exists |
| 3 | Localization stack (TEASER++ / ICP / refine) | ⬜ not started | go/no-go gate: must beat sam3d_layout |
| 4 | Reconciliation + export | ⬜ not started | needs Isaac Sim |
| 5 | Benchmark campaign | ⬜ not started | dataset access is the long pole — [AI-2..5](ACTION_ITEMS.md) started early |
| 6 | Paper rewrite | ⬜ not started | |

Legend: ⬜ not started · 🟡 in progress / partially blocked · 🟢 done · 🔴 blocked

## Current focus

Phase 0 is functionally complete and committed. Next unblocked work is **Phase 1**
(`frames.py`). The only Phase-0 gap is the `sam3d_layout` / `sam3d_layout_icp` numbers,
which need the SAM 3D worker running on this desktop ([AI-1](ACTION_ITEMS.md)).

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
