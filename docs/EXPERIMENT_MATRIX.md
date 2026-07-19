# EXPERIMENT MATRIX — the authoritative index of paper runs

_Created 2026-07-19. Every figure/table in the paper regenerates from the `run.json` files listed
here (CLAUDE.md: never hand-edit tables). This doc is the single source of truth for **what to run,
where it lands, and its status** — so data is findable, reruns are one command, and we do NOT run
anything not listed. Aggregate with `scripts/agg_paper.py` (writes `results/paper/_tables/*.csv`)._

## Naming convention (STRICT — enables clean aggregation)

- **Sim runs:** `results/paper/sim/<config>_s<id>/run.json`
  where `<config>` ∈ {`cluster_gt`, `asset_icp_gt`, `asset_layout_gt`, `asset_scaleicp_gt`,
  `cluster_generic`, `asset_icp_generic`}, `<id>` ∈ the 10 val ids.
- **Detections cache:** `results/detections/procthor/<prompt>/<id>/` (shared across runs of an id).
- **Real-robot runs:** existing `results/phase0_<scene>_rs_scaleicp_gate*/` (v2) — indexed below.
- **Clio baseline:** scored on the fly by `scripts/score_clio_baseline.py` (writes
  `results/paper/real/clio_metrics.json`).
- Split is **always `val`**; stride **1** (matches s200). SAM3D queue per config at `<out>/sam3d_queue`.

## Scope lock — WHICH versions go in the paper (do NOT run others)

**IN:**
- **Sim asset** = `object_track_icp`, **gt-vocab** (headline detector-driven asset). val-10.
- **Sim cluster** = `object_track_cluster` (denoised), **gt-vocab** (ablation baseline). val-10.
- **Sim registration ablation** (layout / icp / scale_icp) — **s200 only** (reference; icp is the sim
  headline). Free (shares the s200 asset SAM3D queue).
- **Sim open-vocab** (generic asset + cluster) — **s200 only** (robustness note).
- **Sim GT-mask ceiling** (`sam3d_layout_scale_icp`) — **s200 val only** (perception-gap reference).
- **Real v2** = `object_track` scale_icp+gate, gt-vocab, 4 Go2 scenes (have).
- **Real internal ablation** ±scale-fit, ±gate — 4 scenes (have).
- **Real Clio** baseline — 4 scenes (have).

**OUT (do not spend compute):** train-split runs; `pf` prompt (junk labels, over-seg); `reproj_mv`
(rejected); TEASER (shelved); v1 as a numeric row (provenance only); generic/pf on all 10 (s200 only);
GT-mask ceiling on all 10 (s200 only).

## Sim val-10 matrix (ids: 137 200 428 434 534 569 573 683 771 912)

| config | method | prompt | SAM3D? | out-dir | status |
|---|---|---|---|---|---|
| cluster_gt | object_track_cluster | gt | no (FREE) | paper/sim/cluster_gt_s<id> | **✓ DONE 10/10 (AI-9 fixed)** |
| asset_icp_gt | object_track_icp | gt | YES | paper/sim/asset_icp_gt_s200 | **s200 ONLY (decision)** |

Prereq: gt-vocab **detections** for all 10 (only s200 exists → 9 to run; detector pass, no SAM3D).

**DECISION 2026-07-19 (Chris): asset = s200 only; cluster = val-10.** Rationale via the claim split:
(A) *instance-first* is shown by perception recall, not this ablation; (B) *generation does NOT
localize* (depth+registration does) is carried at n=10 by the **free cluster** runs ("cluster localizes
fine across 10 scenes"); (C) *generation adds a well-posed box + sim-ready mesh* is corroborated by the
s200 ablation + the 4-scene real-robot scale-fit win + the qualitative sim-ready-mesh point. So val-10
asset (a multi-hour SAM3D campaign) is NOT needed — s200 asset suffices. Revisit only for the archival
resubmission. s200 asset reuses the drained queue `results/procthor_procthor_object_track_icp_s200_val/
sam3d_queue` (no regeneration).

## Real-robot matrix (4 Go2 scenes: hallway-1, lounge-0, smalloffice-0, smalloffice-1) — DONE

| config | dir pattern | status |
|---|---|---|
| v2 scale_icp | `results/phase0_<scene>_rs_scaleicp*` | ✓ |
| v2 scale_icp + gate | `results/phase0_<scene>_rs_scaleicp_gate*` | ✓ |
| Clio | `/data/Clio/*.graphml` → `scripts/score_clio_baseline.py` | ✓ |

## Reruns
Every run is idempotent by out-dir. Batch: `scripts/run_paper_sim_val10.sh` (skips existing).
SAM3D generation needs the worker (conda `sam3d-objects`) draining `<asset out>/sam3d_queue`.
