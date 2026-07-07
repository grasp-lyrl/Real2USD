# Datasets for the v2 benchmark campaign

This is the *how-to* for each dataset. The **actionable human checklist** (what to do,
what it blocks, current status) lives in `ACTION_ITEMS.md` — that is the single source of
truth; the sections here are the instructions those items link to.

Data root on this desktop is **`~/Data/datasets/`** (no writable `/data`; override with
`$R2S3D_DATA`). Target layout:

```
~/Data/datasets/
  replica/        # NICE-SLAM posed RGB-D renders + replica_semantic/ GT (Phase 0) — DONE
  scannet/        # ScanNet v2 scans (.sens extracted)
  scan2cad/       # Scan2CAD alignments + ShapeNetCore CAD models
  clio/           # Clio's office/apartment/cubicle/building rosbags
```

## 1. Replica (Phase 0 — no agreement needed, start here)

The raw Replica release is meshes only. Use the community-standard posed RGB-D
trajectories rendered by the iMAP/NICE-SLAM authors (~2000 frames/scene, GT poses +
intrinsics) — the same data ConceptGraphs/HOV-SG evaluate on.

```bash
bash scripts/datasets/download_replica.sh /data/datasets/replica
```

(~12 GB zip.) GT semantics/instance meshes for metrics come from the original Replica
repo (https://github.com/facebookresearch/Replica-Dataset) — the `habitat/mesh_semantic.ply`
(per-face `object_id`) + `info_semantic.json` files. Fetch them with:

```bash
bash scripts/datasets/download_replica_semantic.sh ~/Data/datasets/replica
```

(~34 GB compressed / ~100 GB extracted to `replica_semantic/<scene>/habitat/`; the
`.part??` files can be deleted after extraction.) The `r2s3d_core` Replica backend reads
GT from there automatically and caches per-scene npz. **Status: done** (see
`ACTION_ITEMS.md`).

## 2. ScanNet v2 (placement table — **gated, human step required**)

1. **[HUMAN]** Fill the ScanNet Terms of Use (PDF linked from
   https://github.com/ScanNet/ScanNet — "ScanNet Data") and email it as instructed
   (scannet@googlegroups.com). They reply with `download-scannet.py`.
2. Download the Scan2CAD-relevant scans (start with the ~100 validation scenes rather
   than all 1513; full RGB-D is multi-TB):
   `python download-scannet.py -o /data/datasets/scannet --id <sceneid>` per scene, types
   `.sens`, `_vh_clean_2.ply`, `.aggregation.json`, `_vh_clean_2.0.010000.segs.json`.
3. Extract posed RGB-D from `.sens` with ScanNet's SensReader
   (https://github.com/ScanNet/ScanNet/tree/master/SensReader/python) — color, depth,
   intrinsics, per-frame camera poses (BundleFusion).

## 3. Scan2CAD (**gated, human step required**)

1. **[HUMAN]** Request access via the form at https://scan2cad.org (requires having
   accepted the ScanNet ToS). Yields `full_annotations.json` (97,607 CAD alignments,
   1506 scans).
2. **[HUMAN]** CAD models: ShapeNetCore.v2 now lives on HuggingFace
   (https://huggingface.co/datasets/ShapeNet/ShapeNetCore) — accept the license with
   your HF account, then `huggingface-cli download`.
3. Alternative/adjunct with sim-ready GT: **MetaScenes** (https://meta-scenes.github.io,
   CVPR 2025) — 706 ScanNet scenes, 15,366 human-verified asset replacements; strong GT
   for the placement table and Scan2Sim is a retrieval baseline. **[HUMAN]** access is
   gated via a Google Form on the project page — submit alongside the ScanNet request.

## 4. Clio datasets (public)

Office / Apartment / Cubicle / Building RealSense rosbags + GT object boxes; links are
in the Clio repo README (https://github.com/MIT-SPARK/Clio). Chris's Clio-Eval fork
(https://github.com/christopher-hsu/Clio-Eval) already consumes them. Note bags are
ROS1 → convert or read with `rosbags` python lib (no ROS needed) for the SequenceSource
backend.

## Sanity order

Replica today (unblocks Phase 0 fully) → submit ScanNet + Scan2CAD + HF ShapeNet
requests the same day (days-long latency, longest pole) → Clio bags whenever (public,
only needed by Phase 5).
