# ACTION ITEMS — things only you (the human) can do

Claude cannot request dataset access, accept licenses, obtain gated checkpoints, log
into services, or make policy/IP calls. **Whenever a build step hits one of those, Claude
adds an item here (explicit: what / why it blocks / how / where) and flags it in chat.**
You work these; mark them done. `STATUS.md` links here for anything blocked.

Convention: each item has an ID (AI-N, stable), a **blocks** field, and a status box.
When done, check the box and add the date; leave it in place for provenance.

Tip: for interactive logins/commands, you can run them in this session by typing
`! <command>` so the output lands in the conversation.

---

## Open

### [x] AI-1 — Set up the SAM 3D worker on this desktop  ✅ DONE 2026-07-08
**Blocks:** Phase 0 `sam3d_layout` / `sam3d_layout_icp` numbers (the paper's motivating
layout-error experiment) and Phase 2/3 mesh generation.
**Why Claude can't:** the checkpoint is HF-gated and needs your account (accept license +
token). Everything else Claude can do locally.

**GPU note:** this desktop has an **RTX 5090 (Blackwell / sm_120, 32 GB)** — the stock
`environments/default.yml` (CUDA 12.1 / torch 2.5 cu121) will NOT run on it. Use the
repo's **"installation with 5090"** recipe: torch 2.8.0 cu128 + `sam3d-objects-single.yml`
(root of the clone; uses `cuda-nvcc 12.8.93`, `spconv-cu120`, `gsplat 1.5.3` wheel; drops
`flash_attn`/`xformers`/`kaolin` from hard deps). `sam3d_setup.sh` is the OLD cu121 flow —
do not use as-is.

**Done (Claude, 2026-07-08):**
- Clone present at `real2sam3d/sam-3d-objects/` (correct fork).
- Installed miniforge → `~/miniforge3` (conda 26.3.2 + mamba 2.5.0, `conda init bash`).
  `hf` CLI in miniforge base (`~/miniforge3/bin/hf`).
- **`sam3d-objects` env fully built + validated on the RTX 5090**: torch 2.8 cu128 +
  `single.yml` (spconv-cu120, gsplat, MoGe, …) + pytorch3d 0.7.8 built from source.
  Worker import path verified (`from inference import Inference, …` loads clean, GPU seen).
- kaolin skipped (patched out — see below). `sam3d_setup.sh` rewritten to the 5090 recipe.
- **[human]** HF login done (`chrishsu`); gated checkpoint downloading → `checkpoints/hf/`.

**Fork patches committed + pushed** to `christopher-hsu/sam-3d-objects` main (commit
`20081af`): `notebook/inference.py` (kaolin viz imports optional), `flexicubes.py` (kaolin
`check_tensor` → new `sam3d_objects/utils/kaolin_compat.py` shim). Fresh clones get them.

**Checkpoint downloaded**: `checkpoints/hf/` (13 GB, `pipeline.yaml` present).

**End-to-end smoke test PASSED (2026-07-08):** loaded pipeline from `checkpoints/hf`
(72s) + inference on the kidsroom sample (32s) → full output (scale/rotation/translation/
mesh/glb/gs). Env + checkpoint + kaolin/pytorch3d patches confirmed working on the 5090.

**COMPLETE:** the worker ran on real Replica room0 jobs (`~/Data/datasets/sam3d_queue`,
44 job outputs with real `object.glb`/`pose.json`) and both baseline rows are produced:
`results/phase0_replica_sam3d_layout/` (F1@.25 0.58, rot 28°, scale 0.54, Scan2CAD 0.0)
and `.../sam3d_layout_icp/` (F1 0.84, rot 14°). See STATUS.md Phase 0. The only leftover is
Claude-doable, not human-gated: extend `sam3d_layout` from room0 to the all-8-scene
aggregate (tracked in STATUS.md, not here).

### [ ] AI-2 — ScanNet v2 access  *(start early: days-long latency)*
**Blocks:** Phase 5 placement table. **Do:** sign the ScanNet ToS PDF and email per
https://github.com/ScanNet/ScanNet (they reply with `download-scannet.py`). See
`DATASETS.md §2`.

### [ ] AI-3 — Scan2CAD access  *(start early)*
**Blocks:** Phase 5 placement table. **Do:** request form at https://scan2cad.org
(requires ScanNet ToS accepted first). See `DATASETS.md §3`.

### [ ] AI-4 — ShapeNetCore.v2 license (HuggingFace)  *(start early)*
**Blocks:** Phase 5 (CAD models for Scan2CAD; retrieval baseline). **Do:** accept the
license at https://huggingface.co/datasets/ShapeNet/ShapeNetCore. See `DATASETS.md §3`.

### [ ] AI-5 — MetaScenes access  *(start early)*
**Blocks:** Phase 5 (alt sim-ready GT + Scan2Sim baseline). **Do:** Google Form on
https://meta-scenes.github.io. See `DATASETS.md §3`.

### [ ] AI-7 — Coworker's scene-graph benchmark harness + exact metric defs  *(comparability-critical)*
**Blocks:** publishing a *comparable* Objects/Mesh column in the coworker's ProcTHOR/
MolmoSpaces scene-graph table (see `PHASE_SPECS.md` Phase-5 side-thread). The adapter +
pipeline are built and validated (`ProcThorSource`, oracle=1.000 on scene 137); we can
produce numbers now, but our `evaluate()` uses *our* matching definitions.
**Why Claude can't:** it's an artifact only the coworker has.
**Do (from the coworker):** (1) his metric code / exact definitions for **Micro F1,
Many-to-one F1, Macro F1, Matched-per-scene, Objects-per-scene, Class-Free Geo Recall,
Footprint IoU, Chamfer** (IoU threshold? label-aware? object set / which THOR types
count?); (2) the **10th ProcTHOR id** (he named 9; the table says "slice of 10 rooms"
≈ 10 houses); (3) the **split** those ids index (train/val/test — same integer is a
different house per split); (4) his camera-trajectory protocol if he wants frame-set
parity. Until then our column is "indicative, our metric defs" — flag in the caption.
**Update (2026-07-14):** the geometric named metrics are now implemented (Option A —
`eval/metrics.py` micro/macro F1, per-scene counts, class-free scene Chamfer + geo recall;
see `PHASE_SPECS.md` Phase-5). What still needs the coworker: (a) confirm the exact
definitions (object set, IoU threshold, label-aware?) so ours match; (b) his **association
family** defs (many-to-one F1, fragmentation, merge, pairwise) to finish Option B against
`scripts/scene_graph_metrics.py:compute_track_metrics`; plus the 10th id + split as before.

### [x] AI-8 — MolmoSpaces / THOR per-object GT meshes for the Mesh rows  *(DONE 2026-07-14)*
**RESOLVED:** downloaded `isaac/objects/thor` (~1 GB, all 9 houses are iThor assets) and
wired the loader: `data/thor_assets.py` (`usd-core` reads the USDA → canonical `trimesh`;
`ThorAssetLibrary` resolves `assetId`; `fit_canonical_to_obb` places it in the trusted OBB).
`ProcThorSource(gt_mesh="asset")` attaches them (loud box fallback for empty-`assetId`
objects); CLI `--gt-mesh asset` + `--extra mesh`. **Validated on scene 200:** 46/52 real
meshes, fitted mesh-span vs OBB-extent median [1.00,1.02,1.01]; oracle eval geo_recall@5cm
0.987, scene_chamfer ~9 mm. Remaining for the Mesh **numbers**: run the campaign with
`--gt-mesh asset` against real predictions (compute, not gated), and add top-down Footprint
IoU. Known limitation: OBB-fit leaves axis SIGN unresolved (possible 180° flip on strongly
asymmetric assets); refine with the Unity yaw if Chamfer looks off.
**Blocks:** the **Mesh** row-group (Chamfer, Footprint IoU) — Real2USD's differentiator.
Objects rows need none of this. **Why Claude can't:** MolmoSpaces asset download may be
license-gated (Objaverse ODC-BY / THOR assets) and needs your account/acceptance.
**Do:** grab the MolmoSpaces object mesh assets
(https://huggingface.co/datasets/allenai/molmospaces) so `ProcThorSource.gt()` can attach
per-instance GT meshes (currently a box placeholder, `gt_mesh="box"`); then Chamfer/F-score
+ `scene_chamfer_mean_m`/`geo_recall@tau` activate and we add top-down Footprint IoU.
Alternative: extract meshes from the THOR asset db per `assetId`.

**Scope resolved (2026-07-14, assetId audit):** the 9 houses reference **410 unique iThor
asset ids, 0 Objaverse hashes** — so only the **`isaac/objects/thor` source (~1 GB, single
tar)** is needed, NOT the 86 GB `isaac/objects/objaverse` and NOT the 13.1 TB full dataset.
The dataset's `download.py` fetches one source at a time via `--data_source_dir`; no
per-scene filtering. Commands:
```
pip install zstandard datasets huggingface-hub tqdm && huggingface-cli login
curl -L -o /tmp/molmospaces_download.py \
  https://huggingface.co/datasets/allenai/molmospaces/resolve/main/download.py
python /tmp/molmospaces_download.py ~/Data/datasets/molmospaces --list --source isaac  # confirm version
python /tmp/molmospaces_download.py ~/Data/datasets/molmospaces \
  --data_source_dir isaac/objects/thor/20260128
```
Then Claude wires the tar→`assetId`→mesh loader into `ProcThorSource.gt()`.

### [ ] AI-6 — Meta SAM 3 checkpoint (detector)  *(not blocking — YOLOE is live)*
**Blocks:** nothing hard. Phase 2 runs on **YOLOE** (ungated `ultralytics`, auto-downloads
weights, installed via the `detector` uv extra + torch cu128; validated on the 5090
2026-07-08). SAM 3 is the *gated upgrade* — better track persistence on jittery streams,
mask conventions matching SAM 3D — and a detector-ablation row. **Do:** request access on
HuggingFace (gated, like sam-3d-objects). See `DATASETS.md`. Until then, YOLOE (gt/generic/
prompt-free) is the detector for the degradation study.

---

## Done

- [x] **Replica dataset** — downloaded 2026-07-07 to `~/Data/datasets/replica` (RGB-D via
  `download_replica.sh`, GT semantic via `download_replica_semantic.sh`). Unblocked Phase 0.
- [x] **Git commit identity for this repo** — set to personal Gmail 2026-07-07 (per-repo
  override; personal/academic side project). No further action.
