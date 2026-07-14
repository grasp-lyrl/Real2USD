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

### [ ] AI-8 — MolmoSpaces / THOR per-object GT meshes for the Mesh rows  *(not blocking Objects)*
**Blocks:** the **Mesh** row-group (Chamfer, Footprint IoU) — Real2USD's differentiator.
Objects rows need none of this. **Why Claude can't:** MolmoSpaces asset download may be
license-gated (Objaverse ODC-BY / THOR assets) and needs your account/acceptance.
**Do:** grab the MolmoSpaces object USD/mesh assets
(https://huggingface.co/datasets/allenai/molmospaces) so `ProcThorSource.gt()` can attach
per-instance GT meshes (currently `mesh=None`); then Chamfer/F-score activate and we add
top-down Footprint IoU. Alternative: extract meshes from the THOR asset db per `assetId`.

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
