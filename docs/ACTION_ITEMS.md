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

### [ ] AI-1 — Set up the SAM 3D worker on this desktop
**Blocks:** Phase 0 `sam3d_layout` / `sam3d_layout_icp` numbers (the paper's motivating
layout-error experiment) and Phase 2/3 mesh generation.
**Why Claude can't:** the checkpoint is HF-gated and needs your account; the conda env +
external repo aren't on this desktop (`conda env list` empty, no repo clone).
**Do:**
1. Clone the fork used in v1: `christopher-hsu/sam-3d-objects` (repo expects `notebook/`
   + `checkpoints/`).
2. Create the `sam3d-objects` conda env (`sam3d_setup.sh`).
3. Obtain the gated SAM 3D (sam-3d-objects, arXiv 2511.16624) checkpoint on HuggingFace.
4. Tell Claude the paths; Claude will run the worker against the queue at
   `~/Data/datasets/sam3d_queue` and fill the baseline rows (outputs are input-hash cached).

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

### [ ] AI-6 — Meta SAM 3 checkpoint (detector)
**Blocks:** Phase 2 primary detector (YOLOE is the fallback, so not hard-blocking). **Do:**
request access on HuggingFace (gated, like sam-3d-objects). See `DATASETS.md`.

---

## Done

- [x] **Replica dataset** — downloaded 2026-07-07 to `~/Data/datasets/replica` (RGB-D via
  `download_replica.sh`, GT semantic via `download_replica_semantic.sh`). Unblocked Phase 0.
- [x] **Git commit identity for this repo** — set to personal Gmail 2026-07-07 (per-repo
  override; personal/academic side project). No further action.
