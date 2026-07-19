# ACTION ITEMS — things only you (the human) can do

Claude cannot request dataset access, accept licenses, obtain gated checkpoints, log
into services, or make policy/IP calls. **Whenever a build step hits one of those, Claude
adds an item here (explicit: what / why it blocks / how / where) and flags it in chat.**
You work these; mark them done. `STATUS.md` links here for anything blocked.

Convention: each item has an ID (AI-N, stable), a **blocks** field, and a status box.
When done, check the box and add the date; leave it in place for provenance.

### [x] AI-9 — AI2-THOR rendering hung (Wayland session)  *(RESOLVED 2026-07-19)*
**FIX:** the machine had booted into a GNOME **Wayland** session (no GPU Xorg → AI2-THOR's X11/GLX
hung). Chris switched to **Ubuntu on Xorg** (session now x11); render works — new-scene test
`GOT FRAME @ 23s, depth 480×640, valid 1.00`. The session display is now **`:0`** (was `:1` before)
with `XAUTHORITY=/run/user/1003/gdm/Xauthority`. `run_paper_sim_val10.sh` updated to use those
(overridable via `R2S3D_DISPLAY`/`R2S3D_XAUTHORITY`). If it hangs again, first check
`loginctl show-session <id> -p Type` is `x11`, not `wayland`. (NB `glxinfo` is NOT installed here —
don't use its "timeout" as evidence; test an actual AI2-THOR render instead.) Original diagnosis below.

### [~] AI-9 (orig diagnosis) — AI2-THOR rendering hung on display `:1`  *(2026-07-19)*
**Blocks:** rendering ANY new ProcTHOR scene → the **val-10 sim cluster** aggregate (and any future
sim run on an uncached scene). Cached scenes (s200/137/428) still replay fine.
**Symptom:** `make_source` builds, but pulling the first frame hangs >200 s with no error / no Unity
process; a clean retry (hung procs cleared) reproduces it → environmental, not a stale lock.
**Root cause (diagnosed):** `DISPLAY=:1 glxinfo` also hangs/times out while `nvidia-smi` is healthy
(RTX 5090 idle) → the **GL rendering context on `:1` is wedged**, not the GPU. `:1` is a GNOME/mutter
session; it worked 2026-07-13 (caches + `~/.ai2thor` build locks dated then).
**How to fix (human — Claude won't restart X, could kill the desktop):** reset the GPU-backed X/GL on
`:1` — e.g. restart the X server / GNOME session on `:1`, or whatever headless GPU-X the render used;
then verify with `! DISPLAY=:1 glxinfo | grep -i "OpenGL renderer"` (must show NVIDIA, not llvmpipe/
timeout). Once GL works, re-run `scripts/run_paper_sim_val10.sh` (idempotent — resumes).
**Fallback if not fixed:** s200-only sim (already in hand); val-10 breadth deferred.

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

### [~] AI-7 — Coworker's scene-graph benchmark metric defs  *(ANSWERED 2026-07-14; reconciliation in progress)*
**Blocks:** publishing a *comparable* Objects/Mesh column. **CORRECTION (2026-07-19, per Chris):
SuperMap (AirLab/Super Odometry, RSS'26) is NOT the coworker's method** — it is a separate recently-
published system (no code released), earlier notes conflated the two. The coworker's benchmark is
their OWN harness; the metric code we hold is `scripts/scene_graph_metrics.py` (a DAAAM-lineage lexicon
is used for the ProcTHOR table).
**Read the coworker code (2026-07-19):** `scene_graph_metrics.py` defines `symmetric_chamfer_distance`,
`point_coverage_f_score`, `assignment_precision_recall_f1`, `compute_track_metrics` — a POINT-CLOUD
tracking+geometry stack. **Chamfer DEFINITIVELY CONFIRMED == our `scene_chamfer_mean_m`** (mean of the
two directional mean-NN distances). **But NO footprint-IoU, NO Micro/Macro F1, NO class-free recall in
this file** → those live in the OTHER harness (`harness.py`/`objects.py`, NOT in our repo). So **Q5
(footprint def) is STILL unresolved** — need the footprint function specifically. **Actionable: lead
the Mesh cross-method comparison with CHAMFER (verified), treat Footprint IoU as internal-ablation-only
until the coworker's footprint code is obtained.**

**RESOLVED ANSWERS (from coworker, 2026-07-14) — two INVALIDATE our current numbers:**
1. **Split = procthor-10k `val`** (val.jsonl), NOT train. Same integer id is a *different
   house* per split → **every ProcTHOR result so far (train split) is on the wrong houses**
   and must be regenerated on `val`.
2. **Matching = Hungarian on CENTROID (Euclidean) distance ≤ τ**, default **τ=1.0 m**, swept
   [0.25,0.5,0.75,1.0,1.5] m. **NOT** 3D OBB IoU. Our IoU@0.25 is a *new, stricter, non-
   comparable* protocol (tiny/thin ProcTHOR objects pass a 1 m gate but fail IoU@0.25).
3. **Object F1 (Micro/Macro) is LABEL-AWARE** (match = centroid≤τ ∧ label agreement). The
   label-agnostic headline in their work is *class-free recall*, not a label-agnostic F1 —
   so our label-agnostic `f1` is a NEW number, not a reproduction.
4. **Filter GT to the same mapped ProcTHOR vocabulary as predictions** — drop STRUCTURAL
   (background/unlabelled/void, wall/floor/curtain/…) + DAAAM-lexicon-unmatched classes on
   BOTH sides. Their GT (val.jsonl) ≈ **74.7 objects/scene**. Counting all GT → structural
   classes become unrecoverable FNs, asymmetrically depressing recall. **NEED the DAAAM
   lexicon / mapping from coworker.**
5. **Chamfer:** scene-level pooled, symmetric = mean of the two directional means
   ((pred→gt + gt→pred)/2) — **our `scene_chamfer_mean_m` already matches.** ✓ (can also
   report the two halves = accuracy/completeness).
6. **Class-Free Geo Recall = 1 m object-CENTROID recall** (GT object found if ANY detection
   within 1 m, label ignored); published Clio 0.873, MoM 0.778. This is **object-level, NOT
   the 2/5 cm surface metric** — our `geo_recall@tau` is mislabeled: it's a surface point-
   coverage F-score (fine, but rename; different metric). Note: saturates ~1.0 for dense
   point-field methods → they report N/A there.
7. **Association "detection" = per-frame instance-mask observation** (Option B): expose
   `detection_id → (oracle/GT object id, predicted track id)` at per-mask granularity.
8. **"Many-to-one F1" = fragment-collapsing object F1** (dominant-overlap, lets many pred
   match one GT); `micro_f1 − many_to_one_f1` = over-segmentation penalty. Distinct from the
   *pairwise* F1 in `scene_graph_metrics.py` (exact same-object pair F1). Label both clearly.
9. **10th id = 434.** Canonical slice `[137,200,428,434,534,569,573,683,771,912]` — **we were
   missing 434.** (771 has a short/sparse trajectory and is dropped in some runs — confirm.)
10. **Split = val** (see #1).
11. **Trajectory/camera:** GT poses; 640×480 pinhole fx=fy=320 cx=cy=(320,240) (90° HFOV);
    depth invalid=0; dense ~10 Hz (~504–6452 frames), stride 5–10; poses in `poses.csv` ROS
    body frame, Z-up world → OpenCV c2w via `R_BODY_OPTICAL=[[0,0,1],[-1,0,0],[0,-1,0]]`.
    For frame-set parity we'd consume THEIR trajectory, not our AI2-THOR-rendered one.

**CONFIRMED VS SOURCE (2026-07-14, harness.py + objects.py) — supersedes #2/#3/#8 above:**
- **Q2 matching = GREEDY on 2D top-down (X,Y) centroid distance ≤ τ (default 1.0 m).** NOT
  Hungarian, NOT 3D, NOT IoU (`use_hungarian=False`; `_xy_distance` uses X,Y only). Our
  `cd_*` now uses greedy-XY (`eval/metrics.py:greedy_match_centroid`).
- **Q3 = NO GT FILTERING, no size floor** — `_load_gt_objects` takes the entire objects map
  (doors/windows included; category defaults `UNLABELLED`). **This REMOVES the DAAAM-lexicon
  blocker — we do NOT need it to produce comparable numbers.** (Asymmetry baked into their
  harness: predictions drop STRUCTURAL, GT doesn't; unlocalizable GT (position=None) are FNs
  in micro but excluded from class-free recall.) NB our `procthor._THOR_EXCLUDE_TYPES` drops
  floor/wall/window/door/room from GT — diverges from their "count all"; decide whether to
  stop excluding for parity.
- **Q4 macro set = GT ∪ predicted** ✓ (unweighted mean per-category F1; hallucinated/missed
  categories score 0). Caveat: they bucket on **raw case-sensitive** category strings; we
  normalise case (harmless if our two sides agree on casing).
- **Q8 many-to-one F1 = any-/dominant-overlap association accuracy** (GT with ≥1 valid same-
  category detection within τ; detection with ≥1 valid same-category GT), label-aware, NOT
  exact-pair. Object-level, needs NO track provenance → implemented now as
  `cd_micro_f1_many_to_one`. (The pairwise F1 in `scene_graph_metrics.py` is a different
  Stack-B quantity.)

**Label protocol RESOLVED (confirmed with coworker, 2026-07-14):** open-set methods ARE given the
scene's label vocabulary ("in-set"); label-aware Object F1 snaps each predicted label to the
**CLOSEST in-set word by CLIP text cosine** — NOT exact string. So `refrigerator→fridge`,
`tv→television`, `couch→sofa`. (Current harness forces the argmax, no cosine floor.)
Implemented as `eval/label_map.py` (+ `--label-map clip` on `eval.run`/`eval.rescore`); kept
out of the torch-free `metrics.py`. Impact (val s200, cached preds): `gt` micro-F1 unchanged
0.325 (already in-set = no-op ✓); `generic` 0.152→0.203; `pf` 0.101→0.201. Our old exact-string
scoring under-credited open-vocab/`generic`/`pf` — this is REQUIRED for a comparable label F1.

**Still to get from coworker (reconciliation, non-blocking — needed only to *bit-match* his
label F1):**
- (r1) **Exact CLIP model + prompt template.** We used `ViT-B/32`, `"a photo of a {}"`. Which
  model (ViT-B/32? L/14? open_clip variant?) and template does the harness use? Different
  choices shift which synonyms snap.
- (r2) **Cosine threshold / fallback?** Does he force the argmax (our default) or drop below a
  floor to `unknown`/background? (A cosine threshold is the natural refinement — matters for
  `pf`'s forced-garbage maps like `aircraft model→book`.)
- (r3) **In-set = per-scene GT vocab or a fixed global benchmark vocab?** We snap to the
  scene's GT categories; confirm the target set.
Plus prior: (c) frame-set parity (#11) — deferred (render `val` ourselves, flag "our
trajectory"). **No hard blocker remains for the Objects column** — regenerate on `val` (+id 434),
greedy-XY matching, `--label-map clip`.

**(Q5) — ELEVATED to a real gate for the MESH cross-method claim (2026-07-19).** Need the coworker's
EXACT Footprint-IoU + Chamfer definition (the literal `evaluate_mesh` code). WHY it now blocks: our
open-vocab cluster scores Footprint IoU ~0.44 vs the published field 0.006–0.095, and we confirmed
this gap is NOT a gt-vocab artifact (open-vocab `generic` cluster still 0.445). So either our
front-end genuinely covers more, OR the metric is computed differently — the field matches very few
objects/scene (Hydra 5.6, DAAAM 3.5) vs our ~65, so a per-object / recall-weighted footprint (vs our
scene-level pooled) would explain their low numbers as coverage, not quality. Cannot make ANY
cross-method Mesh claim until this is pinned. ASK: is Footprint IoU scene-level pooled occupancy or
per-object-matched? cell size? mesh-footprint vs OBB-footprint? Chamfer confirmed scene-level pooled
symmetric (#5 ✓). This is a one-question ask, not compute.

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
