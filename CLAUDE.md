# Real2USD / real2sam3d

Research code for "Asset-Centric Metric-Semantic Maps of Indoor Environments"
(arXiv 2510.10778). A Unitree Go2 robot builds an object-centric USD scene map:
detect objects (YOLOE) → retrieve (CLIP/FAISS) or generate (Meta SAM 3D) a mesh →
register it into the metric scene (ICP) → reconcile and export USD/GLB.

## Current status (July 2026)

The paper was rejected from IROS 2026 and is being reworked for resubmission.

- **`docs/STATUS.md` is the live progress tracker — read it first to know where we are;
  `docs/ACTION_ITEMS.md` is the explicit list of things only the human can do.**
- **Read `docs/REWORK_PLAN.md` (strategy) and `docs/PHASE_SPECS.md` (interfaces,
  resolved decisions, defaults, definitions of done) before changing anything in
  `real2sam3d` or `r2s3d_core`** — the plan contains
  the full assessment of v1 and the phased v2 redesign (ObjectTrack multi-view fusion,
  Sim(3) registration, multi-view refinement, public-benchmark evaluation).
- v2 core logic lives in the ROS-free `humble_ws/src_Real2USD/r2s3d_core/` uv package
  (Phase 0+); ROS2 nodes become thin wrappers over it.
- `v2-rework` branch: all rework happens here. `main` + tag `v1-iros2026` are the frozen
  paper state (kept for reproducibility and an *optional* "vs v1" ablation row — never a
  blocker). A frozen v1 worktree may exist at `../Real2USD-v1`.
- Baselines and evaluation run on **public datasets** (Replica first, then
  ScanNet/Scan2CAD) through the `SequenceSource` adapter — not on v1's custom-bag
  outputs. The number to beat is SAM3D's own predicted layout (`make_scene()` lives in
  the external `sam-3d-objects` repo; in-repo the stand-in is `sam3d_layout`).
- Future docs go in `docs/`.

## Documentation discipline (keep these current AS YOU BUILD)

These docs are load-bearing for continuity across sessions — treat updating them as part
of "done", not an afterthought:

- **`docs/STATUS.md`** — update at the end of every milestone/session: the phase
  dashboard, current focus, what is *verified* (not just written), and blockers. "Done"
  means tests pass / numbers produced.
- **`docs/ACTION_ITEMS.md`** — the moment you hit something only the human can do (dataset
  access, gated checkpoints, licenses, credentials/logins, policy or IP decisions), add an
  explicit item (what / why it blocks / how / where) AND surface it in chat. Check items
  off when resolved; leave them for provenance.
- **`docs/PHASE_SPECS.md`** — when a resolved decision or default changes in practice,
  update it there and note it in the run record. It is not immutable.
- Keep the split clean: REWORK_PLAN = *why/strategy*, PHASE_SPECS = *how/interfaces*,
  STATUS = *where we are*, ACTION_ITEMS = *what the human must do*. Link, don't duplicate.
- Every eval run writes `results/<phase>_<name>/run.json`; tables/plots regenerate from
  those, never hand-edited.

## Layout

- `humble_ws/src_Real2USD/real2sam3d/` — the active package (SAM3D-based pipeline).
  ROS2 nodes in `real2sam3d/`, SAM3D worker (separate conda env, disk-queue handoff via
  `input/`/`output/` job dirs) in `scripts_sam3d_worker/`.
- `humble_ws/src_Real2USD/real2usd/` — v1 predecessor (CLIP retrieval + ICP, no SAM3D).
- `humble_ws/evaluations/` — metrics harness (3D IoU, open-set P/R, comparisons vs
  Clio/SAM3D). Extend this; don't fork it.
- `humble_ws/src_Real2USD/scripts_isaacsim/` — Isaac Sim preprocessing + USD building
  (runs OUTSIDE the docker container).
- `humble_ws/src/`, `src_go2_ros2_webrtc_sdk/` — Isaac Sim ROS workspaces and Go2 SDK
  (mostly third-party forks; rarely need changes).

## Environment & running

- ROS2 Humble in Docker (see root `build_*.sh` / `run_*.sh`, README.md). Isaac Sim 4.5
  and the SAM3D worker run outside the container.
- Build: `colcon build` inside the container, then `source install/setup.bash`.
- Typical run: play a ros2 bag + `ros2 launch real2sam3d real2sam3d.launch.py`; the SAM3D
  worker is started separately in its `sam3d-objects` conda env (`sam3d_setup.sh`).
- Data: v1/robot pipeline expects `/data`, but **this desktop has no writable `/data`** —
  v2 datasets live under `~/Data/datasets/` (`r2s3d_core` resolves `$R2S3D_DATA` else that;
  SAM3D queue at `~/Data/datasets/sam3d_queue`).

## Gotchas

- "SAM3D" here = **Meta's sam-3d-objects** (arXiv 2511.16624), NOT Yang et al. 2023
  SegmentAnything3D. Keep citations/naming disambiguated.
- SAM 3D's predicted translation/scale are unreliable (normalized camera coords, ~3x
  scale errors documented upstream). v2 policy: use SAM3D for shape only; poses must come
  from registration against sensor depth.
- The coordinate-frame chain (raw mesh → PyTorch3D → camera → odom) uses hardcoded
  constants in `real2sam3d/ply_frame_utils.py` and camera extrinsics duplicated in three
  nodes. It is fragile — do not touch transforms without round-trip tests (v2 Phase 1
  consolidates this into a tested `frames.py`).
- Several failure paths are currently silent (FAISS load failure, low-fitness ICP falling
  back to SAM3D pose, malformed pose.json). When editing, make fallbacks loud and stamp
  outputs with which code path produced them.
