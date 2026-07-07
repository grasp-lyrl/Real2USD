# Real2USD / real2sam3d

Research code for "Asset-Centric Metric-Semantic Maps of Indoor Environments"
(arXiv 2510.10778). A Unitree Go2 robot builds an object-centric USD scene map:
detect objects (YOLOE) → retrieve (CLIP/FAISS) or generate (Meta SAM 3D) a mesh →
register it into the metric scene (ICP) → reconcile and export USD/GLB.

## Current status (July 2026)

The paper was rejected from IROS 2026 and is being reworked for resubmission.

- **Read `docs/REWORK_PLAN.md` before changing anything in `real2sam3d`** — it contains
  the full assessment of v1 and the phased v2 redesign (ObjectTrack multi-view fusion,
  Sim(3) registration, multi-view refinement, public-benchmark evaluation).
- `v2-rework` branch: all rework happens here. `main` + tag `v1-iros2026` are the frozen
  paper baseline (v1 must stay reproducible — it is an ablation row in the resubmission).
  A frozen v1 worktree may exist at `../Real2USD-v1`.
- Future docs go in `docs/`.

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
- Data lives under `/data` (FAISS index, sam3d_queue, bags, preprocessed USD pkls).

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
