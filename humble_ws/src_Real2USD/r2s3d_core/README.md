# r2s3d_core

ROS-free core library for R2S3D v2 (see `docs/REWORK_PLAN.md`,
`docs/PHASE_SPECS.md` at the repo root). ROS2 nodes import this package; it never
imports ROS. Python 3.10 (pinned to match ROS Humble so it installs editable in the
docker container).

## Install

```bash
cd humble_ws/src_Real2USD/r2s3d_core
uv sync --extra dev                 # core + pytest
uv sync --extra registration       # + open3d, for the ICP baseline variant B
uv run pytest -q
```

## Layout

```
src/r2s3d_core/
  data/       SequenceSource interface + backends (replica, synthetic)
  eval/       geometry, metrics, run (runner), table
  baselines/  oracle (sanity), sam3d_layout (+icp)
  frames/ tracks/ registration/ refine/ recon/   (Phase 1+ , stubs)
tests/        golden metric tests, harness smoke test, data-gated Replica test
results/      run.json per experiment (gitignored except *.json)
```

## Data

Data root defaults to `$R2S3D_DATA` or `~/Data/datasets` (this workstation has no
writable `/data`). Download Replica:

```bash
bash scripts/datasets/download_replica.sh          ~/Data/datasets/replica  # RGB-D + poses (~12 GB)
bash scripts/datasets/download_replica_semantic.sh ~/Data/datasets/replica  # GT semantic (~100 GB extracted)
```

The RGB-D loader is verified against `room0_mesh.ply`: back-projected depth lands
within the room bounds (validates intrinsics, depth scale, camera convention, and
the Z-up world frame — see `data/replica.py`).

## Run the Phase-0 evaluation

```bash
# harness sanity (no data/GPU needed): GT-as-prediction must score perfectly
uv run python -m r2s3d_core.eval.run --source synthetic --scene synthetic --method oracle

# on Replica once GT semantic assets are present:
uv run python -m r2s3d_core.eval.run --source replica --scene room0 --method oracle
uv run python -m r2s3d_core.eval.run --source replica --scene room0 --method sam3d_layout
uv run python -m r2s3d_core.eval.run --source replica --scene room0 --method sam3d_layout_icp

# render the comparison table from run.json files
uv run python -m r2s3d_core.eval.table --glob 'results/phase0_*'
```

Each run writes `results/phase0_<name>/run.json` with `git_sha`, full `config`,
per-scene + aggregate metrics, `wall_time_s`, `created_at`.

### Metrics (`eval/metrics.py`)

Hungarian matching on exact OBB-IoU (0.25 primary, 0.5 secondary), label-agnostic;
per matched pair: centroid L2, symmetry-aware rotation error (Scan2CAD-style
per-class yaw groups), per-axis scale-ratio error, Chamfer-L1, F-score@5cm/2cm;
Scan2CAD accuracy (≤20 cm ∧ ≤20° ∧ ≤20% scale); scene P/R/F1; duplicate rate;
count ratio. Golden-value tests in `tests/test_metrics.py`.

## Gated dependency: SAM3D

`sam3d_layout` / `sam3d_layout_icp` call Meta's `sam-3d-objects` via the disk-queue
worker (`real2sam3d/scripts_sam3d_worker/`), which needs its own `sam3d-objects`
conda env, the external repo, and the **gated HuggingFace checkpoint** (request
access — human step, see `docs/DATASETS.md`). The baseline writes jobs to the queue
and reads cached outputs (keyed by input hash, so reruns are free); if outputs are
missing it raises a loud, actionable error telling you to run the worker. The pure
placement math and best-view selection are unit-tested independently
(`tests/test_sam3d_layout.py`), so the baseline is correct-by-construction once
SAM3D outputs exist.
