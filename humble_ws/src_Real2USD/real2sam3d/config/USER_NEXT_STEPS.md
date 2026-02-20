# Real2SAM3D Next Steps

Use this as the planning page for pipeline improvements.

## Current state (from code)

- Transform pipeline parity with `demo_go2` is fixed and validated.
- Worker writes raw SAM3D outputs and optional compare artifacts.
- Injector computes and writes world transforms and `object_odom.glb`.
- Registration refines pose from `initial_position` / `initial_orientation`.
- Debug tooling is in place:
  - `transform_debug.json` (injector path)
  - `demo_go2_compare.json` (worker path)

For exact commands and debug usage, use `config/RUN_PIPELINE.md`.

## Most valuable next improvements

1) Transform regression test (highest priority)
- Add a small deterministic test that feeds a fixed basis + known odom/SAM3D params.
- Assert injector and worker basis outputs match within tolerance.
- Goal: prevent future parity regressions.

2) Better registration quality gates
- Log per-job ICP fitness and inlier RMSE to a json/csv summary.
- Add threshold-based fallback policy (`use initial pose` vs `accept ICP`) as tunable params.
- Goal: easier diagnosis when final scene is wrong but transforms are right.

3) Queue observability
- Write per-job stage timing (`job_written`, `worker_done`, `injector_done`, `registered`) and latency.
- Goal: identify bottlenecks and backlog behavior quickly.

4) Retrieval quality
- Add an offline eval script: top-k retrieval accuracy on a small labeled set.
- Compare single-view indexing vs multi-view indexing.
- Goal: quantify if FAISS changes help before modifying runtime logic.

5) Scene output consistency checks
- Add a validator script that checks each output job has:
  - `pose.json` in `z_up_odom`
  - valid 4x4 transforms
  - `object.glb` and `object_odom.glb`
- Goal: catch partial-job failures early.

## Suggested execution order

1. Add transform regression test.
2. Add registration metrics logging.
3. Add job timing/latency summary.
4. Run 1-2 bag sessions and review metrics.
5. Then decide retrieval or registration tuning based on data.

## Notes

- Keep debug artifacts and env flags; they have already proven useful.
- Avoid changing transform math again without test coverage and side-by-side JSON diffs.
