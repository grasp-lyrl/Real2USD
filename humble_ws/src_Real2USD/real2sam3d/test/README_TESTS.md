# Running SAM3D integration tests

**Run the main pipeline with ROS2:** `ros2 launch real2sam3d real2sam3d.launch.py` (see config/context_sam3d_integration.txt and config/SAM3D_BUILD_PLAN.md). Tests below are pytest only.

**Where to run tests:** Run pytest **inside the project’s Docker container** so the workspace is built, `source install/setup.bash` is available, and rclpy/ROS work. Integration checks (e.g. with a bag) are also done there.

Build and source the workspace, then run the SAM3D-related tests:

```bash
cd humble_ws
colcon build --packages-select real2sam3d
source install/setup.bash
python3 -m pytest src_Real2USD/real2sam3d/test/test_sam3d_*.py -v
```

Or from the real2sam3d package directory (with workspace sourced):

```bash
cd humble_ws && source install/setup.bash
python3 -m pytest src_Real2USD/real2sam3d/test/test_sam3d_job_writer.py src_Real2USD/real2sam3d/test/test_sam3d_worker.py src_Real2USD/real2sam3d/test/test_sam3d_track_id_and_dedup.py -v
```

Use `python3 -m pytest` if the `pytest` command is not on your PATH.

**What each test file does:**

- **test_sam3d_job_writer.py**: Exercises `write_sam3d_job()` with synthetic data; checks that job dirs contain `rgb.png`, `mask.png`, `depth.npy`, `meta.json` and that meta has the expected structure. No ROS or live topics.
- **test_sam3d_worker.py**: Creates a fixture job dir, runs `load_job` and `process_one_job(..., dry_run=True)`; checks that output dir gets `pose.json` and `object.ply`. No SAM3D model required.
- **test_sam3d_track_id_and_dedup.py**: Unit tests for `track_ids_from_boxes_id()` (no ultralytics needed) and for job writer dedup (`_should_skip_dedup` / `_record_written`). Dedup tests need rclpy and the built real2sam3d package (run inside Docker after `colcon build` and `source install/setup.bash`).

Full build plan and phase-to-test mapping: see `config/SAM3D_BUILD_PLAN.md`.
