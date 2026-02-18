# SAM3D Worker (disk-based handoff)

Run on the **host** in the **sam3d-objects** conda env. Processes job directories produced by `sam3d_job_writer_node` (pipeline runs in Docker). Setup: install sam-3d-objects on the host (keep repo outside Real2USD); see `config/USER_NEXT_STEPS.md`.

## Queue layout

- `queue_dir/input/<job_id>/` — per job: `rgb.png`, `mask.png`, `depth.npy`, `meta.json`. After successful processing, the job is **moved** to `queue_dir/input_processed/<job_id>/` so the worker moves on to the next job (FIFO by oldest first).
- `queue_dir/output/<job_id>/` — per result: `object.ply` (or `.usd`), `pose.json`

## Run (on host; queue dir = `/data/sam3d_queue`)

```bash
conda activate sam3d-objects
cd /path/to/real2sam3d/scripts_sam3d_worker
python run_sam3d_worker.py --queue-dir /data/sam3d_queue --sam3d-repo /path/to/sam-3d-objects
```

**`--sam3d-repo`** must point to your sam-3d-objects repo clone (the directory that contains `notebook/` and `checkpoints/`). Required for real inference; omit and use `--dry-run` to test without SAM3D.

The queue lives at `/data/sam3d_queue` on both host and container. If the launch uses per-run subdirs, pass e.g. `--queue-dir /data/sam3d_queue/run_YYYYMMDD_HHMMSS`.

## Dry-run (no SAM3D; e.g. inside Docker to confirm pipeline)

```bash
python run_sam3d_worker.py --queue-dir /data/sam3d_queue --once --dry-run
```

Validates one job, writes dummy `pose.json` and placeholder `object.ply`, and exits. Use with `ros2 launch ... run_sam3d_worker:=true` to confirm the full path without the conda env.

## With real SAM3D (on host)

Ensure the sam-3d-objects repo is on the host and the worker can import it (e.g. `PYTHONPATH` or run from a path that includes the repo). Run without `--dry-run`; results are written to `output/<job_id>/`.
