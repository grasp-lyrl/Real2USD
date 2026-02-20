# Copyright 2025 Real2SAM3D contributors.
# Tests for SAM3D worker: load_job and process_one_job (dry-run).
# Run: python3 -m pytest src_Real2USD/real2sam3d/test/test_sam3d_worker.py -v

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Worker script lives in scripts_sam3d_worker; add parent so we can import run_sam3d_worker's functions
import sys
_real2sam3d_root = Path(__file__).resolve().parents[1]
_worker_dir = _real2sam3d_root / "scripts_sam3d_worker"
if str(_worker_dir) not in sys.path:
    sys.path.insert(0, str(_worker_dir))
from run_sam3d_worker import load_job, process_one_job


def _make_fixture_job(job_path: Path) -> None:
    """Create a minimal valid job directory for worker tests."""
    job_path.mkdir(parents=True, exist_ok=True)
    rgb = np.random.randint(0, 255, (32, 48, 3), dtype=np.uint8)
    mask = np.zeros((32, 48), dtype=np.uint8)
    mask[8:24, 12:36] = 255
    depth = np.random.rand(32, 48).astype(np.float32) * 2.0
    import cv2
    cv2.imwrite(str(job_path / "rgb.png"), rgb)
    cv2.imwrite(str(job_path / "mask.png"), mask)
    np.save(str(job_path / "depth.npy"), depth)
    meta = {
        "job_id": job_path.name,
        "track_id": 7,
        "label": "Chair",
        "odometry": {
            "position": [1.0, 2.0, 0.0],
            "orientation": [0.0, 0.0, 0.0, 1.0],
        },
    }
    with open(job_path / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def test_load_job():
    """load_job returns rgb, mask, depth, meta from a valid job dir."""
    with tempfile.TemporaryDirectory() as tmp:
        job_path = Path(tmp) / "fixture_job"
        _make_fixture_job(job_path)
        rgb, mask, depth, meta = load_job(job_path)
        assert rgb is not None and rgb.ndim == 3
        assert mask is not None and mask.ndim == 2
        assert depth is not None and depth.ndim == 2
        assert meta["job_id"] == "fixture_job"
        assert meta["track_id"] == 7
        assert "odometry" in meta


def test_load_job_missing_file_raises():
    """load_job raises FileNotFoundError when a required file is missing."""
    with tempfile.TemporaryDirectory() as tmp:
        job_path = Path(tmp) / "incomplete_job"
        job_path.mkdir()
        (job_path / "rgb.png").write_bytes(b"x")
        (job_path / "mask.png").write_bytes(b"x")
        (job_path / "depth.npy").write_bytes(b"x")
        # no meta.json
        with pytest.raises(FileNotFoundError, match="meta.json"):
            load_job(job_path)


def test_process_one_job_dry_run_writes_output():
    """process_one_job with dry_run=True writes pose.json and object.ply to output dir."""
    with tempfile.TemporaryDirectory() as tmp:
        input_dir = Path(tmp) / "input"
        output_dir = Path(tmp) / "output"
        output_dir.mkdir()
        job_path = input_dir / "test_job_1"
        _make_fixture_job(job_path)

        success = process_one_job(job_path, output_dir, dry_run=True)
        assert success is True

        out_path = output_dir / "test_job_1"
        assert (out_path / "pose.json").exists()
        assert (out_path / "object.ply").exists()
        with open(out_path / "pose.json") as f:
            pose = json.load(f)
        assert pose["track_id"] == 7
        assert pose["label"] == "Chair"
        assert "go2_odom_position" in pose and "go2_odom_orientation" in pose
