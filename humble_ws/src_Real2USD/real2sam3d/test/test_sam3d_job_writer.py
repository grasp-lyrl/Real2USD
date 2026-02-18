# Copyright 2025 Real2SAM3D contributors.
# Tests for SAM3D job writer (write_sam3d_job and job directory layout).
# Run: python3 -m pytest src_Real2USD/real2sam3d/test/test_sam3d_job_writer.py -v

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Import from real2sam3d (works when run from workspace with real2sam3d built)
from real2sam3d.sam3d_job_writer_node import write_sam3d_job


def test_write_sam3d_job_produces_all_files():
    """write_sam3d_job creates rgb.png, mask.png, depth.npy, meta.json."""
    with tempfile.TemporaryDirectory() as tmp:
        job_path = Path(tmp) / "job1"
        h_full, w_full = 100, 120
        rgb = np.random.randint(0, 255, (40, 50, 3), dtype=np.uint8)
        depth_full = np.random.randint(0, 5000, (h_full, w_full), dtype=np.uint16)
        seg_pts = np.array([[25, 20], [26, 20], [25, 21], [30, 25]], dtype=np.int32)
        crop_bbox = [10, 10, 60, 50]
        meta = {
            "job_id": "job1",
            "track_id": 42,
            "label": "Chair",
            "frame_id": "odom",
            "stamp_sec": 0,
            "stamp_nanosec": 0,
            "camera_info": {"K": [1.0] * 9, "width": w_full, "height": h_full},
            "odometry": {
                "position": [0.0, 0.0, 0.0],
                "orientation": [0.0, 0.0, 0.0, 1.0],
            },
        }
        write_sam3d_job(rgb, depth_full, seg_pts, meta, job_path, crop_bbox=crop_bbox)

        assert (job_path / "rgb.png").exists()
        assert (job_path / "mask.png").exists()
        assert (job_path / "depth.npy").exists()
        assert (job_path / "meta.json").exists()


def test_write_sam3d_job_meta_structure():
    """meta.json contains job_id, track_id, label, odometry, crop_bbox, camera_info."""
    with tempfile.TemporaryDirectory() as tmp:
        job_path = Path(tmp) / "job2"
        rgb = np.zeros((20, 30, 3), dtype=np.uint8)
        depth_full = np.zeros((80, 100), dtype=np.uint16)
        seg_pts = np.array([[40, 10], [41, 10]], dtype=np.int32)
        meta = {
            "job_id": "job2",
            "track_id": 1,
            "label": "Table",
            "frame_id": "odom",
            "stamp_sec": 1,
            "stamp_nanosec": 0,
            "camera_info": {"K": [1.0] * 9, "width": 100, "height": 80},
            "odometry": {
                "position": [1.0, 2.0, 0.5],
                "orientation": [0.0, 0.0, 0.0, 1.0],
            },
        }
        write_sam3d_job(rgb, depth_full, seg_pts, meta, job_path, crop_bbox=[0, 0, 30, 20])

        with open(job_path / "meta.json") as f:
            loaded = json.load(f)
        assert loaded["job_id"] == "job2"
        assert loaded["track_id"] == 1
        assert loaded["label"] == "Table"
        assert "odometry" in loaded and "position" in loaded["odometry"]
        assert "orientation" in loaded["odometry"]
        assert "crop_bbox" in loaded and len(loaded["crop_bbox"]) == 4
        assert "camera_info" in loaded


def test_write_sam3d_job_infers_bbox_from_seg_points():
    """Without crop_bbox, bbox is inferred from seg_points (with padding)."""
    with tempfile.TemporaryDirectory() as tmp:
        job_path = Path(tmp) / "job3"
        rgb = np.zeros((15, 25, 3), dtype=np.uint8)
        depth_full = np.zeros((60, 80), dtype=np.uint16)
        seg_pts = np.array([[20, 15], [40, 35], [25, 30]], dtype=np.int32)
        meta = {
            "job_id": "job3",
            "track_id": 0,
            "label": "Misc",
            "frame_id": "odom",
            "stamp_sec": 0,
            "stamp_nanosec": 0,
            "camera_info": {"K": [], "width": 80, "height": 60},
            "odometry": {"position": [0, 0, 0], "orientation": [0, 0, 0, 1]},
        }
        write_sam3d_job(rgb, depth_full, seg_pts, meta, job_path, crop_bbox=None)

        with open(job_path / "meta.json") as f:
            loaded = json.load(f)
        bbox = loaded["crop_bbox"]
        assert len(bbox) == 4
        assert bbox[0] <= 20 and bbox[2] >= 40
        assert bbox[1] <= 15 and bbox[3] >= 35


def test_write_sam3d_job_requires_seg_pts_or_bbox():
    """Raises when both seg_pts is empty and crop_bbox is missing."""
    with tempfile.TemporaryDirectory() as tmp:
        job_path = Path(tmp) / "job4"
        rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        depth_full = np.zeros((50, 50), dtype=np.uint16)
        seg_pts = np.empty((0, 2), dtype=np.int32)
        meta = {"job_id": "j", "track_id": 0, "label": "x", "frame_id": "odom",
                "stamp_sec": 0, "stamp_nanosec": 0,
                "camera_info": {"K": [], "width": 50, "height": 50},
                "odometry": {"position": [0, 0, 0], "orientation": [0, 0, 0, 1]}}
        with pytest.raises(ValueError, match="seg_pts or crop_bbox"):
            write_sam3d_job(rgb, depth_full, seg_pts, meta, job_path, crop_bbox=None)
