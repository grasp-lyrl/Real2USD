# Copyright 2025 Real2SAM3D contributors.
# Tests for SAM3D FAISS indexer: fixture output dir, index one job, assert index exists.
# Run: python3 -m pytest src_Real2USD/real2sam3d/test/test_sam3d_faiss_indexer.py -v
# Requires: real2usd in workspace, and (for full test) clip, faiss, torch.

import json
import tempfile
from pathlib import Path

import pytest

# Add scripts_sam3d_worker so we can import indexer helpers
_real2sam3d_root = Path(__file__).resolve().parents[1]
_worker_dir = _real2sam3d_root / "scripts_sam3d_worker"
import sys
if str(_worker_dir) not in sys.path:
    sys.path.insert(0, str(_worker_dir))

# Add real2usd/scripts_r2u for CLIPUSDSearch
_scripts_r2u = _real2sam3d_root.parent / "real2usd" / "scripts_r2u"
if _scripts_r2u.is_dir() and str(_scripts_r2u) not in sys.path:
    sys.path.insert(0, str(_scripts_r2u))


def _make_fixture_output_job(job_path: Path) -> str:
    """Create output/<job_id>/ with pose.json, object.ply, rgb.png. Returns object path."""
    job_path.mkdir(parents=True, exist_ok=True)
    with open(job_path / "pose.json", "w") as f:
        json.dump({"position": [0, 0, 0], "orientation": [0, 0, 0, 1], "track_id": 1}, f)
    (job_path / "object.ply").write_text("ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n")
    import numpy as np
    import cv2
    rgb = np.random.randint(0, 255, (32, 48, 3), dtype=np.uint8)
    cv2.imwrite(str(job_path / "rgb.png"), rgb)
    return str((job_path / "object.ply").resolve())


def test_indexer_helpers():
    """Test _completed_job_dirs, _image_paths_for_job, and _faiss_dir_and_paths without CLIP."""
    from index_sam3d_faiss import (
        _completed_job_dirs,
        _image_paths_for_job,
        _load_indexed_jobs,
        _faiss_dir_and_paths,
        _save_indexed_job,
    )

    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / "output"
        output_dir.mkdir()
        # Empty -> no jobs
        assert list(_completed_job_dirs(output_dir)) == []

        job_path = output_dir / "job1"
        _make_fixture_output_job(job_path)
        jobs = list(_completed_job_dirs(output_dir))
        assert len(jobs) == 1
        assert jobs[0][0].name == "job1"
        assert jobs[0][1].endswith("object.ply")

        imgs = list(_image_paths_for_job(job_path))
        assert len(imgs) == 1
        assert imgs[0].endswith("rgb.png")

        # faiss/ subdir and state file
        index_arg = Path(tmp) / "idx"
        faiss_dir, index_base, state_path = _faiss_dir_and_paths(index_arg)
        assert faiss_dir == index_arg / "faiss"
        assert faiss_dir.exists()
        assert state_path == faiss_dir / "indexed_jobs.txt"
        assert _load_indexed_jobs(state_path) == set()
        _save_indexed_job(state_path, "job1")
        assert _load_indexed_jobs(state_path) == {"job1"}


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1].parent / "real2usd" / "scripts_r2u" / "clipusdsearch_cls.py").exists(),
    reason="real2usd/scripts_r2u not in workspace",
)
def test_indexer_adds_one_job():
    """Run indexer on fixture output dir; assert index .faiss and .pkl exist and have one entry."""
    try:
        from clipusdsearch_cls import CLIPUSDSearch
    except ImportError:
        pytest.skip("CLIP/FAISS not installed")

    from index_sam3d_faiss import (
        index_pending_jobs,
        _load_indexed_jobs,
        _faiss_dir_and_paths,
    )

    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / "output"
        output_dir.mkdir()
        job_path = output_dir / "test_job_001"
        _make_fixture_output_job(job_path)

        index_arg = Path(tmp) / "sam3d_faiss"
        faiss_dir, index_base, state_path = _faiss_dir_and_paths(index_arg)
        indexed = _load_indexed_jobs(state_path)
        clip_search = CLIPUSDSearch()

        n = index_pending_jobs(output_dir, index_base, clip_search, indexed, state_path)
        assert n == 1
        assert (Path(str(index_base) + ".faiss")).exists()
        assert (Path(str(index_base) + ".pkl")).exists()
        assert (faiss_dir / "indexed_jobs.txt").exists()
        assert len(clip_search.image_paths) == 1
        assert len(clip_search.usd_paths) == 1
        assert clip_search.usd_paths[0].endswith("object.ply")


def test_render_sam3d_views_no_ply_returns_false():
    """render_job_dir returns False when there is no object.ply."""
    _worker_dir = Path(__file__).resolve().parents[1] / "scripts_sam3d_worker"
    if str(_worker_dir) not in sys.path:
        sys.path.insert(0, str(_worker_dir))
    try:
        from render_sam3d_views import render_job_dir
    except ImportError:
        pytest.skip("open3d not installed")
    with tempfile.TemporaryDirectory() as tmp:
        job_dir = Path(tmp) / "job1"
        job_dir.mkdir()
        # No object.ply
        ok = render_job_dir(job_dir, num_views=4, size=64)
        assert ok is False


def test_render_sam3d_views_skips_when_views_exist():
    """render_job_dir returns False when views/ already has pngs (skip re-render)."""
    _worker_dir = Path(__file__).resolve().parents[1] / "scripts_sam3d_worker"
    if str(_worker_dir) not in sys.path:
        sys.path.insert(0, str(_worker_dir))
    try:
        from render_sam3d_views import render_job_dir
    except ImportError:
        pytest.skip("open3d not installed")
    with tempfile.TemporaryDirectory() as tmp:
        job_dir = Path(tmp) / "job1"
        job_dir.mkdir()
        (job_dir / "object.ply").write_text("ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
        views_dir = job_dir / "views"
        views_dir.mkdir()
        (views_dir / "0.png").write_bytes(b"\x89PNG\r\n\x1a\n")  # placeholder so "views exist"
        ok = render_job_dir(job_dir, num_views=4, size=64)
        assert ok is False
