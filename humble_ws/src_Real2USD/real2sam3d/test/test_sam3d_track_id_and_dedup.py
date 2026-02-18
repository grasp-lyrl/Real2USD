# Copyright 2025 Real2SAM3D contributors.
# Tests for track_id handling (segment_cls) and job writer dedup logic.
# Run inside the project Docker container (after colcon build and source install/setup.bash):
#   python3 -m pytest src_Real2USD/real2sam3d/test/test_sam3d_track_id_and_dedup.py -v

import sys
import tempfile
from pathlib import Path

# Import track_id_utils without pulling in ultralytics (segment_cls). Add scripts_r2s3d dir to path.
_real2sam3d_root = Path(__file__).resolve().parents[1]
_scripts_r2s3d = _real2sam3d_root / "scripts_r2s3d"
if str(_scripts_r2s3d) not in sys.path:
    sys.path.insert(0, str(_scripts_r2s3d))
from track_id_utils import track_ids_from_boxes_id

import pytest


class _MockBoxesId:
    """Minimal mock for result.boxes.id with .int().cpu().tolist()."""

    def __init__(self, values):
        self._values = values

    def int(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self._values


def test_track_ids_from_boxes_id_none():
    """When boxes.id is None, all ids are -1."""
    out = track_ids_from_boxes_id(None, 3)
    assert out == [-1, -1, -1]


def test_track_ids_from_boxes_id_list():
    """When boxes.id yields a list of ints, they are preserved."""
    mock = _MockBoxesId([1, 2, 3])
    out = track_ids_from_boxes_id(mock, 3)
    assert out == [1, 2, 3]


def test_track_ids_from_boxes_id_with_none_elements():
    """Per-element None is converted to -1."""
    mock = _MockBoxesId([1, None, 3])
    out = track_ids_from_boxes_id(mock, 3)
    assert out == [1, -1, 3]


def test_track_ids_from_boxes_id_short_list_padded():
    """If raw list is shorter than num_boxes, pad with -1."""
    mock = _MockBoxesId([10, 20])
    out = track_ids_from_boxes_id(mock, 4)
    assert out == [10, 20, -1, -1]


def test_track_ids_from_boxes_id_long_list_truncated():
    """If raw list is longer, truncate to num_boxes."""
    mock = _MockBoxesId([1, 2, 3, 4, 5])
    out = track_ids_from_boxes_id(mock, 3)
    assert out == [1, 2, 3]


def test_job_writer_dedup_track_id_and_position():
    """Dedup: same track_id within window skips; -1 skips track_id dedup; position+label dedup works."""
    from real2sam3d.sam3d_job_writer_node import Sam3dDedupState

    dedup = Sam3dDedupState(dedup_track_id_sec=60.0, dedup_position_m=0.5)

    # Nothing recorded yet -> should not skip
    assert dedup.should_skip(1, "Chair", (0.0, 0.0, 0.0)) is False

    # Record track_id=1, same label/position
    dedup.record(1, "Chair", (0.0, 0.0, 0.0))
    assert dedup.should_skip(1, "Chair", (0.0, 0.0, 0.0)) is True

    # Different track_id, same position -> not skipped by track_id; same label+position -> skipped by position
    assert dedup.should_skip(2, "Chair", (0.0, 0.0, 0.0)) is True

    # track_id=-1 (no id): no track_id dedup; but position+label still applies
    assert dedup.should_skip(-1, "Chair", (0.0, 0.0, 0.01)) is True

    # Same label, position > 0.5 m away -> not skipped
    assert dedup.should_skip(-1, "Chair", (1.0, 0.0, 0.0)) is False

    # track_id=-1 is not recorded in _track_id_last_write; position is recorded
    dedup.record(-1, "Table", (2.0, 2.0, 0.0))
    assert 1 in dedup._track_id_last_write
    assert -1 not in dedup._track_id_last_write
    assert dedup.should_skip(-1, "Table", (2.0, 2.0, 0.1)) is True


def test_job_writer_dedup_disabled():
    """When dedup_track_id_sec=0 and dedup_position_m=0, never skip."""
    from real2sam3d.sam3d_job_writer_node import Sam3dDedupState

    dedup = Sam3dDedupState(dedup_track_id_sec=0.0, dedup_position_m=0.0)
    dedup.record(1, "Chair", (0.0, 0.0, 0.0))
    assert dedup.should_skip(1, "Chair", (0.0, 0.0, 0.0)) is False
