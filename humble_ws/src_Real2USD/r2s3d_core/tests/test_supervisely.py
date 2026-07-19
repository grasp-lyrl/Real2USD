"""Supervisely 3D-cuboid GT loader (data/supervisely.py)."""

import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from r2s3d_core.data.supervisely import (
    CANONICAL_LABELS,
    load_supervisely_gt,
    normalize_label,
)
from r2s3d_core.data.rosbag import _supervisely_dir


def test_label_normalization_matches_v1_aliases():
    assert normalize_label("high chair") == "chair"
    assert normalize_label("lounge chair") == "chair"
    assert normalize_label("round table") == "table"
    assert normalize_label("tall table") == "table"
    assert normalize_label("Door") == "door"
    assert normalize_label("couch") == "chair"      # exact alias
    assert normalize_label("office desk") == "table"  # contains-rule
    assert normalize_label("outlet") == "outlet"    # unresolved -> raw lower
    assert normalize_label("") == "__unlabeled__"


def test_parse_geometry_pose_and_extents(tmp_path):
    yaw = 0.7
    doc = {
        "objects": [{"id": 42, "classId": 1, "classTitle": "high chair"}],
        "figures": [{
            "objectId": 42,
            "geometry": {
                "position": {"x": 1.0, "y": 2.0, "z": 0.5},
                "rotation": {"x": 0.0, "y": 0.0, "z": yaw},
                "dimensions": {"x": 0.6, "y": 0.5, "z": 1.1},
            },
        }],
    }
    p = tmp_path / "scene.pcd.json"
    p.write_text(json.dumps(doc))
    gts = load_supervisely_gt(p)
    assert len(gts) == 1
    g = gts[0]
    assert g.label == "chair"                         # normalized from "high chair"
    np.testing.assert_allclose(g.T_world_obj[:3, 3], [1.0, 2.0, 0.5])
    np.testing.assert_allclose(g.extents, [0.6, 0.5, 1.1])   # full extents, not halved
    # rotation is radians "xyz" euler (pure yaw here)
    np.testing.assert_allclose(g.T_world_obj[:3, :3],
                               Rotation.from_euler("xyz", [0, 0, yaw]).as_matrix(), atol=1e-9)


def test_canonical_only_filters_non_eval_classes(tmp_path):
    doc = {
        "objects": [
            {"id": 1, "classId": 1, "classTitle": "little chair"},
            {"id": 2, "classId": 2, "classTitle": "outlet"},
        ],
        "figures": [
            {"objectId": 1, "geometry": {"position": {"x": 0, "y": 0, "z": 0},
                "rotation": {"x": 0, "y": 0, "z": 0}, "dimensions": {"x": 0.5, "y": 0.5, "z": 0.9}}},
            {"objectId": 2, "geometry": {"position": {"x": 1, "y": 0, "z": 0},
                "rotation": {"x": 0, "y": 0, "z": 0}, "dimensions": {"x": 0.1, "y": 0.1, "z": 0.1}}},
        ],
    }
    p = tmp_path / "s.pcd.json"
    p.write_text(json.dumps(doc))
    assert len(load_supervisely_gt(p)) == 2
    canon = load_supervisely_gt(p, canonical_only=True)
    assert [g.label for g in canon] == ["chair"]
    assert all(g.label in CANONICAL_LABELS for g in canon)


def test_real_lounge_gt_if_present():
    gt = _supervisely_dir() / "lounge-0_voxel_pointcloud.pcd.json"
    if not gt.is_file():
        pytest.skip("lounge-0 GT not on disk")
    gts = load_supervisely_gt(gt)
    assert len(gts) == 39                             # figure count
    # absolute-odom Z-up: centers in the tens of meters, small positive Z
    c = np.array([g.T_world_obj[:3, 3] for g in gts])
    assert c[:, 0].min() > 20 and c[:, 1].max() < 0
    assert 0.0 < np.median(c[:, 2]) < 1.5
    assert {"chair", "table", "door"} & {g.label for g in gts}
