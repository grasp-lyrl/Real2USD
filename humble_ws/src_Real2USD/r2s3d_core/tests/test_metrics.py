"""Golden-value tests for the evaluation metrics.

Hand-constructed box pairs with known IoU / rotation / scale answers, plus a small
synthetic scene exercising matching, duplicate rate, and Scan2CAD accuracy.
"""

import numpy as np
import pytest

from r2s3d_core.eval import geometry as geo
from r2s3d_core.eval import metrics as M


def _box(center, extents, yaw_deg=0.0):
    th = np.radians(yaw_deg)
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    return geo.Box(center=np.asarray(center, float), R=R, extents=np.asarray(extents, float))


def _pose(center, yaw_deg=0.0):
    th = np.radians(yaw_deg)
    c, s = np.cos(th), np.sin(th)
    T = np.eye(4)
    T[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    T[:3, 3] = center
    return T


# ------------------------------------------------------------------- OBB IoU

def test_iou_identical():
    b = _box([0, 0, 0], [1, 1, 1])
    assert geo.obb_iou(b, b) == pytest.approx(1.0, abs=1e-6)


def test_iou_half_offset_x():
    # unit cubes offset 0.5 in x: intersection 0.5, union 1.5 -> 1/3
    a = _box([0, 0, 0], [1, 1, 1])
    b = _box([0.5, 0, 0], [1, 1, 1])
    assert geo.obb_iou(a, b) == pytest.approx(1.0 / 3.0, abs=1e-4)


def test_iou_disjoint():
    a = _box([0, 0, 0], [1, 1, 1])
    b = _box([5, 0, 0], [1, 1, 1])
    assert geo.obb_iou(a, b) == pytest.approx(0.0, abs=1e-9)


def test_iou_rotated_45_same_center():
    # a unit cube and a 45deg-yaw unit cube, same center: known intersection area
    # in-plane = 2*(sqrt2 - 1) ~= 0.8284; height 1 -> inter vol; union = 2 - inter.
    a = _box([0, 0, 0], [1, 1, 1])
    b = _box([0, 0, 0], [1, 1, 1], yaw_deg=45)
    inter = 2.0 * (np.sqrt(2.0) - 1.0)
    expect = inter / (2.0 - inter)
    assert geo.obb_iou(a, b) == pytest.approx(expect, abs=2e-3)


# ------------------------------------------------------------ rotation error

def test_rotation_geodesic_30():
    R0 = np.eye(3)
    R30 = _box([0, 0, 0], [1, 1, 1], yaw_deg=30).R
    assert geo.rotation_geodesic_deg(R0, R30) == pytest.approx(30.0, abs=1e-6)


def test_rotation_symmetry_c2_180():
    R0 = np.eye(3)
    R180 = _box([0, 0, 0], [1, 1, 1], yaw_deg=180).R
    # no symmetry: 180 deg error
    assert geo.rotation_error_deg(R0, R180, "none") == pytest.approx(180.0, abs=1e-4)
    # c2 symmetry (rectangular table): 0 deg
    assert geo.rotation_error_deg(R0, R180, "c2") == pytest.approx(0.0, abs=1e-4)


def test_rotation_symmetry_inf():
    R0 = np.eye(3)
    R37 = _box([0, 0, 0], [1, 1, 1], yaw_deg=37).R
    # yaw-invariant object: any yaw is ~free
    assert geo.rotation_error_deg(R0, R37, "inf") == pytest.approx(0.0, abs=2.0)


# --------------------------------------------------------------- scale error

def test_scale_ratio_error():
    err = geo.scale_ratio_error([2, 1, 1], [1, 1, 1])
    np.testing.assert_allclose(err, [1.0, 0.0, 0.0], atol=1e-9)


# ----------------------------------------------------------- Chamfer/F-score

def test_chamfer_identical_mesh():
    trimesh = pytest.importorskip("trimesh")
    m = trimesh.creation.box(extents=[1, 1, 1])
    p = geo.sample_surface(m, 20000, seed=1)
    q = geo.sample_surface(m, 20000, seed=2)
    out = geo.chamfer_and_fscore(p, q, taus=(0.05, 0.02))
    # identical surface: chamfer ~ inter-sample spacing (~1.7cm at this density)
    assert out["chamfer_l1"] < 0.03
    assert out["fscore@0.05"] > 0.99
    assert out["fscore@0.02"] > 0.7


# ------------------------------------------------------------------ matching

def test_hungarian_and_scene_metrics():
    gts = [
        M.SceneObject("chair", _pose([0, 0, 0]), np.array([1.0, 1, 1])),
        M.SceneObject("table", _pose([5, 0, 0]), np.array([1.0, 1, 1])),
    ]
    preds = [
        M.SceneObject("chair", _pose([0.05, 0, 0]), np.array([1.0, 1, 1])),  # matches gt0
        M.SceneObject("chair", _pose([0.1, 0, 0]), np.array([1.0, 1, 1])),   # duplicate of gt0
    ]
    m = M.evaluate(preds, gts, iou_threshold=0.25, compute_geometry=False)
    assert m["tp"] == 1
    assert m["fn"] == 1
    assert m["fp"] == 1
    assert m["count_ratio"] == pytest.approx(1.0)
    # both preds land on gt0 -> one duplicate over 2 GTs
    assert m["duplicate_rate"] == pytest.approx(0.5)
    # the matched pair is within 20cm/20deg/20% -> 1 of 2 GT
    assert m["scan2cad_accuracy"] == pytest.approx(0.5)


def test_scan2cad_fails_on_large_error():
    gts = [M.SceneObject("box", _pose([0, 0, 0]), np.array([1.0, 1, 1]))]
    preds = [M.SceneObject("box", _pose([0, 0, 0]), np.array([1.5, 1, 1]))]  # 50% scale err
    m = M.evaluate(preds, gts, iou_threshold=0.25, compute_geometry=False)
    assert m["tp"] == 1
    assert m["scan2cad_accuracy"] == pytest.approx(0.0)
