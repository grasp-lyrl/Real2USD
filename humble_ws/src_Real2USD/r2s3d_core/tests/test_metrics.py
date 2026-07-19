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


def test_box_pose_error_relabel_invariant():
    # same physical box with axes relabeled by a cube rotation -> ~0 rot & scale error
    eg = np.array([2.0, 1.0, 0.5])
    g = np.array([[0, 0, 1.0], [1, 0, 0], [0, 1, 0]])  # cyclic perm, det +1
    Rp = np.eye(3) @ g
    ep = np.abs(g.T @ eg)
    rot, serr = geo.box_pose_error(Rp, ep, np.eye(3), eg, "none")
    assert rot < 1e-6
    assert serr.max() < 1e-6


def test_box_pose_error_wrong_aspect_penalized_by_scale():
    # 90deg-rotated elongated box: rotation resolves small, but scale error stays large
    Rg, eg = np.eye(3), np.array([2.0, 1.0, 0.5])
    th = np.radians(90)
    Rp = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1.0]])
    ep = np.array([2.0, 1.0, 0.5])  # labels NOT swapped -> long axis now along world y
    rot, serr = geo.box_pose_error(Rp, ep, Rg, eg, "none")
    assert serr.max() > 0.5  # mismatch shows up as scale error


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


# ------------------------------------------------ coworker-comparable additions

def test_named_counts():
    gts = [M.SceneObject("chair", _pose([0, 0, 0]), np.array([1.0, 1, 1])),
           M.SceneObject("table", _pose([5, 0, 0]), np.array([1.0, 1, 1]))]
    preds = [M.SceneObject("chair", _pose([0.05, 0, 0]), np.array([1.0, 1, 1]))]
    m = M.evaluate(preds, gts, iou_threshold=0.25, compute_geometry=False)
    assert m["objects_per_scene"] == 2
    assert m["predictions_per_scene"] == 1
    assert m["matched_per_scene"] == 1


def test_micro_macro_f1_all_correct():
    gts = [M.SceneObject("chair", _pose([0, 0, 0]), np.array([1.0, 1, 1])),
           M.SceneObject("table", _pose([5, 0, 0]), np.array([1.0, 1, 1]))]
    preds = [M.SceneObject("chair", _pose([0.05, 0, 0]), np.array([1.0, 1, 1])),
             M.SceneObject("table", _pose([5.05, 0, 0]), np.array([1.0, 1, 1]))]
    m = M.evaluate(preds, gts, iou_threshold=0.25, compute_geometry=False)
    assert m["micro_f1"] == pytest.approx(1.0)
    assert m["macro_f1"] == pytest.approx(1.0)
    assert set(m["per_class"]) == {"chair", "table"}


def test_micro_macro_f1_mislabel_penalized():
    # geometry is perfect (label-agnostic f1 == 1) but one label is wrong, so the
    # label-aware micro/macro F1 must drop below 1.
    gts = [M.SceneObject("chair", _pose([0, 0, 0]), np.array([1.0, 1, 1])),
           M.SceneObject("table", _pose([5, 0, 0]), np.array([1.0, 1, 1]))]
    preds = [M.SceneObject("chair", _pose([0.05, 0, 0]), np.array([1.0, 1, 1])),
             M.SceneObject("sofa", _pose([5.05, 0, 0]), np.array([1.0, 1, 1]))]  # wrong label
    m = M.evaluate(preds, gts, iou_threshold=0.25, compute_geometry=False)
    assert m["f1"] == pytest.approx(1.0)          # geometry-only unaffected
    assert m["label_aware_matched"] == 1          # only the chair matches label-aware
    assert m["micro_f1"] == pytest.approx(0.5)    # tp=1, n_pred=n_gt=2
    # classes: chair (F1 1), table (missed, 0), sofa (hallucinated, 0) -> mean 1/3
    assert m["macro_f1"] == pytest.approx(1.0 / 3.0)


def test_chamfer_mean_is_half_l1():
    trimesh = pytest.importorskip("trimesh")
    m = trimesh.creation.box(extents=[1, 1, 1])
    p = geo.sample_surface(m, 20000, seed=1)
    q = geo.sample_surface(m, 20000, seed=2)
    out = geo.chamfer_and_fscore(p, q, taus=(0.05, 0.02))
    assert out["chamfer_mean"] == pytest.approx(0.5 * out["chamfer_l1"])
    # recall@tau (class-free geo recall) present and sane on identical surfaces
    assert out["recall@0.05"] > 0.99


def test_scene_geometry_keys_present_with_meshes():
    trimesh = pytest.importorskip("trimesh")
    box = trimesh.creation.box(extents=[1, 1, 1])
    gts = [M.SceneObject("box", _pose([0, 0, 0]), np.array([1.0, 1, 1]), mesh=box)]
    preds = [M.SceneObject("box", _pose([0, 0, 0]), np.array([1.0, 1, 1]), mesh=box)]
    m = M.evaluate(preds, gts, iou_threshold=0.25, compute_geometry=True, surface_points=4000)
    assert "scene_chamfer_mean_m" in m
    assert m["surf_recall@0.05"] > 0.99   # surface coverage (renamed from geo_recall)
    assert np.isfinite(m["chamfer_symmetric_mean_m"])


def test_centroid_matching_metrics():
    # coworker protocol: Hungarian on centroid distance <= tau (default 1 m), not IoU.
    gts = [M.SceneObject("chair", _pose([0, 0, 0]), np.array([1.0, 1, 1])),
           M.SceneObject("table", _pose([5, 0, 0]), np.array([1.0, 1, 1]))]
    preds = [M.SceneObject("chair", _pose([0.3, 0, 0]), np.array([1.0, 1, 1])),  # 0.3 m, right label
             M.SceneObject("sofa", _pose([5.2, 0, 0]), np.array([1.0, 1, 1]))]   # 0.2 m, WRONG label
    m = M.evaluate(preds, gts, compute_geometry=False)
    assert m["cd_tau"] == 1.0
    assert m["cd_f1"] == pytest.approx(1.0)             # both within 1 m (label-agnostic)
    assert m["cd_recall"] == pytest.approx(1.0)
    assert m["class_free_recall_1m"] == pytest.approx(1.0)
    assert m["cd_micro_f1"] == pytest.approx(0.5)       # only the chair is label-correct
    # tau sweep: at 0.25 m only the 0.2 m (sofa/table) pair survives -> recall 0.5
    assert m["centroid_recall_by_tau"]["0.25"] == pytest.approx(0.5)
    assert m["centroid_recall_by_tau"]["1.0"] == pytest.approx(1.0)


def test_many_to_one_exceeds_micro_on_oversegmentation():
    # one GT chair, two predicted chairs near it: greedy micro caps at one match, but the
    # any-overlap many-to-one metric credits both -> many_to_one_f1 > micro_f1.
    gts = [M.SceneObject("chair", _pose([0, 0, 0]), np.array([1.0, 1, 1]))]
    preds = [M.SceneObject("chair", _pose([0.1, 0, 0]), np.array([1.0, 1, 1])),
             M.SceneObject("chair", _pose([0.2, 0, 0]), np.array([1.0, 1, 1]))]
    m = M.evaluate(preds, gts, compute_geometry=False)
    assert m["cd_micro_f1"] == pytest.approx(2 / 3)          # tp=1, prec=1/2, rec=1
    assert m["cd_micro_f1_many_to_one"] == pytest.approx(1.0)  # both preds credited
    assert m["cd_micro_f1_many_to_one"] > m["cd_micro_f1"]


def test_centroid_matching_is_2d_topdown():
    # pred stacked 2 m directly ABOVE the GT (same X,Y): 3D distance 2 m would miss, but the
    # 2D top-down distance is 0 -> matched.
    gts = [M.SceneObject("book", _pose([0, 0, 0]), np.array([0.3, 0.3, 0.1]))]
    preds = [M.SceneObject("book", _pose([0, 0, 2.0]), np.array([0.3, 0.3, 0.1]))]
    m = M.evaluate(preds, gts, compute_geometry=False)
    assert m["cd_recall"] == pytest.approx(1.0)
    assert m["class_free_recall_1m"] == pytest.approx(1.0)


def test_centroid_matching_far_objects_unmatched():
    # a prediction 3 m from the only GT: outside every tau up to 1.5 -> no match
    gts = [M.SceneObject("box", _pose([0, 0, 0]), np.array([1.0, 1, 1]))]
    preds = [M.SceneObject("box", _pose([3.0, 0, 0]), np.array([1.0, 1, 1]))]
    m = M.evaluate(preds, gts, compute_geometry=False)
    assert m["cd_f1"] == pytest.approx(0.0)
    assert m["class_free_recall_1m"] == pytest.approx(0.0)


def test_scene_geometry_absent_without_meshes():
    gts = [M.SceneObject("box", _pose([0, 0, 0]), np.array([1.0, 1, 1]))]
    preds = [M.SceneObject("box", _pose([0, 0, 0]), np.array([1.0, 1, 1]))]
    m = M.evaluate(preds, gts, iou_threshold=0.25, compute_geometry=True)
    assert "scene_chamfer_mean_m" not in m  # no meshes -> class-free geometry skipped


# ------------------------------------------------------------------- footprint IoU

def test_footprint_iou_identical_clouds_is_one():
    # same points -> same occupied cells -> IoU 1
    rng = np.random.RandomState(0)
    pts = rng.uniform(-1, 1, size=(500, 3))
    assert geo.footprint_iou(pts, pts, cell=0.05) == pytest.approx(1.0)


def test_footprint_iou_z_invariant():
    # shifting one cloud purely in Z must not change the top-down footprint
    rng = np.random.RandomState(1)
    pts = rng.uniform(-1, 1, size=(500, 3))
    shifted = pts.copy()
    shifted[:, 2] += 3.0
    assert geo.footprint_iou(pts, shifted, cell=0.05) == pytest.approx(1.0)


def test_footprint_iou_disjoint_is_zero():
    # two clouds in far-apart XY regions share no cells -> IoU 0
    rng = np.random.RandomState(2)
    a = rng.uniform(0, 1, size=(300, 3))
    b = rng.uniform(100, 101, size=(300, 3))
    assert geo.footprint_iou(a, b, cell=0.05) == pytest.approx(0.0)


def test_footprint_iou_half_overlap():
    # two 10x10 blocks of cells (points at cell centres to avoid boundary fp jitter),
    # offset by 5 cells in x: x-cells {0..9} vs {5..14} -> intersection 5, union 15,
    # y identical -> IoU = 50/150 = 1/3.
    g = 0.1
    centres = (np.arange(10) + 0.5) * g          # 0.05, 0.15, ..., 0.95
    grid_a = np.array([[x, y, 0.0] for x in centres for y in centres])
    grid_b = grid_a + np.array([5 * g, 0.0, 0.0])
    assert geo.footprint_iou(grid_a, grid_b, cell=g) == pytest.approx(1 / 3)


def test_footprint_iou_empty_is_nan():
    pts = np.zeros((0, 3))
    assert np.isnan(geo.footprint_iou(pts, np.ones((3, 3)), cell=0.05))


def test_footprint_iou_in_evaluate_with_meshes():
    trimesh = pytest.importorskip("trimesh")
    box = trimesh.creation.box(extents=[1, 1, 1])
    gts = [M.SceneObject("box", _pose([0, 0, 0]), np.array([1.0, 1, 1]), mesh=box)]
    preds = [M.SceneObject("box", _pose([0, 0, 0]), np.array([1.0, 1, 1]), mesh=box)]
    m = M.evaluate(preds, gts, iou_threshold=0.25, compute_geometry=True, surface_points=4000)
    assert "footprint_iou" in m
    assert m["footprint_iou"] > 0.9  # coincident boxes -> near-full footprint overlap


def test_scene_geometry_scores_cluster_surface_pts():
    # cluster payload (no mesh) is scored from its attached surface points -- a coincident
    # GT mesh and pred point cloud must produce a finite Chamfer + high footprint IoU.
    trimesh = pytest.importorskip("trimesh")
    box = trimesh.creation.box(extents=[1, 1, 1])
    cloud, _ = trimesh.sample.sample_surface(box, 3000)
    gts = [M.SceneObject("box", _pose([0, 0, 0]), np.array([1.0, 1, 1]), mesh=box)]
    preds = [M.SceneObject("box", _pose([0, 0, 0]), np.array([1.0, 1, 1]),
                           surface_pts=np.asarray(cloud))]
    m = M.evaluate(preds, gts, iou_threshold=0.25, compute_geometry=True, surface_points=3000)
    assert np.isfinite(m["scene_chamfer_mean_m"])
    # two independent surface samplings of the same box disagree only on a few boundary
    # cells -> high but not perfect footprint overlap; the point is the cluster path scores.
    assert m["footprint_iou"] > 0.8
