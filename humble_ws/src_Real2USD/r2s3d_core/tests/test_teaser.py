"""TEASER++ Sim(3) registration: recovers a known scale+rotation+translation.

Gated on the teaserpp_python + open3d build (see docs/DATASETS.md build notes); skipped
where they aren't installed so the core suite stays light.
"""

import numpy as np
import pytest

teaserpp = pytest.importorskip("teaserpp_python")
o3d = pytest.importorskip("open3d")
trimesh = pytest.importorskip("trimesh")

from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from r2s3d_core.registration.teaser import register_teaser


def _asymmetric_cloud(n=3000, seed=0):
    # L-shaped (box + offset box) so rotation is well-determined (not symmetric)
    m = trimesh.util.concatenate([
        trimesh.creation.box(extents=[0.5, 0.5, 0.1]),
        trimesh.creation.box(extents=[0.5, 0.1, 0.5]).apply_translation([0, -0.2, 0.25]),
    ])
    return m.sample(n)


def test_teaser_recovers_similarity():
    rng = np.random.default_rng(0)
    src = _asymmetric_cloud()
    s_true = 0.7
    R_true = Rotation.from_euler("xyz", [10, 40, 5], degrees=True).as_matrix()
    t_true = np.array([1.2, -0.5, 0.3])
    dst = (s_true * (R_true @ src.T)).T + t_true + rng.normal(0, 0.003, src.shape)

    T, info = register_teaser(src, dst, voxel=0.03)
    assert info["ok"], info
    # scale recovered within 8%
    assert abs(info["scale"] - s_true) / s_true < 0.08, info["scale"]
    # applying T aligns source onto target (median NN residual small)
    src_reg = (T[:3, :3] @ src.T).T + T[:3, 3]
    resid = np.median(cKDTree(dst).query(src_reg)[0])
    assert resid < 0.03, resid


def test_teaser_loud_fallback_on_no_correspondences():
    # two unrelated tiny clouds -> too few correspondences -> identity + ok=False, not a crash
    a = np.random.default_rng(1).normal(size=(4, 3))
    T, info = register_teaser(a, a + 100.0, voxel=0.05)
    assert T.shape == (4, 4)
    assert info["ok"] is False
