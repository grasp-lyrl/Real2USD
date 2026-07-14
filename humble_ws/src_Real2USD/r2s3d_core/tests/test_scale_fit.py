"""Depth-extent scale-fit: rescales a mesh so its OBB extents match a target span."""

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from scipy.spatial.transform import Rotation

from r2s3d_core.baselines.sam3d_layout import _fit_scale_to_extent, _observed_obb_extent


def test_scale_fit_matches_target_extent():
    m = trimesh.creation.box(extents=[0.4, 0.8, 1.2])
    m.apply_transform(np.block([[Rotation.from_euler("z", 25, degrees=True).as_matrix(),
                                 np.array([[1.0], [-2.0], [0.5]])], [np.zeros(3), 1.0]]))
    target = np.array([1.5, 1.0, 0.5])  # desired sorted extents
    M, info = _fit_scale_to_extent(m, target)
    m.apply_transform(M)
    got = np.sort(np.asarray(m.bounding_box_oriented.primitive.extents))[::-1]
    np.testing.assert_allclose(got, np.sort(target)[::-1], rtol=0.02)


def test_observed_obb_extent_of_box_cloud():
    pts = trimesh.creation.box(extents=[0.3, 0.6, 0.9]).sample(4000)
    e = _observed_obb_extent(pts)
    np.testing.assert_allclose(np.sort(e)[::-1], [0.9, 0.6, 0.3], rtol=0.05)


def test_observed_obb_extent_degenerate_returns_none():
    assert _observed_obb_extent(np.zeros((5, 3))) is None
