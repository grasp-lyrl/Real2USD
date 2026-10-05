"""Tests for THOR asset-mesh fitting (download-free; USD reading is covered separately)."""

import numpy as np
import pytest

from r2s3d_core.data import thor_assets as TA


def _pose(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _yaw(deg):
    th = np.radians(deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def test_fit_places_canonical_mesh_into_obb():
    trimesh = pytest.importorskip("trimesh")
    # canonical asset: an axis-aligned box with distinct extents (so axis matching is
    # unambiguous), centered away from origin to exercise the centroid shift.
    ext = np.array([2.0, 1.0, 0.5])
    canonical = trimesh.creation.box(extents=ext,
                                     transform=_pose(np.eye(3), [3.0, -1.0, 2.0]))
    # target OBB: yawed 30deg, translated, same physical extents
    R, t = _yaw(30.0), np.array([5.0, 6.0, 1.0])
    fitted = TA.fit_canonical_to_obb(canonical, _pose(R, t), ext)

    # centroid lands at the OBB center
    np.testing.assert_allclose(fitted.vertices.mean(axis=0), t, atol=1e-6)
    # spans measured along the OBB axes match the declared extents (sorted, since the fit
    # matches axes by descending extent)
    local = (np.asarray(fitted.vertices) - t) @ R
    span = local.max(0) - local.min(0)
    np.testing.assert_allclose(np.sort(span), np.sort(ext), atol=1e-6)


def test_fit_preserves_shape_no_rescale():
    trimesh = pytest.importorskip("trimesh")
    ext = np.array([1.5, 1.0, 0.5])
    canonical = trimesh.creation.box(extents=ext)
    fitted = TA.fit_canonical_to_obb(canonical, _pose(_yaw(80.0), [1, 2, 3]), ext)
    # rigid transform => volume and edge lengths unchanged
    assert fitted.volume == pytest.approx(canonical.volume, rel=1e-6)


def test_default_asset_root_missing_returns_none(monkeypatch, tmp_path):
    monkeypatch.setenv("R2S3D_THOR_ASSETS", str(tmp_path / "nope"))
    assert TA.default_asset_root() is None
