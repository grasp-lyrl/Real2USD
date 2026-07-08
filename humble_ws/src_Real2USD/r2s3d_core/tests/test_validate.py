"""Phase 1: boundary validation checks."""

import numpy as np
import pytest

from r2s3d_core.frames import validate as V


def test_scale_ok():
    assert V.check_sam3d_scale([1.0, 2.0, 0.5]) is None


def test_scale_out_of_range_raises():
    with pytest.raises(V.ValidationError):
        V.check_sam3d_scale([0.0, 1.0, 1.0])          # below min
    with pytest.raises(V.ValidationError):
        V.check_sam3d_scale([1.0, 1.0, 100.0])        # above max


def test_scale_nonfinite_raises():
    with pytest.raises(V.ValidationError):
        V.check_sam3d_scale([np.nan, 1.0, 1.0])


def test_scale_no_raise_returns_reason():
    reason = V.check_sam3d_scale([50.0, 1.0, 1.0], raise_on_fail=False)
    assert reason and "outside" in reason


def test_translation_nonfinite():
    assert V.check_translation([0, 0, 1.0]) is None
    with pytest.raises(V.ValidationError):
        V.check_translation([0.0, np.inf, 1.0])


def test_depth_holes():
    good = np.full((10, 10), 2.0, np.float32)
    assert V.check_depth(good) is None
    holey = np.zeros((10, 10), np.float32)  # all invalid
    assert V.check_depth(holey) is not None
    with pytest.raises(V.ValidationError):
        V.check_depth(holey, raise_on_fail=True)
