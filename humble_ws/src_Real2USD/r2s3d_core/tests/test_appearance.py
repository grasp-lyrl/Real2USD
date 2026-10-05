"""Appearance descriptor (HSV histogram) + cosine sanity."""

import numpy as np

from r2s3d_core.tracks.appearance import HSVHistogram, cosine


def _solid(color, h=40, w=40):
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[:] = color
    mask = np.ones((h, w), bool)
    return rgb, mask


def test_same_crop_cosine_is_one():
    app = HSVHistogram()
    rgb, mask = _solid((200, 30, 30))
    a = app.embed(rgb, mask)
    b = app.embed(rgb.copy(), mask.copy())
    assert a is not None and b is not None
    assert cosine(a, b) > 0.999


def test_different_colors_low_cosine():
    app = HSVHistogram()
    red, m = _solid((220, 20, 20))
    blue, _ = _solid((20, 20, 220))
    assert cosine(app.embed(red, m), app.embed(blue, m)) < 0.3


def test_empty_mask_returns_none():
    app = HSVHistogram()
    rgb, _ = _solid((10, 10, 10))
    assert app.embed(rgb, np.zeros((40, 40), bool)) is None
    assert cosine(None, np.ones(app.dim)) == 0.0
