"""Seeded, optional corruptions applied to a cached :class:`DetectionSet`.

The detector already introduces natural imperfection (missed objects, sloppy masks,
track breaks). These corruptions are a *stress amplifier* on top — most importantly
``split``, which fractures one object's mask into several detections, exercising the
exact case ObjectTrack association + late-merge must clean up (one real object should
collapse back to one). Everything is deterministic given ``seed``.

All operations take and return a :class:`DetectionSet` (new object; input untouched).
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np

from .cache import Detection, DetectionSet


def _bbox_from_mask(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.array([0, 0, 0, 0], np.float64)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], np.float64)


def apply(ds: DetectionSet, *, dropout: float = 0.0, jitter_px: int = 0,
          track_break: float = 0.0, split: float = 0.0, seed: int = 0) -> DetectionSet:
    """Return a corrupted copy of ``ds``.

    Parameters
    ----------
    dropout : probability a detection is dropped entirely.
    jitter_px : max magnitude of a random bbox/mask translation (pixels).
    track_break : probability a detection's ``track_id`` is reset to -1 (forces the
        associator onto its re-ID path instead of the id shortcut).
    split : probability a detection is split into two along its major axis (one
        object -> two detections; each half gets a fresh track id).
    """
    rng = np.random.RandomState(seed)
    H, W = ds.height, ds.width
    out = []
    next_new_id = 10_000_000  # fresh ids for split halves / breaks, disjoint from real
    for det in ds.detections:
        if dropout > 0 and rng.rand() < dropout:
            continue
        d = copy.deepcopy(det)
        if jitter_px > 0:
            dx = int(rng.randint(-jitter_px, jitter_px + 1))
            dy = int(rng.randint(-jitter_px, jitter_px + 1))
            d.mask = np.roll(np.roll(d.mask, dy, axis=0), dx, axis=1)
            d.bbox = _bbox_from_mask(d.mask)
        if track_break > 0 and rng.rand() < track_break:
            d.track_id = -1
        if split > 0 and rng.rand() < split and int(d.mask.sum()) > 40:
            halves = _split_mask(d.mask, rng)
            if halves is not None:
                for hm in halves:
                    if hm.sum() < 10:
                        continue
                    dd = copy.deepcopy(d)
                    dd.mask = hm
                    dd.bbox = _bbox_from_mask(hm)
                    dd.track_id = next_new_id
                    next_new_id += 1
                    out.append(dd)
                continue
        out.append(d)
    meta = dict(ds.meta)
    meta["corruptions"] = {"dropout": dropout, "jitter_px": jitter_px,
                           "track_break": track_break, "split": split, "seed": seed}
    return DetectionSet(scene=ds.scene, height=H, width=W, detections=out, meta=meta)


def _split_mask(mask: np.ndarray, rng: np.random.RandomState) -> Optional[list]:
    """Split a mask in two along its dominant pixel-spread axis."""
    ys, xs = np.where(mask)
    if len(xs) < 2:
        return None
    if xs.ptp() >= ys.ptp():  # split on x
        mid = int(np.median(xs))
        a = mask.copy(); a[:, mid:] = False
        b = mask.copy(); b[:, :mid] = False
    else:                      # split on y
        mid = int(np.median(ys))
        a = mask.copy(); a[mid:, :] = False
        b = mask.copy(); b[:mid, :] = False
    return [a, b]
