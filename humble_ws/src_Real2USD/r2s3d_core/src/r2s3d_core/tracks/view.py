"""View scoring + a diverse top-K best-view buffer.

PHASE_SPECS §Phase 2:
    view_score = mask_area_norm · (1 - edge_contact) · sharpness_norm ·
                 (1 + 0.5·angle_novelty); keep top-6.

The base score (area · non-edge · sharpness) is intrinsic to an observation; the
``angle_novelty`` factor is relative to the views already kept, so a new view that
looks from a fresh angle can displace a higher-base-score but redundant one. This is
the multi-view "best-view for generation" lever (see [[phase2-multiview-rationale]]).
"""

from __future__ import annotations

import cv2
import numpy as np

from .types import MAX_KEPT_VIEWS, Observation

# Reference scales for normalization (relative ranking within a track; not absolute).
_AREA_REF = 0.10          # mask area fraction that saturates area_norm
_SHARP_REF = 500.0        # variance-of-Laplacian that saturates sharpness_norm
_EDGE_BAND_PX = 6         # border band for edge-contact (matches v1 filter)


def laplacian_sharpness(rgb_crop: np.ndarray) -> float:
    """Variance of the Laplacian (blur metric; higher = sharper)."""
    if rgb_crop is None or rgb_crop.size == 0:
        return 0.0
    g = cv2.cvtColor(np.ascontiguousarray(rgb_crop), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def edge_contact(mask: np.ndarray, band: int = _EDGE_BAND_PX) -> float:
    """Fraction of mask pixels within ``band`` px of any image border."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 1.0
    H, W = mask.shape
    on = (xs < band) | (xs >= W - band) | (ys < band) | (ys >= H - band)
    return float(np.count_nonzero(on)) / len(xs)


def base_view_score(mask: np.ndarray, rgb_crop: np.ndarray) -> float:
    """area_norm · (1 - edge_contact) · sharpness_norm (angle novelty added at
    insertion time, since it depends on the kept set)."""
    H, W = mask.shape
    area_frac = float(mask.sum()) / (H * W)
    area_norm = min(1.0, area_frac / _AREA_REF)
    ec = edge_contact(mask)
    sharp_norm = min(1.0, laplacian_sharpness(rgb_crop) / _SHARP_REF)
    return area_norm * (1.0 - ec) * sharp_norm


def _angle_novelty(view_dir: np.ndarray, kept: list) -> float:
    """0..1: normalized min angular distance from ``view_dir`` to kept view dirs."""
    if not kept:
        return 1.0
    v = view_dir / (np.linalg.norm(view_dir) + 1e-9)
    min_ang = np.pi
    for o in kept:
        k = o.view_dir_world / (np.linalg.norm(o.view_dir_world) + 1e-9)
        ang = np.arccos(np.clip(np.dot(v, k), -1.0, 1.0))
        min_ang = min(min_ang, ang)
    return float(min_ang / np.pi)


def full_view_score(obs: Observation, kept: list) -> float:
    return obs.view_score * (1.0 + 0.5 * _angle_novelty(obs.view_dir_world, kept))


def insert_kept_view(kept: list, obs: Observation, k: int = MAX_KEPT_VIEWS) -> None:
    """Insert ``obs`` into the diverse top-K buffer ``kept`` (in place).

    Keeps the K observations maximizing base_score·(1 + 0.5·novelty). Evicted
    observations have their mask/rgb_crop nulled to bound memory (the masked depth
    was already fused into the track cloud at ingest).
    """
    kept.append(obs)
    if len(kept) <= k:
        return
    # recompute full (novelty-aware) score for each kept view against the others
    def score_of(i):
        others = kept[:i] + kept[i + 1:]
        return full_view_score(kept[i], others)

    scores = [score_of(i) for i in range(len(kept))]
    drop = int(np.argmin(scores))
    victim = kept.pop(drop)
    victim.mask = None
    victim.rgb_crop = None
