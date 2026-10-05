"""Appearance descriptors for track re-identification.

PHASE_SPECS's association cascade uses a CLIP-cosine term to heal detector track
breaks. CLIP isn't in the light uv env, so Phase 2 uses a real, ROS-free **masked-crop
HSV color histogram** behind the :class:`Appearance` protocol — swap in CLIP/SigLIP
later without touching the associator (it only calls ``embed`` + :func:`cosine`).
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import cv2
import numpy as np


@runtime_checkable
class Appearance(Protocol):
    dim: int

    def embed(self, rgb: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
        """L2-normalized descriptor of the masked region, or None if empty."""
        ...


def cosine(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    """Cosine similarity of two (already ~L2-normalized) descriptors; 0 if either
    is missing. Clamped to [0, 1] (histograms are non-negative)."""
    if a is None or b is None:
        return 0.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), 0.0, 1.0))


class HSVHistogram:
    """Masked HSV color histogram (H,S,V bins), L2-normalized.

    Hue dominates (indoor objects separate more by hue than brightness); value is
    coarse to stay lighting-tolerant. Deterministic.
    """

    def __init__(self, bins=(12, 4, 2)):
        self.bins = tuple(int(b) for b in bins)
        self.dim = int(np.prod(self.bins))

    def embed(self, rgb: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
        m = np.asarray(mask, bool)
        if not m.any():
            return None
        hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1, 2], m.astype(np.uint8),
            list(self.bins), [0, 180, 0, 256, 0, 256],
        ).astype(np.float64).reshape(-1)
        n = np.linalg.norm(hist)
        return hist / n if n > 0 else hist
