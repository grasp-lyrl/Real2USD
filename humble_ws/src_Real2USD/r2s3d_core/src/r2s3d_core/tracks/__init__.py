"""ObjectTrack: persistent, multi-view, per-object entities (Phase 2).

The core abstraction of the v2 rework — accumulates multi-view evidence (fused cloud,
label votes, appearance, best views) so a single-image generator (SAM3D) becomes a
scene mapper, and so detector fragmentation (one object seen as several detections /
track ids) collapses back to one object via association + late merge.

See docs/PHASE_SPECS.md §Phase 2 and docs/REWORK_PLAN.md §2.2.
"""

from __future__ import annotations

from .appearance import Appearance, HSVHistogram, cosine
from .fusion import VoxelCloud
from .tracker import GATE_MIN_OBS, GATE_MIN_SCORE, run_tracker
from .types import (MAX_KEPT_VIEWS, MIN_ACTIVE_OBS, MIN_MATURE_VIEWS, Observation,
                    ObjectTrack, TrackState)

__all__ = [
    "run_tracker", "ObjectTrack", "Observation", "TrackState",
    "Appearance", "HSVHistogram", "cosine", "VoxelCloud",
    "MIN_ACTIVE_OBS", "MIN_MATURE_VIEWS", "MAX_KEPT_VIEWS",
    "GATE_MIN_OBS", "GATE_MIN_SCORE",
]
