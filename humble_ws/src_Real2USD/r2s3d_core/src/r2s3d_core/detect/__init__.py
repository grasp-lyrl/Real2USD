"""Detector step for Phase 2 (detector-in-sim).

Runs a real open-vocabulary detector (YOLOE) over a :class:`SequenceSource`'s RGB
frames and caches per-frame detections to disk. The heavy model (torch/ultralytics)
lives behind the ``detector`` optional extra and runs as a standalone step; the
tracker, eval, and tests consume the cached :class:`DetectionSet` and import no
torch (mirrors the SAM3D disk-queue handoff in ``baselines/sam3d_layout.py``).

Design rationale: Phase 0/1 fed SAM3D *perfect GT masks* (one call per GT instance).
Phase 2 drives the ObjectTrack pipeline with a real detector so we can (a) measure how
much placement/coverage degrades vs the GT-mask ceiling, (b) study detector prompting
as its own variable, and (c) show multi-view association + late-merge cleaning up the
duplicate objects a real detector produces. See docs/PHASE_SPECS.md §Phase 2.
"""

from __future__ import annotations

from .cache import Detection, DetectionSet

__all__ = ["Detection", "DetectionSet"]
