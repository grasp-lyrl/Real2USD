"""ObjectTrack data model (docs/PHASE_SPECS.md §Phase 2, REWORK_PLAN.md §2.2).

A track is the persistent per-object entity that turns a single-image generator into a
scene mapper: it accumulates per-view observations, a fused multi-view point cloud, a
label-vote histogram, and a running appearance descriptor, and it maintains a bounded
buffer of its best views (the ones fed to SAM3D).

Everything is in the gravity-aligned Z-up world frame (see data/base.py). No ROS, no
torch.
"""

from __future__ import annotations

import enum
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


class TrackState(enum.Enum):
    TENTATIVE = "tentative"        # created on an unmatched detection
    ACTIVE = "active"              # >= MIN_ACTIVE_OBS associated observations
    MATURE = "mature"             # >= MIN_MATURE_VIEWS kept views or sequence end
    RECONSTRUCTED = "reconstructed"  # SAM3D mesh generated
    REGISTERED = "registered"     # posed into the scene
    MERGED = "merged"             # terminal: folded into another track
    REJECTED = "rejected"         # terminal: filtered out


# Lifecycle thresholds (PHASE_SPECS §Phase 2).
MIN_ACTIVE_OBS = 3
MIN_MATURE_VIEWS = 6
MAX_KEPT_VIEWS = 6


@dataclass
class Observation:
    """One view of a track. ``mask``/``rgb_crop`` are retained only while the
    observation is in the track's top-K best-view buffer; evicted views null them to
    bound memory (the masked depth was already fused at ingest)."""

    frame_index: int                 # index into the tracker's in-memory frames list
    frame_id: int                    # source Frame.frame_id
    stamp: float
    bbox: np.ndarray                 # (4,) xyxy full-image px
    det_label: str
    det_score: float
    det_track_id: int
    centroid_world: np.ndarray       # (3,) from mask-median depth
    appearance: np.ndarray           # (D,) appearance descriptor
    view_dir_world: np.ndarray       # (3,) camera optical axis in world
    view_score: float = 0.0
    n_mask_px: int = 0
    mask: Optional[np.ndarray] = None      # (H,W) bool, kept-views only
    rgb_crop: Optional[np.ndarray] = None  # small bbox crop, kept-views only


@dataclass
class ObjectTrack:
    track_id: int
    state: TrackState = TrackState.TENTATIVE
    observations: List[Observation] = field(default_factory=list)
    kept_views: List[Observation] = field(default_factory=list)   # top-K by view_score
    label_votes: Counter = field(default_factory=Counter)
    appearance_sum: Optional[np.ndarray] = None    # running sum -> mean
    n_appearance: int = 0
    voxels: object = None                           # tracks.fusion.VoxelCloud
    fused_cloud: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    centroid: Optional[np.ndarray] = None
    last_seen_stamp: float = -1.0
    det_track_ids: set = field(default_factory=set)
    merged_from: List[int] = field(default_factory=list)
    canonical_mesh: object = None

    # ------------------------------------------------------------------ helpers
    @property
    def appearance_mean(self) -> Optional[np.ndarray]:
        if self.appearance_sum is None or self.n_appearance == 0:
            return None
        v = self.appearance_sum / self.n_appearance
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def label(self) -> str:
        if not self.label_votes:
            return "object"
        return self.label_votes.most_common(1)[0][0]

    @property
    def n_obs(self) -> int:
        return len(self.observations)

    def is_active(self) -> bool:
        return self.state in (TrackState.ACTIVE, TrackState.MATURE,
                              TrackState.RECONSTRUCTED, TrackState.REGISTERED)

    def add_appearance(self, emb: Optional[np.ndarray]) -> None:
        if emb is None:
            return
        if self.appearance_sum is None:
            self.appearance_sum = np.zeros_like(emb, dtype=np.float64)
        self.appearance_sum += emb
        self.n_appearance += 1
