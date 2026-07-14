"""Core data interfaces shared by every SequenceSource backend.

Frame convention (load-bearing, enforced at the loader boundary — never downstream):

* Camera frame is the OpenCV optical frame: **x right, y down, z forward**.
* ``T_a_b`` is a 4x4 homogeneous transform that maps *column-vector* points
  expressed in frame ``b`` into frame ``a``:  ``p_a = T_a_b @ p_b``.
* The world frame is **gravity-aligned, Z-up**. Backends must convert their
  native world convention (e.g. Replica/NICE-SLAM traj poses) to Z-up world at
  load time so all downstream code sees one convention.
"""

from __future__ import annotations

from typing import Iterator, List, NamedTuple, Optional, Protocol, runtime_checkable

import numpy as np

try:  # trimesh is a hard dep but keep the import guard tidy for type-only use
    import trimesh
except Exception:  # pragma: no cover
    trimesh = None  # type: ignore


class Frame(NamedTuple):
    """One posed RGB-D frame."""

    rgb: np.ndarray          # (H, W, 3) uint8
    depth: np.ndarray        # (H, W) float32, meters; 0 or NaN = invalid
    K: np.ndarray            # (3, 3) intrinsics for THIS rgb/depth resolution
    T_world_cam: np.ndarray  # (4, 4) camera-to-world; camera = OpenCV optical
    stamp: float             # seconds
    frame_id: int


class GTObject(NamedTuple):
    """One ground-truth object instance in the gravity-aligned Z-up world frame."""

    instance_id: int
    label: str
    T_world_obj: np.ndarray        # (4, 4) object-to-world (OBB pose)
    extents: np.ndarray            # (3,) full OBB dimensions, meters
    mesh: Optional["trimesh.Trimesh"] = None  # per-instance GT mesh (world frame)
    asset_id: Optional[str] = None  # source asset identifier (e.g. THOR assetId), if known


@runtime_checkable
class SequenceSource(Protocol):
    """A posed RGB-D sequence with optional ground truth.

    Backends: ``replica`` (Phase 0), ``scannet`` / ``rosbag`` (later). Every
    benchmark and the robot data flow through this one interface.
    """

    def __iter__(self) -> Iterator[Frame]: ...

    def __len__(self) -> int: ...

    def gt(self) -> Optional[List[GTObject]]: ...
