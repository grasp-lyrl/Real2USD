"""Supervisely 3D-cuboid ground truth for the Go2 real-robot scenes.

The v1 paper scenes (smalloffice-0/1, hallway-1, lounge-0) were hand-annotated in
Supervisely with oriented 3D bounding boxes drawn on the fused LiDAR point cloud
(``<scene>_voxel_pointcloud.pcd``). Those clouds are in the robot **odom frame,
Z-up, absolute coordinates** (not re-centered to the first pose) — the same frame
:func:`r2s3d_core.frames.T_odom_cam_go2` puts camera poses in — so GT boxes and
predictions built by the pipeline align directly with no extra transform.

Annotation format (verified against the dumps):

* ``objects[]`` : ``{id, classId, classTitle}`` — ``classTitle`` is the class name.
* ``figures[]`` : one per box instance, ``{objectId, geometry}`` where
  ``geometry = {position:{x,y,z}, rotation:{x,y,z}, dimensions:{x,y,z}}``.
  ``position`` is the box center, ``rotation`` is Euler **radians** (scipy "xyz",
  in practice pure yaw about Z), ``dimensions`` are **full** extents (meters).
* ``figures[].objectId`` -> ``objects[].id`` gives the per-box label.

This mirrors v1 ``evaluations/eval_common.load_supervisely_gt`` but emits the
r2s3d_core :class:`~r2s3d_core.data.base.GTObject` directly. Label normalization is
vendored from ``evaluations/label_aliases.json`` so this stays ROS/eval-harness free.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from .base import GTObject

log = logging.getLogger(__name__)

# Vendored from evaluations/label_aliases.json + canonical_labels.json (v1 eval).
# Exact raw->canonical string map, tried first.
_ALIASES = {
    "high chair": "chair", "little chair": "chair", "lounge chair": "chair",
    "couch": "chair", "sofa": "chair", "stool": "chair", "bench": "chair",
    "round table": "table", "tall table": "table", "desk": "table",
    "door": "door", "elevator": "door",
}
# Substring fallback when no exact alias hits.
_CONTAINS = [
    ("chair", "chair"), ("stool", "chair"), ("bench", "chair"),
    ("sofa", "chair"), ("couch", "chair"),
    ("table", "table"), ("desk", "table"),
    ("door", "door"),
]
# The classes v1 actually evaluated. Anything else stays as its raw lowercased
# label (kept in GT, but downstream class-restricted metrics will ignore it).
CANONICAL_LABELS = ("chair", "table", "door")


def normalize_label(raw: str) -> str:
    """Map a Supervisely ``classTitle`` to the v1 canonical label (else raw lower)."""
    s = (raw or "").strip().lower()
    if not s:
        return "__unlabeled__"
    if s in _ALIASES:
        return _ALIASES[s]
    for needle, canon in _CONTAINS:
        if needle in s:
            return canon
    return s


def load_supervisely_gt(
    json_path: str | Path,
    canonical_only: bool = False,
) -> List[GTObject]:
    """Parse a Supervisely ``.pcd.json`` into odom-frame :class:`GTObject` boxes.

    Parameters
    ----------
    json_path : path to ``<scene>_voxel_pointcloud.pcd.json``.
    canonical_only : if True, drop boxes whose label is not one of
        :data:`CANONICAL_LABELS` (the v1 chair/table/door eval set).
    """
    json_path = Path(json_path)
    with open(json_path) as f:
        data = json.load(f)

    id_to_title = {o["id"]: o.get("classTitle", "") for o in data.get("objects", [])}
    out: List[GTObject] = []
    for i, fig in enumerate(data.get("figures", [])):
        geom = fig.get("geometry") or {}
        pos = geom.get("position") or {}
        rot = geom.get("rotation") or {}
        dim = geom.get("dimensions") or {}
        center = np.array([pos.get("x", 0.0), pos.get("y", 0.0), pos.get("z", 0.0)], dtype=np.float64)
        rot_xyz = np.array([rot.get("x", 0.0), rot.get("y", 0.0), rot.get("z", 0.0)], dtype=np.float64)
        extents = np.array([dim.get("x", 0.0), dim.get("y", 0.0), dim.get("z", 0.0)], dtype=np.float64)
        if not np.all(extents > 0):
            log.warning("%s figure %d has non-positive extents %s; skipping", json_path.name, i, extents)
            continue

        Rm = Rotation.from_euler("xyz", rot_xyz, degrees=False).as_matrix()  # radians
        T = np.eye(4)
        T[:3, :3] = Rm
        T[:3, 3] = center

        raw_label = id_to_title.get(fig.get("objectId"), "")
        label = normalize_label(raw_label)
        if canonical_only and label not in CANONICAL_LABELS:
            continue
        out.append(GTObject(
            instance_id=i,
            label=label,
            T_world_obj=T,
            extents=extents,
        ))
    if not out:
        log.warning("no GT boxes parsed from %s", json_path)
    return out
