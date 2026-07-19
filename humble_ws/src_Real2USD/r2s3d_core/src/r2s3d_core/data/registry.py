"""SequenceSource backend registry and data-root resolution.

Data root defaults to ``$R2S3D_DATA`` or ``~/Data/datasets`` (the workstation does
not expose a writable ``/data``; see docs/DATASETS.md). Override per-call or via env.
"""

from __future__ import annotations

import os
from pathlib import Path

from .base import SequenceSource


def data_root() -> Path:
    return Path(os.environ.get("R2S3D_DATA", str(Path.home() / "Data" / "datasets"))).expanduser()


def make_source(source: str, scene: str, root: str | os.PathLike | None = None, **kwargs) -> SequenceSource:
    source = source.lower()
    if source == "replica":
        from .replica import ReplicaSource

        base = Path(root) if root else data_root() / "replica"
        return ReplicaSource(base, scene, **kwargs)
    if source == "synthetic":
        from .synthetic import SyntheticSource

        return SyntheticSource(root, scene, **kwargs)
    if source in ("procthor", "molmospaces"):
        from .procthor import ProcThorSource

        return ProcThorSource(root, scene, **kwargs)
    if source in ("rosbag", "go2", "lidar"):
        from .rosbag import RosbagSource, resolve_scene

        bag, gt = resolve_scene(scene, root)
        return RosbagSource(bag, gt_json=gt, scene=scene, **kwargs)
    if source in ("realsense", "rs"):
        from .realsense import RealSenseSource, resolve_rs_scene

        bag, gt = resolve_rs_scene(scene, root)
        return RealSenseSource(bag, gt_json=gt, scene=scene, **kwargs)
    raise ValueError(
        f"unknown SequenceSource backend: {source!r} "
        "(have: replica, synthetic, procthor, rosbag, realsense)"
    )
