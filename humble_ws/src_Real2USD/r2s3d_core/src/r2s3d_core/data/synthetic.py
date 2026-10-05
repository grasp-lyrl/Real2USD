"""Synthetic SequenceSource for harness self-tests.

A tiny, deterministic fake scene (a few boxed GT objects on a common floor plane
plus a couple of dummy posed frames) so the full runner -> run.json -> table path
can be exercised without any downloaded data, checkpoint, or GPU. Used by the
oracle sanity methods and the runner smoke test.
"""

from __future__ import annotations

from typing import Iterator, List, Optional

import numpy as np
import trimesh

from .base import Frame, GTObject


def _box_object(instance_id: int, label: str, center, extents, yaw_deg=0.0) -> GTObject:
    th = np.radians(yaw_deg)
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(center, float)
    mesh = trimesh.creation.box(extents=np.asarray(extents, float))
    mesh.apply_transform(T)
    return GTObject(instance_id=instance_id, label=label, T_world_obj=T,
                    extents=np.asarray(extents, float), mesh=mesh)


class SyntheticSource:
    """A fixed 4-object scene; ``scene`` argument is ignored (single scene)."""

    def __init__(self, root=None, scene: str = "synthetic", stride: int = 1, load_gt: bool = True):
        self.scene = scene
        self.stride = stride
        self._load_gt = load_gt
        # objects sit on the floor (bottom at z=0), gravity-consistent Z-up
        self._gt = [
            _box_object(1, "chair", [0.0, 0.0, 0.45], [0.5, 0.5, 0.9], yaw_deg=10),
            _box_object(2, "table", [1.5, 0.5, 0.375], [1.2, 0.8, 0.75], yaw_deg=0),
            _box_object(3, "sofa", [-1.5, 1.0, 0.4], [1.8, 0.9, 0.8], yaw_deg=90),
            _box_object(4, "lamp", [0.5, -1.5, 0.6], [0.3, 0.3, 1.2], yaw_deg=45),
        ]
        self.K = np.array([[300.0, 0, 320], [0, 300.0, 240], [0, 0, 1.0]])

    def __len__(self) -> int:
        return 2

    def __iter__(self) -> Iterator[Frame]:
        for i in range(2):
            T = np.eye(4)
            T[:3, 3] = [0.0, -3.0 + i, 1.0]
            yield Frame(
                rgb=np.zeros((480, 640, 3), np.uint8),
                depth=np.zeros((480, 640), np.float32),
                K=self.K.copy(),
                T_world_cam=T,
                stamp=float(i),
                frame_id=i,
            )

    def gt(self) -> Optional[List[GTObject]]:
        return self._gt if self._load_gt else None
