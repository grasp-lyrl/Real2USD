"""Oracle methods: sanity baselines that exercise the harness end-to-end.

``oracle`` returns GT placed exactly — every metric should be perfect (IoU 1,
0 cm / 0 deg error, Scan2CAD accuracy 1.0). ``oracle_noisy`` perturbs GT poses by
a configurable SE(3)+scale noise so metrics degrade predictably. These require no
SAM3D / checkpoint / GPU and validate the whole run.json + metrics + table path
while the heavy baselines are gated on data + model access.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..data.base import GTObject
from ..eval.metrics import SceneObject, gt_to_scene_object


def oracle(source, gt: Optional[List[GTObject]], config: dict) -> List[SceneObject]:
    if not gt:
        return []
    return [gt_to_scene_object(g) for g in gt]


def oracle_noisy(source, gt: Optional[List[GTObject]], config: dict) -> List[SceneObject]:
    if not gt:
        return []
    rng = np.random.RandomState(int(config.get("seed", 0)))
    t_sigma = float(config.get("trans_noise_m", 0.05))
    r_sigma = float(config.get("rot_noise_deg", 5.0))
    s_sigma = float(config.get("scale_noise", 0.05))
    out = []
    for g in gt:
        T = g.T_world_obj.copy()
        T[:3, 3] += rng.normal(0, t_sigma, 3)
        yaw = np.radians(rng.normal(0, r_sigma))
        c, s = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        T[:3, :3] = Rz @ T[:3, :3]
        ext = g.extents * (1.0 + rng.normal(0, s_sigma, 3))
        out.append(SceneObject(label=g.label, T_world_obj=T, extents=ext, mesh=g.mesh))
    return out
