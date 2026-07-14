"""Baseline method registry.

A method is a callable ``method(source, gt, config) -> list[SceneObject]`` that
produces predicted objects in the world frame for one scene. Heavy baselines are
imported lazily so the oracle/harness path runs in a light environment.
"""

from __future__ import annotations

from typing import Callable


def get_method(name: str) -> Callable:
    name = name.lower()
    if name in ("oracle", "oracle_noisy"):
        from . import oracle as _o

        return {"oracle": _o.oracle, "oracle_noisy": _o.oracle_noisy}[name]
    if name in ("sam3d_layout", "sam3d_layout_icp", "sam3d_layout_teaser"):
        from .sam3d_layout import sam3d_layout, sam3d_layout_icp, sam3d_layout_teaser

        return {"sam3d_layout": sam3d_layout, "sam3d_layout_icp": sam3d_layout_icp,
                "sam3d_layout_teaser": sam3d_layout_teaser}[name]
    if name in ("object_track", "object_track_naive"):
        from .object_track import object_track, object_track_naive

        return {"object_track": object_track, "object_track_naive": object_track_naive}[name]
    raise ValueError(
        f"unknown method {name!r} (have: {', '.join(AVAILABLE)})"
    )


AVAILABLE = ["oracle", "oracle_noisy", "sam3d_layout", "sam3d_layout_icp",
             "sam3d_layout_teaser", "object_track", "object_track_naive"]
