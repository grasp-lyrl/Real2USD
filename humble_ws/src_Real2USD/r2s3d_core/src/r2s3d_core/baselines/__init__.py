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
    if name in ("sam3d_layout", "sam3d_layout_icp"):
        from .sam3d_layout import sam3d_layout, sam3d_layout_icp

        return {"sam3d_layout": sam3d_layout, "sam3d_layout_icp": sam3d_layout_icp}[name]
    raise ValueError(
        f"unknown method {name!r} (have: oracle, oracle_noisy, sam3d_layout, sam3d_layout_icp)"
    )


AVAILABLE = ["oracle", "oracle_noisy", "sam3d_layout", "sam3d_layout_icp"]
