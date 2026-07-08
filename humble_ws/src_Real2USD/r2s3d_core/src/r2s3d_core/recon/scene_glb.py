"""Export posed objects to a viewable GLB scene (quick visual inspection).

Not the Phase-4 canonical exporter — just enough to eyeball placement results in any
glTF viewer (Blender, VS Code glTF Tools, https://gltf-viewer.donmccurdy.com).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import trimesh

log = logging.getLogger(__name__)


def _make_lite(mesh, max_faces: int, tint):
    """Strip textures (the bulk of SAM3D GLB weight) and optionally decimate.

    Textures dominate size (~9 MB/object for SAM3D); a flat-colored, decimated mesh
    preserves shape+pose for inspection at a tiny fraction of the size.
    """
    g = mesh.copy()
    if max_faces and len(g.faces) > max_faces:
        try:  # needs fast-simplification: uv sync --extra viz
            import fast_simplification

            reduction = 1.0 - min(0.99, max_faces / len(g.faces))  # fraction to remove
            v, f = fast_simplification.simplify(
                np.asarray(g.vertices), np.asarray(g.faces), target_reduction=reduction)
            g = trimesh.Trimesh(vertices=v, faces=f, process=False)
        except Exception as e:
            log.warning("decimation unavailable (%s); lite mesh keeps %d faces — "
                        "run `uv sync --extra viz`", e, len(g.faces))
    color = np.asarray(tint if tint is not None else [180, 180, 185, 255], np.uint8)
    g.visual = trimesh.visual.ColorVisuals(g, face_colors=np.tile(color, (len(g.faces), 1)))
    return g


def export_scene_glb(meshes: Iterable, path, tint=None, lite: bool = False,
                     max_faces: int = 6000) -> Path:
    """Write world-frame trimesh meshes into one GLB.

    lite=True strips textures (flat color) and decimates to ~max_faces/object for
    fast viewing; full fidelity otherwise. Optional RGBA ``tint`` (applied in both).
    """
    scene = trimesh.Scene()
    for i, m in enumerate(meshes):
        if m is None or len(getattr(m, "faces", [])) == 0:
            continue
        if lite:
            g = _make_lite(m, max_faces, tint)
        else:
            g = m.copy()
            if tint is not None:
                g.visual = trimesh.visual.ColorVisuals(
                    g, face_colors=np.tile(np.asarray(tint, np.uint8), (len(g.faces), 1)))
        scene.add_geometry(g, node_name=f"obj_{i}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(path))
    return path


def export_pred_vs_gt(preds, gts, out_dir, lite: bool = True, full: bool = False) -> dict:
    """Write pred / gt / compare GLBs for one scene.

    lite (default): fast-viewing versions (`*_lite.glb`, textures stripped). full:
    also write the full-texture versions (large; for figures/deliverables). The
    compare scene overlays predictions (colored) with GT as translucent gray.
    """
    out_dir = Path(out_dir)
    pred_meshes = [p.mesh for p in preds if getattr(p, "mesh", None) is not None]
    gt_meshes = [g.mesh for g in gts if getattr(g, "mesh", None) is not None]
    paths = {}

    def _write(suffix, l):
        p = export_scene_glb(pred_meshes, out_dir / f"scene_pred{suffix}.glb", lite=l,
                             tint=[70, 130, 200, 255] if l else None)
        g = export_scene_glb(gt_meshes, out_dir / f"scene_gt{suffix}.glb", lite=l)
        # compare = pred (colored) + GT translucent gray, both lite for viewing
        scene = trimesh.Scene()
        for i, m in enumerate(pred_meshes):
            mm = _make_lite(m, 6000, [70, 130, 200, 255]) if l else m.copy()
            scene.add_geometry(mm, node_name=f"pred_{i}")
        for i, m in enumerate(gt_meshes):
            scene.add_geometry(_make_lite(m, 6000, [160, 160, 160, 120]), node_name=f"gt_{i}")
        c = out_dir / f"scene_compare{suffix}.glb"
        scene.export(str(c))
        return {f"pred{suffix}": p, f"gt{suffix}": g, f"compare{suffix}": c}

    if lite:
        paths.update(_write("_lite", True))
    if full:
        paths.update(_write("", False))
    return paths
