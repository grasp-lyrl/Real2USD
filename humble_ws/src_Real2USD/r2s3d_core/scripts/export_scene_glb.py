"""Assemble the placed objects from sam3d_layout into combined world-frame scene GLBs,
for visual comparison of scene reconstruction quality: FULL vs CROP vs GT.

Uses the cached SAM3D outputs (no inference). Writes to results/ablation_full_vs_crop/.

  uv run python scripts/export_scene_glb.py [scene]
"""
import sys, os
import numpy as np
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from r2s3d_core.data.registry import make_source
from r2s3d_core.baselines import get_method
from r2s3d_core.eval.metrics import gt_to_scene_object

SCENE = sys.argv[1] if len(sys.argv) > 1 else "room0"
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "ablation_full_vs_crop")


def combine(meshes, color):
    geoms = []
    for m in meshes:
        if m is None or not len(getattr(m, "vertices", [])):
            continue
        g = m.copy()
        g.visual = trimesh.visual.ColorVisuals(g, vertex_colors=np.tile(color, (len(g.vertices), 1)))
        geoms.append(g)
    return trimesh.util.concatenate(geoms) if geoms else None


def scene_from(method, config):
    src = make_source("replica", SCENE, stride=20)
    gt = src.gt()
    preds = get_method(method)(src, gt, config)
    return [p.mesh for p in preds], gt


os.makedirs(OUT, exist_ok=True)
base = dict(sam3d_queue=None, compute_geometry=False)

for tag, cfg in [("full", {**base, "full_frame": True}), ("crop", {**base, "full_frame": False})]:
    meshes, gt = scene_from("sam3d_layout", cfg)
    col = (80, 170, 255, 255) if tag == "full" else (255, 140, 80, 255)
    scene = combine(meshes, col)
    if scene is not None:
        p = os.path.join(OUT, f"scene_{tag}.glb")
        scene.export(p)
        print(f"{tag}: {len(meshes)} objects -> {p}  ({len(scene.vertices)} verts)")
    if tag == "full":
        gt_scene = combine([gt_to_scene_object(g).mesh for g in gt], (120, 120, 120, 255))
        if gt_scene is not None:
            p = os.path.join(OUT, "scene_gt.glb")
            gt_scene.export(p)
            print(f"gt: {len(gt)} objects -> {p}  ({len(gt_scene.vertices)} verts)")

print("open the .glb files in any viewer (blender, https://gltf-viewer.donmccurdy.com, VS Code glTF ext)")
