"""Export the FULL-IMAGE-mask SAM3D reconstructions (exp_full_scene_fullmask) as GLBs
you can open in any 3D viewer, to eyeball what "whole scene as one object" produces.

Reuses the cached worker outputs (no inference). For each selected frame it places the
raw object.glb into the world via place_from_sam3d() @ T_world_cam, decimates it to keep
the file light, and writes:
  results/exp_full_scene_fullmask_<src>_<scene>/glb/
    frame<FID>.glb        # one posed blob per view (inspect a single instance)
    combined.glb          # all views in one world frame, color-coded per frame

Usage (after running scripts/exp_full_scene_fullmask.py so the jobs are cached):
  uv run python scripts/export_fullmask_glb.py [--source replica] [--scene room0]
                                               [--split val] [--max-frames 6]
                                               [--target-faces 80000]
"""
import argparse
import os
import sys

import numpy as np
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from r2s3d_core.data.registry import make_source  # noqa: E402
from r2s3d_core.baselines.sam3d_layout import (  # noqa: E402
    place_from_sam3d, run_sam3d, _default_queue,
)
from exp_full_scene_fullmask import pick_frames  # noqa: E402 (scripts/ is on sys.path)

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
# distinct per-frame colors (RGBA) for the combined view
PALETTE = [(66, 135, 245, 255), (245, 130, 66, 255), (95, 212, 138, 255),
           (240, 112, 112, 255), (190, 130, 245, 255), (240, 210, 90, 255),
           (90, 220, 220, 255), (245, 150, 200, 255)]


def decimate(mesh, target_faces):
    """Quadric-decimate to ~target_faces; fall back to the full mesh if unavailable."""
    if target_faces <= 0 or len(mesh.faces) <= target_faces:
        return mesh
    try:
        return mesh.simplify_quadric_decimation(face_count=target_faces)
    except Exception as e:
        print(f"    [warn] decimation unavailable ({e}); exporting full-res")
        return mesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="replica")
    ap.add_argument("--scene", default="room0")
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--max-frames", type=int, default=6)
    ap.add_argument("--split", default="val")
    ap.add_argument("--target-faces", type=int, default=80000,
                    help="per-mesh decimation target (0 = no decimation)")
    args = ap.parse_args()

    src_kwargs = {"stride": args.stride}
    if args.source in ("procthor", "molmospaces"):
        src_kwargs["split"] = args.split
    src = make_source(args.source, args.scene, **src_kwargs)
    frames_list = list(src)
    queue = _default_queue()
    outdir = os.path.join(RESULTS, f"exp_full_scene_fullmask_{args.source}_{args.scene}", "glb")
    os.makedirs(outdir, exist_ok=True)
    sel = pick_frames(frames_list, args.max_frames)
    print(f"[{args.source}/{args.scene}] exporting {len(sel)} frame(s) -> {outdir}")

    combined, missing = [], 0
    for i, vi in enumerate(sel):
        frame = frames_list[vi]
        H, W = frame.depth.shape[:2]
        full_mask = np.full((H, W), 255, np.uint8)
        job_key = f"{args.source}_{args.scene}_frame{int(frame.frame_id)}_fullscene"
        result = run_sam3d(  # cached: returns (mesh, pose) or None if not yet run
            frame.rgb, full_mask, frame.depth, frame.K, (0, 0, W - 1, H - 1),
            meta={"track_id": 0, "label": "full_scene", "full_width": W, "full_height": H},
            queue=queue, job_key=job_key)
        if result is None:
            print(f"  frame {frame.frame_id}: NOT cached — run exp_full_scene_fullmask.py + worker first")
            missing += 1
            continue
        mesh_raw, pose = result
        posed, _, extents = place_from_sam3d(
            mesh_raw, pose["sam3d_scale"], pose["sam3d_rotation"], pose["sam3d_translation"],
            frame.T_world_cam)
        posed = decimate(posed, args.target_faces)
        fp = os.path.join(outdir, f"frame{int(frame.frame_id)}.glb")
        posed.export(fp)
        print(f"  frame {frame.frame_id:>4}: {len(posed.faces)} faces, "
              f"OBB diag {float(np.linalg.norm(extents)):.1f}m -> {os.path.basename(fp)}")
        g = posed.copy()
        col = PALETTE[i % len(PALETTE)]
        g.visual = trimesh.visual.ColorVisuals(g, vertex_colors=np.tile(col, (len(g.vertices), 1)))
        combined.append(g)

    if combined:
        scene = trimesh.util.concatenate(combined)
        cp = os.path.join(outdir, "combined.glb")
        scene.export(cp)
        print(f"  combined ({len(combined)} views, color-coded per frame): "
              f"{len(scene.faces)} faces -> {cp}")
    if missing:
        print(f"\n{missing} frame(s) not cached — nothing exported for them.")
    print("\nopen in any glTF viewer (blender, https://gltf-viewer.donmccurdy.com, VS Code glTF ext).")


if __name__ == "__main__":
    main()
