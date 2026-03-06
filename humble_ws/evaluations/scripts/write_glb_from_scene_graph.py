#!/usr/bin/env python3
"""
Write a single GLB from an existing scene-graph JSON (no label resolution).

Use this after editing a scene_graph*.json by hand: read the JSON and export
meshes + optional RealSense point cloud to one GLB. No eval config or label
filtering.

Usage (from evaluations/ or with PYTHONPATH including real2sam3d):

  # GLB only (meshes from JSON)
  PYTHONPATH=../src_Real2USD/real2sam3d python scripts/write_glb_from_scene_graph.py \\
    --scene-graph /path/to/edited.json --out-glb /path/to/out.glb

  # GLB + RealSense point cloud from run_dir
  PYTHONPATH=../src_Real2USD/real2sam3d python scripts/write_glb_from_scene_graph.py \\
    --scene-graph /path/to/edited.json --out-glb /path/to/out.glb --run-dir /path/to/run_dir

  # With z offset for point cloud (e.g. move down 0.2m)
  ... --run-dir /path/to/run --realsense-z-offset -0.2
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Allow imports from parent (evaluations/)
_SCRIPT_DIR = Path(__file__).resolve().parent
_EVAL_ROOT = _SCRIPT_DIR.parent
if str(_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_EVAL_ROOT))


def _objects_to_scene_list(objs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build scene-graph-style list for write_joint_glb_from_list_standalone from JSON objects."""
    out = []
    for i, o in enumerate(objs):
        entry = {
            "id": o.get("id", i),
            "data_path": o.get("data_path"),
            "position": o.get("position"),
            "orientation": o.get("orientation"),
        }
        if not entry["data_path"]:
            continue
        if o.get("transform_odom_from_raw") is not None:
            entry["transform_odom_from_raw"] = o["transform_odom_from_raw"]
        out.append(entry)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write one GLB from a scene-graph JSON (optional: add RealSense point cloud from run_dir)."
    )
    parser.add_argument("--scene-graph", type=Path, required=True, help="Input scene_graph JSON")
    parser.add_argument("--out-glb", type=Path, required=True, help="Output GLB path")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="Run directory for RealSense point cloud (diagnostics/pointclouds/realsense/*.npy)")
    parser.add_argument("--realsense-voxel-size", type=float, default=0.02, metavar="M",
                        help="Voxel size (m) for point cloud downsampling (default 0.02). Use 0 to disable.")
    parser.add_argument("--realsense-use-all-files", action="store_true",
                        help="Merge all realsense_*.npy files (default: single file with most points)")
    parser.add_argument("--realsense-z-offset", type=float, default=0.0, metavar="M",
                        help="Z translation for point cloud in meters (e.g. -0.2 to move down). Default 0.")
    args = parser.parse_args()

    scene_graph = args.scene_graph.resolve()
    if not scene_graph.exists():
        print("Error: scene graph not found:", scene_graph, file=sys.stderr)
        sys.exit(1)

    with open(scene_graph) as f:
        data = json.load(f)
    objs = data.get("objects", [])
    if not objs:
        print("Error: no objects in", scene_graph, file=sys.stderr)
        sys.exit(1)

    scene_list = _objects_to_scene_list(objs)
    print("Loaded %d objects from %s" % (len(scene_list), scene_graph))

    try:
        from real2sam3d.glb_export_standalone import write_joint_glb_from_list_standalone
    except ImportError as e:
        print("Error: real2sam3d required for GLB writing. Set PYTHONPATH to real2sam3d package root.", file=sys.stderr)
        raise SystemExit(1) from e

    class Log:
        def info(self, msg): print(msg)
        def warn(self, msg): print(msg, file=sys.stderr)

    sig = inspect.signature(write_joint_glb_from_list_standalone)
    kwargs = {
        "run_dir": args.run_dir,
        "log": Log(),
    }
    if "realsense_voxel_size" in sig.parameters:
        kwargs["realsense_voxel_size"] = args.realsense_voxel_size if args.realsense_voxel_size > 0 else None
    if "realsense_use_all_files" in sig.parameters:
        kwargs["realsense_use_all_files"] = args.realsense_use_all_files
    if "realsense_z_offset" in sig.parameters:
        kwargs["realsense_z_offset"] = args.realsense_z_offset

    out_glb = args.out_glb.resolve()
    out_glb.parent.mkdir(parents=True, exist_ok=True)
    write_joint_glb_from_list_standalone(out_glb, scene_list, **kwargs)
    print("Wrote %s" % out_glb)


if __name__ == "__main__":
    main()
