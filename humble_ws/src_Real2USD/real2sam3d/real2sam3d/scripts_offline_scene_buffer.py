"""
Offline scene buffer: rebuild scene_graph.json and optionally scene.glb from an existing run directory.
Uses registration output when present in each output/<job_id>/pose.json (position, orientation,
registered_data_path written by registration_node); otherwise uses initial poses from the injector.

Run from humble_ws (requires ROS env for rclpy):

  source install/setup.bash
  ros2 run real2sam3d run_offline_scene_buffer --run-dir /data/sam3d_queue/run_20260224_053223

Or (same env):

  cd humble_ws && source install/setup.bash
  PYTHONPATH=src_Real2USD/real2sam3d python3 -m real2sam3d.scripts_offline_scene_buffer --run-dir /data/sam3d_queue/run_20260224_053223
"""

import argparse
import json
import sys
from pathlib import Path

if __name__ == "__main__":
    # Run as script: add package root so real2sam3d can be imported
    _this_dir = Path(__file__).resolve().parent
    _pkg_root = _this_dir.parent
    if str(_pkg_root) not in sys.path:
        sys.path.insert(0, str(_pkg_root))
    from real2sam3d.simple_scene_buffer_node import (
        _scan_sam3d_only_scene,
        write_joint_glb_from_list_standalone,
    )
else:
    from .simple_scene_buffer_node import (
        _scan_sam3d_only_scene,
        write_joint_glb_from_list_standalone,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild scene_graph.json (and optionally scene.glb) from an existing run directory."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory (e.g. /data/sam3d_queue/run_20260224_053223) containing output/",
    )
    parser.add_argument(
        "--no-glb",
        action="store_true",
        help="Only write JSON (scene_graph.json, scene_graph_sam3d_only.json); skip GLB generation.",
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        print(f"Error: run_dir is not a directory: {run_dir}", file=sys.stderr)
        sys.exit(1)
    out_sub = run_dir / "output"
    if not out_sub.exists():
        print(f"Error: output/ not found under {run_dir}", file=sys.stderr)
        sys.exit(1)

    # Full scene: registration when present in pose.json (for scene_graph.json)
    full_list = list(_scan_sam3d_only_scene(run_dir, use_registration_when_present=True))
    # SAM3D-only: slot-initial mesh and pose only, no retrieval, no registration
    sam3d_list = list(_scan_sam3d_only_scene(run_dir, use_registration_when_present=False))
    if not full_list:
        print("No job dirs with pose.json + object.glb found under output/. Nothing to write.", file=sys.stderr)
        sys.exit(0)

    def to_objs(lst):
        objs = []
        for oid, dp, pos, quat, T_raw, label in lst:
            obj = {"id": oid, "data_path": dp, "position": pos, "orientation": quat, "transform_odom_from_raw": T_raw}
            if label is not None:
                obj["label"] = label
            objs.append(obj)
        return objs

    scene_objs = to_objs(full_list)
    sam3d_objs = to_objs(sam3d_list)

    run_dir.mkdir(parents=True, exist_ok=True)

    # scene_graph.json (registration when in pose.json, else initial)
    scene_json = run_dir / "scene_graph.json"
    with open(scene_json, "w") as f:
        json.dump({"objects": scene_objs}, f, indent=2)
    print(f"Wrote {scene_json} ({len(scene_objs)} objects)")

    # scene_graph_sam3d_only.json (slot-initial mesh and pose only, no retrieval, no registration)
    sam3d_json = run_dir / "scene_graph_sam3d_only.json"
    with open(sam3d_json, "w") as f:
        json.dump({"objects": sam3d_objs, "source": "sam3d_only_no_registration"}, f, indent=2)
    print(f"Wrote {sam3d_json} ({len(sam3d_objs)} objects)")

    if not args.no_glb:
        class Log:
            def info(self, msg): print(msg)
            def warn(self, msg): print(msg, file=sys.stderr)
        log = Log()
        scene_glb = run_dir / "scene.glb"
        write_joint_glb_from_list_standalone(scene_glb, scene_objs, run_dir=run_dir, log=log)
        sam3d_glb = run_dir / "scene_sam3d_only.glb"
        write_joint_glb_from_list_standalone(sam3d_glb, sam3d_objs, run_dir=run_dir, log=log)
    else:
        print("Skipping GLB (--no-glb).")


if __name__ == "__main__":
    main()
