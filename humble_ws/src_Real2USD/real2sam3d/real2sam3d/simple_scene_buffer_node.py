"""
Simple scene buffer node: subscribes to registration output (/usd/StringIdPose),
maintains a single scene buffer, and writes:
  - scene_graph.json: one JSON describing the whole scene (id, data_path, pose per object)
  - scene.glb: one joint GLB with all objects merged at their poses.
"""

import json
import os
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R

from custom_message.msg import UsdStringIdPoseMsg


def _pose_to_matrix(position_xyz, quat_xyzw):
    """Build 4x4 transform from position (3,) and quaternion (x,y,z,w)."""
    T = np.eye(4)
    T[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
    T[:3, 3] = position_xyz
    return T


def _load_meshes_from_path(data_path: str):
    """Load trimesh geometry from data_path (GLB or PLY). Returns list of Trimesh or None."""
    try:
        import trimesh
    except ImportError:
        return None
    path = Path(data_path)
    if not path.exists():
        return None
    suffix = path.suffix.lower()
    if suffix == ".ply":
        # Prefer object.glb in same dir if present
        glb = path.parent / "object.glb"
        if glb.exists():
            path = glb
            suffix = ".glb"
    if suffix != ".glb":
        return None
    try:
        scene = trimesh.load(str(path), process=False)
        if isinstance(scene, trimesh.Scene):
            return [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if isinstance(scene, trimesh.Trimesh):
            return [scene]
    except Exception:
        pass
    return None


class SimpleSceneBufferNode(Node):
    """
    Subscribes to /usd/StringIdPose (registration output), keeps a single scene buffer,
    and periodically writes scene_graph.json and a joint scene.glb.
    """

    def __init__(self):
        super().__init__("simple_scene_buffer_node")

        self.declare_parameter("output_dir", "/data/sam3d_queue")
        self.declare_parameter("write_interval_sec", 5.0)
        self.declare_parameter("scene_json_name", "scene_graph.json")
        self.declare_parameter("scene_glb_name", "scene.glb")

        self._output_dir = Path(self.get_parameter("output_dir").value)
        self._write_interval_sec = self.get_parameter("write_interval_sec").value
        self._scene_json_name = self.get_parameter("scene_json_name").value
        self._scene_glb_name = self.get_parameter("scene_glb_name").value

        # id -> { "data_path", "position", "orientation" }
        self._buffer: dict[int, dict] = {}

        self._sub = self.create_subscription(
            UsdStringIdPoseMsg,
            "/usd/StringIdPose",
            self._pose_callback,
            10,
        )
        self._timer = self.create_timer(self._write_interval_sec, self._write_scene)

        self.get_logger().info(
            "simple_scene_buffer_node: output_dir=%s, write_interval=%.1fs"
            % (self._output_dir, self._write_interval_sec)
        )

    def _pose_callback(self, msg: UsdStringIdPoseMsg):
        position = [
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ]
        orientation = [
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ]
        self._buffer[int(msg.id)] = {
            "data_path": msg.data_path,
            "position": position,
            "orientation": orientation,
        }

    def _write_scene(self):
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # 1) Write scene graph JSON
        scene_list = []
        for obj_id, data in self._buffer.items():
            scene_list.append({
                "id": obj_id,
                "data_path": data["data_path"],
                "position": data["position"],
                "orientation": data["orientation"],
            })
        json_path = self._output_dir / self._scene_json_name
        try:
            with open(json_path, "w") as f:
                json.dump({"objects": scene_list}, f, indent=2)
            self.get_logger().debug("Wrote %s (%d objects)" % (json_path, len(scene_list)))
        except Exception as e:
            self.get_logger().warn("Failed to write %s: %s" % (json_path, e))

        # 2) Build and write joint GLB
        glb_path = self._output_dir / self._scene_glb_name
        try:
            self._write_joint_glb(glb_path)
        except Exception as e:
            self.get_logger().warn("Failed to write joint GLB %s: %s" % (glb_path, e))

    def _write_joint_glb(self, glb_path: Path):
        try:
            import trimesh
        except ImportError:
            self.get_logger().warn("trimesh not available; skipping joint GLB")
            return

        scene = trimesh.Scene()
        added = 0
        for obj_id, data in self._buffer.items():
            path = data["data_path"]
            meshes = _load_meshes_from_path(path)
            if not meshes:
                continue
            T = _pose_to_matrix(
                np.array(data["position"], dtype=np.float64),
                np.array(data["orientation"], dtype=np.float64),
            )
            for i, mesh in enumerate(meshes):
                name = "obj_%d" % obj_id if len(meshes) == 1 else "obj_%d_%d" % (obj_id, i)
                scene.add_geometry(mesh.copy(), transform=T, node_name=name)
                added += 1

        if added == 0:
            self.get_logger().debug("No meshes to export for joint GLB")
            return
        scene.export(str(glb_path))
        self.get_logger().info("Wrote joint GLB %s (%d meshes)" % (glb_path, added))


def main(args=None):
    rclpy.init(args=args)
    node = SimpleSceneBufferNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
