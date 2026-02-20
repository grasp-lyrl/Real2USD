"""
SAM3D injector: watches queue_dir/output for new job results.

- Per slot: if pose.json has frame=sam3d_raw, runs demo_go2-style transform (ply_frame_utils)
  and writes only transforms to pose.json (cam_to_odom_*, transform_odom_from_raw, initial_*).
  object.glb is not modified. Poses are normalized by init_odom (first-step-relative, like demo_go2).
- FAISS mode: publishes SlotReadyMsg -> retrieval -> Sam3dObjectForSlotMsg -> bridge.
- No-FAISS mode (publish_object_for_slot=true): publishes Sam3dObjectForSlotMsg directly.
"""

import json
from pathlib import Path

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from std_msgs.msg import Header
from custom_message.msg import SlotReadyMsg, Sam3dObjectForSlotMsg

from real2sam3d.ply_frame_utils import (
    apply_sam3d_transforms_to_slot,
    transform_mesh_vertices,
    transform_glb_to_world,
    R_pytorch3d_to_cam,
)

TOPIC_SLOT_READY = "/usd/SlotReady"
TOPIC_OBJECT_FOR_SLOT = "/usd/Sam3dObjectForSlot"
OBJECT_ODOM_GLB = "object_odom.glb"


class Sam3dInjectorNode(Node):
    def __init__(self):
        super().__init__("sam3d_injector_node")

        self.declare_parameter("queue_dir", "/data/sam3d_queue")
        self.declare_parameter("watch_interval_sec", 1.0)
        self.declare_parameter("publish_object_for_slot", False)

        self.queue_dir = Path(self.get_parameter("queue_dir").value)
        self.watch_interval_sec = self.get_parameter("watch_interval_sec").value
        p = self.get_parameter("publish_object_for_slot").value
        self.publish_object_for_slot = p is True or (isinstance(p, str) and p.lower() == "true")
        self.output_dir = self.queue_dir / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.pub_slot_ready = self.create_publisher(SlotReadyMsg, TOPIC_SLOT_READY, 10)
        self.pub_object_for_slot = self.create_publisher(Sam3dObjectForSlotMsg, TOPIC_OBJECT_FOR_SLOT, 10)
        self._published_job_ids = set()
        self._init_odom = None
        self._timer = self.create_timer(self.watch_interval_sec, self._check_output_dir)

        if self.publish_object_for_slot:
            self.get_logger().info(
                f"SAM3D injector (no-FAISS): watching {self.output_dir}, publishing to {TOPIC_OBJECT_FOR_SLOT}"
            )
        else:
            self.get_logger().info(
                f"SAM3D injector: watching {self.output_dir}, publishing to {TOPIC_SLOT_READY}"
            )

    def _ensure_init_odom(self, job_path: Path):
        """Load init_odom from queue_dir/init_odom.json or create from first job (demo_go2-style frame)."""
        if self._init_odom is not None:
            return
        init_path = self.queue_dir / "init_odom.json"
        if init_path.exists():
            try:
                with open(init_path) as f:
                    self._init_odom = json.load(f)
                self.get_logger().info("Using init_odom from %s" % init_path)
                return
            except (OSError, json.JSONDecodeError):
                pass
        pose_path = job_path / "pose.json"
        if pose_path.exists():
            try:
                with open(pose_path) as f:
                    pose = json.load(f)
                if pose.get("frame") == "sam3d_raw" and "go2_odom_position" in pose:
                    self._init_odom = {
                        "position": list(pose["go2_odom_position"]),
                        "orientation": list(pose.get("go2_odom_orientation", [0, 0, 0, 1])),
                    }
                    with open(init_path, "w") as f:
                        json.dump(self._init_odom, f, indent=2)
                    self.get_logger().info("Wrote init_odom.json from first job %s (demo_go2-style frame)" % job_path.name)
            except (OSError, json.JSONDecodeError):
                pass

    def _check_output_dir(self):
        if not self.output_dir.exists():
            return
        # Process oldest-first so init_odom is anchored to the earliest frame,
        # matching demo_go2's "step_0001 as origin" behavior.
        job_dirs = [p for p in self.output_dir.iterdir() if p.is_dir()]
        job_dirs.sort(key=lambda p: p.stat().st_mtime)
        for job_path in job_dirs:
            if not job_path.is_dir() or job_path.name in self._published_job_ids:
                continue
            pose_path = job_path / "pose.json"
            if not pose_path.exists():
                continue
            self._ensure_init_odom(job_path)
            try:
                if apply_sam3d_transforms_to_slot(job_path, init_odom=self._init_odom):
                    self.get_logger().debug("Transformed slot %s to Z-up odom" % job_path.name)
            except Exception as e:
                self.get_logger().warn("Transform failed for %s: %s" % (job_path.name, e))
            object_glb = job_path / "object.glb"
            object_usd = job_path / "object.usd"
            object_ply = job_path / "object.ply"
            data_path = None
            if object_glb.exists():
                data_path = str(object_glb.resolve())
            elif object_usd.exists():
                data_path = str(object_usd.resolve())
            elif object_ply.exists():
                data_path = str(object_ply.resolve())
            if data_path is None:
                continue
            try:
                with open(pose_path) as f:
                    pose_data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                self.get_logger().warn("Invalid pose.json in %s: %s" % (job_path, e))
                continue

            # Write per-slot object_odom.glb immediately in injector so updates are visible
            # even if simple_scene_buffer is not running.
            self._write_object_odom_glb(job_path, pose_data)

            track_id = pose_data.get("track_id", 0)
            if isinstance(track_id, float):
                track_id = int(track_id)
            if self.publish_object_for_slot:
                msg = Sam3dObjectForSlotMsg()
                msg.header = Header()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = "odom"
                msg.job_id = job_path.name
                msg.id = track_id
                msg.data_path = data_path
                msg.pose = Pose()
                msg.pose.position.x = 0.0
                msg.pose.position.y = 0.0
                msg.pose.position.z = 0.0
                msg.pose.orientation.x = 0.0
                msg.pose.orientation.y = 0.0
                msg.pose.orientation.z = 0.0
                msg.pose.orientation.w = 1.0
                self.pub_object_for_slot.publish(msg)
                self.get_logger().info("Object for slot (no-FAISS): job_id=%s track_id=%s data_path=%s" % (job_path.name, track_id, data_path))
            else:
                msg = SlotReadyMsg()
                msg.header = Header()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = "odom"
                msg.job_id = job_path.name
                msg.track_id = track_id
                msg.candidate_data_path = data_path
                self.pub_slot_ready.publish(msg)
                self.get_logger().info("Slot ready: job_id=%s track_id=%s candidate=%s" % (job_path.name, track_id, data_path))
            self._published_job_ids.add(job_path.name)

    def _write_object_odom_glb(self, job_path: Path, pose_data: dict):
        """Write output/<job_id>/object_odom.glb from object.glb + transform_odom_from_raw if available."""
        T_raw = pose_data.get("transform_odom_from_raw")
        scale = pose_data.get("sam3d_scale")
        rot = pose_data.get("sam3d_rotation")
        trans = pose_data.get("sam3d_translation")
        odom_pos = pose_data.get("go2_odom_position")
        odom_ori = pose_data.get("go2_odom_orientation")
        object_glb = job_path / "object.glb"
        if not object_glb.exists():
            return
        try:
            import trimesh
            import numpy as np
            T = None
            if T_raw is not None:
                T_candidate = np.asarray(T_raw, dtype=np.float64)
                if T_candidate.shape == (4, 4):
                    T = T_candidate

            has_direct_inputs = (
                scale is not None
                and rot is not None
                and trans is not None
                and odom_pos is not None
                and odom_ori is not None
            )
            if not has_direct_inputs and T is None:
                return

            scene_or_mesh = trimesh.load(str(object_glb), process=False)
            meshes = []
            if isinstance(scene_or_mesh, trimesh.Scene):
                # Flatten scene graph transforms so we transform the true GLB geometry.
                for node_name in scene_or_mesh.graph.nodes_geometry:
                    try:
                        node_tf, geom_name = scene_or_mesh.graph[node_name]
                        geom = scene_or_mesh.geometry.get(geom_name)
                        if isinstance(geom, trimesh.Trimesh):
                            g = geom.copy()
                            g.apply_transform(np.asarray(node_tf, dtype=np.float64))
                            meshes.append(g)
                    except Exception:
                        continue
            elif isinstance(scene_or_mesh, trimesh.Trimesh):
                meshes = [scene_or_mesh]
            if not meshes:
                return
            out_path = job_path / OBJECT_ODOM_GLB
            out_meshes = []
            for m in meshes:
                verts = np.asarray(m.vertices, dtype=np.float64)
                # Prefer direct demo_go2-style vertex path so object_odom.glb
                # matches worker debug behavior exactly.
                if has_direct_inputs:
                    odom = {"position": odom_pos, "orientation": odom_ori}
                    verts_pointmap = transform_mesh_vertices(
                        verts,
                        rotation=np.asarray(rot, dtype=np.float64),
                        translation=np.asarray(trans, dtype=np.float64),
                        scale=np.asarray(scale, dtype=np.float64),
                    )
                    verts_cam = verts_pointmap @ R_pytorch3d_to_cam
                    verts_new = transform_glb_to_world(verts_cam, odom, init_odom=self._init_odom)
                else:
                    # Fallback to matrix path
                    ones = np.ones((verts.shape[0], 1), dtype=np.float64)
                    verts_h = np.hstack([verts, ones])
                    verts_new = (T @ verts_h.T).T[:, :3]
                mc = m.copy()
                mc.vertices = verts_new
                if hasattr(mc, "volume") and mc.volume < 0:
                    mc.invert()
                out_meshes.append(mc)
            if len(out_meshes) == 1:
                out_meshes[0].export(str(out_path))
            else:
                s = trimesh.Scene()
                for i, m in enumerate(out_meshes):
                    s.add_geometry(m, node_name="mesh_%d" % i)
                s.export(str(out_path))
            self.get_logger().debug("Wrote %s" % out_path)
        except Exception as e:
            self.get_logger().warn("Failed writing object_odom.glb for %s: %s" % (job_path.name, e))


def main():
    rclpy.init()
    node = Sam3dInjectorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
