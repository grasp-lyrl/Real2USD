"""
SAM3D slot node (formerly injector): watches queue_dir/output for new job results.

- Normal (FAISS) mode: publishes SlotReadyMsg so the retrieval node picks the best object,
  then retrieval publishes Sam3dObjectForSlotMsg; bridge runs registration.
- No-FAISS mode (publish_object_for_slot=true): publishes Sam3dObjectForSlotMsg directly
  with the candidate object so the bridge runs registration without the retrieval node.

Does NOT import or call SAM3D — only ROS and stdlib (json, pathlib).
"""

import json
from pathlib import Path

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from std_msgs.msg import Header
from custom_message.msg import SlotReadyMsg, Sam3dObjectForSlotMsg

TOPIC_SLOT_READY = "/usd/SlotReady"
TOPIC_OBJECT_FOR_SLOT = "/usd/Sam3dObjectForSlot"


class Sam3dInjectorNode(Node):
    def __init__(self):
        super().__init__("sam3d_injector_node")

        self.declare_parameter("queue_dir", "/data/sam3d_queue")
        self.declare_parameter("watch_interval_sec", 1.0)
        self.declare_parameter("publish_object_for_slot", False)  # no-FAISS: publish directly to bridge

        self.queue_dir = Path(self.get_parameter("queue_dir").value)
        self.watch_interval_sec = self.get_parameter("watch_interval_sec").value
        p = self.get_parameter("publish_object_for_slot").value
        self.publish_object_for_slot = p is True or (isinstance(p, str) and p.lower() == "true")
        self.output_dir = self.queue_dir / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.pub_slot_ready = self.create_publisher(SlotReadyMsg, TOPIC_SLOT_READY, 10)
        self.pub_object_for_slot = self.create_publisher(Sam3dObjectForSlotMsg, TOPIC_OBJECT_FOR_SLOT, 10)
        self._published_job_ids = set()
        self._timer = self.create_timer(self.watch_interval_sec, self._check_output_dir)
        if self.publish_object_for_slot:
            self.get_logger().info(
                f"SAM3D slot node (no-FAISS): watching {self.output_dir}, publishing to {TOPIC_OBJECT_FOR_SLOT} (candidate → bridge → registration)"
            )
        else:
            self.get_logger().info(
                f"SAM3D slot node: watching {self.output_dir}, publishing to {TOPIC_SLOT_READY} (retrieval picks best object → bridge → registration)"
            )

    def _check_output_dir(self):
        if not self.output_dir.exists():
            return
        for job_path in self.output_dir.iterdir():
            if not job_path.is_dir() or job_path.name in self._published_job_ids:
                continue
            pose_path = job_path / "pose.json"
            object_ply = job_path / "object.ply"
            object_usd = job_path / "object.usd"
            if not pose_path.exists():
                continue
            object_glb = job_path / "object.glb"
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
                self.get_logger().warn(f"Invalid pose.json in {job_path}: {e}")
                continue
            track_id = pose_data.get("track_id", 0)
            if isinstance(track_id, float):
                track_id = int(track_id)

            if self.publish_object_for_slot:
                # No-FAISS mode: publish object-for-slot directly so bridge runs registration
                out = Sam3dObjectForSlotMsg()
                out.header = Header()
                out.header.stamp = self.get_clock().now().to_msg()
                out.header.frame_id = "odom"
                out.job_id = job_path.name
                out.id = track_id
                out.data_path = data_path
                out.pose = Pose()
                out.pose.position.x = 0.0
                out.pose.position.y = 0.0
                out.pose.position.z = 0.0
                out.pose.orientation.x = 0.0
                out.pose.orientation.y = 0.0
                out.pose.orientation.z = 0.0
                out.pose.orientation.w = 1.0
                self.pub_object_for_slot.publish(out)
                self.get_logger().info(f"Object for slot (no-FAISS): job_id={job_path.name} track_id={track_id} data_path={data_path}")
            else:
                msg = SlotReadyMsg()
                msg.header = Header()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = "odom"
                msg.job_id = job_path.name
                msg.track_id = track_id
                msg.candidate_data_path = data_path
                self.pub_slot_ready.publish(msg)
                self.get_logger().info(f"Slot ready: job_id={job_path.name} track_id={track_id} candidate={data_path}")
            self._published_job_ids.add(job_path.name)


def main():
    rclpy.init()
    node = Sam3dInjectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
