"""
SAM3D injector node: watches queue_dir/output for new job results (pose.json +
object.ply or object.usd), publishes UsdStringIdPoseMsg on /usd/StringIdPose so
usd_buffer_node and downstream see generated objects. Run via ros2 launch.

Does NOT import or call SAM3D — only ROS and stdlib (json, pathlib). It just
reads files on disk and publishes; works without SAM3D installed. The worker
(when run with --dry-run) writes placeholder object.ply + pose.json, which the
injector picks up the same way as real SAM3D output.
"""

import json
from pathlib import Path

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from std_msgs.msg import Header
from custom_message.msg import UsdStringIdPoseMsg


class Sam3dInjectorNode(Node):
    def __init__(self):
        super().__init__("sam3d_injector_node")

        self.declare_parameter("queue_dir", "/data/sam3d_queue")
        self.declare_parameter("watch_interval_sec", 1.0)

        self.queue_dir = Path(self.get_parameter("queue_dir").value)
        self.watch_interval_sec = self.get_parameter("watch_interval_sec").value
        self.output_dir = self.queue_dir / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.pub = self.create_publisher(
            UsdStringIdPoseMsg,
            "/usd/StringIdPose",
            10,
        )
        self._published_job_ids = set()
        self._timer = self.create_timer(self.watch_interval_sec, self._check_output_dir)
        self.get_logger().info(
            f"SAM3D injector: watching {self.output_dir}, publishing to /usd/StringIdPose"
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
            data_path = None
            if object_usd.exists():
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
            position = pose_data.get("position", [0.0, 0.0, 0.0])
            orientation = pose_data.get("orientation", [0.0, 0.0, 0.0, 1.0])
            track_id = pose_data.get("track_id", 0)
            if isinstance(track_id, float):
                track_id = int(track_id)

            msg = UsdStringIdPoseMsg()
            msg.header = Header()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "odom"
            msg.data_path = data_path
            msg.id = track_id
            msg.pose = Pose()
            msg.pose.position.x = float(position[0])
            msg.pose.position.y = float(position[1])
            msg.pose.position.z = float(position[2])
            msg.pose.orientation.x = float(orientation[0])
            msg.pose.orientation.y = float(orientation[1])
            msg.pose.orientation.z = float(orientation[2])
            msg.pose.orientation.w = float(orientation[3])
            self.pub.publish(msg)
            self._published_job_ids.add(job_path.name)
            self.get_logger().info(f"Injected generated object: {data_path} (track_id={track_id})")


def main():
    rclpy.init()
    node = Sam3dInjectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
