"""
RealSense2-driven camera node that publishes the same CropImgDepthMsg and
global point cloud interface as lidar_cam_node, for use with RealSense bags
or live streams.

Subscribes to topics matching dataset_rs_recorder_node (aligned depth, color,
camera_info, robot pose). Does not filter frames (no blur/artifact checks).
Depth is used as the rate limiter; each depth frame triggers segmentation
and CropImgDepthMsg publishing. Global point cloud is built by unprojecting
depth into odom frame and accumulating in a buffer.
"""

import time
import numpy as np
import cv2
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry

from custom_message.msg import CropImgDepthMsg, PipelineStepTiming

from scripts_r2s3d.segment_cls import Segmentation
from scripts_r2s3d.utils import ProjectionUtils
from scripts_r2s3d.utils import PointCloudBuffer


class RealsenseCamNode(Node):
    def __init__(self):
        super().__init__("realsense_cam_node")

        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("odom_topic", "/utlidar/robot_pose")
        self.declare_parameter("camera_position_in_odom", [0.285, 0.0, 0.01])  # same default as Go2

        depth_topic = self.get_parameter("depth_topic").value
        image_topic = self.get_parameter("image_topic").value
        camera_info_topic = self.get_parameter("camera_info_topic").value
        odom_topic = self.get_parameter("odom_topic").value
        T_cam = self.get_parameter("camera_position_in_odom").value
        self.T_cam_in_odom = np.array(T_cam, dtype=np.float64)

        self.sub_depth = self.create_subscription(Image, depth_topic, self.depth_callback, 10)
        self.sub_rgb = self.create_subscription(Image, image_topic, self.rgb_callback, 10)
        self.sub_camera_info = self.create_subscription(CameraInfo, camera_info_topic, self.cam_info_callback, 10)
        self.sub_odom = self.create_subscription(PoseStamped, odom_topic, self.odom_callback, 10)

        self.crop_rgb_depth_pub = self.create_publisher(CropImgDepthMsg, "/usd/CropImgDepth", 10)
        self.seg_pub = self.create_publisher(Image, "/segment/image_segmented", 10)
        self.global_pcd_pub = self.create_publisher(PointCloud2, "/global_lidar_points", 10)
        self.depth_pub = self.create_publisher(Image, "/depth_image/lidar", 10)
        self.rgbd_pub = self.create_publisher(Image, "/depth_image/rgbd", 10)
        self.color_pc_pub = self.create_publisher(PointCloud2, "/segment/pointcloud_color", 10)
        self.pub_debug_last_crop = self.create_publisher(Image, "/debug/segment/last_crop", 10)
        self.pub_timing = self.create_publisher(PipelineStepTiming, "/pipeline/timings", 10)
        self._timing_sequence = 0

        model_path = "models/yoloe-11l-seg.pt"
        self.segment = Segmentation(model_path)
        self.projection = ProjectionUtils(T=self.T_cam_in_odom)
        self.bridge = CvBridge()

        self.timer_global_pcd = self.create_timer(5.0, self.global_pcd_callback)
        self.cam_info = None
        self.rgb_image = None
        self.odom_info = {"t": None, "q": None, "eulerXYZ": None}
        self.points = PointCloudBuffer(max_points=1000000, voxel_size=0.01)
        self.points.clear()

        self.get_logger().info(
            "realsense_cam_node: depth=%s image=%s odom=%s"
            % (depth_topic, image_topic, odom_topic)
        )

    def rgb_callback(self, msg):
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def cam_info_callback(self, msg):
        K = np.array(msg.k).reshape(3, 3)
        P = np.array(msg.p).reshape(3, 4)
        self.cam_info = {"K": K, "P": P, "width": msg.width, "height": msg.height}

    def odom_callback(self, msg: PoseStamped):
        p = msg.pose.position
        o = msg.pose.orientation
        t = np.array([p.x, p.y, p.z], dtype=np.float64)
        q = np.array([o.x, o.y, o.z, o.w], dtype=np.float64)
        self.odom_info["t"] = t
        self.odom_info["q"] = q
        self.odom_info["eulerXYZ"] = R.from_quat(q).as_euler("xyz", degrees=True)

    def depth_callback(self, msg: Image):
        t_frame_start = time.perf_counter()
        if self.cam_info is None or self.rgb_image is None or self.odom_info["t"] is None:
            return

        depth_rs = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        if depth_rs is None or depth_rs.size == 0:
            return
        if depth_rs.dtype != np.uint16:
            depth_rs = np.asarray(depth_rs, dtype=np.uint16)

        # RealSense aligned depth is typically mm; job writer expects depth image in mm (saves /1000 for depth.npy)
        depth_mm = depth_rs
        depth_msg = self.bridge.cv2_to_imgmsg(depth_mm, encoding="16UC1")
        depth_msg.header = msg.header
        self.depth_pub.publish(depth_msg)

        # Build depth in "lidar scale" (meters*256) for ProjectionUtils unproject
        depth_m = depth_mm.astype(np.float64) / 1000.0
        depth_256 = (depth_m * 256.0).clip(0, 65535).astype(np.uint16)
        mask = depth_256 > 0
        if np.any(mask):
            _, color_pc_msg = self.projection.twoDtoThreeDColor(
                depth_256, self.rgb_image, self.cam_info, self.odom_info
            )
            self.color_pc_pub.publish(color_pc_msg)
            # Accumulate points in odom for global cloud (same convention as lidar_cam)
            v, u = np.where(depth_256 > 0)
            Z = depth_256[v, u].astype(np.float64) / 256.0
            uv1 = np.stack([u, v, np.ones_like(u, dtype=np.float64)], axis=0)
            xyz_cam = (np.linalg.inv(self.cam_info["K"]) @ uv1) * Z
            points_cam = xyz_cam.T
            R_odom = R.from_quat(self.odom_info["q"]).as_matrix()
            t_odom = self.odom_info["t"]
            T_world_from_odom = np.eye(4)
            T_world_from_odom[:3, :3] = R_odom
            T_world_from_odom[:3, 3] = t_odom
            T_cam_from_world = np.linalg.inv(self.projection.T_odom_from_cam) @ np.linalg.inv(T_world_from_odom)
            T_world_from_cam = np.linalg.inv(T_cam_from_world)
            ones = np.ones((points_cam.shape[0], 1), dtype=np.float64)
            points_odom = (T_world_from_cam @ np.hstack([points_cam, ones]).T).T[:, :3]
            points_odom = points_odom[points_odom[:, 2] > 0]
            if len(points_odom) > 0:
                self.points.add_points(points_odom)

        try:
            overlay = cv2.cvtColor(self.rgb_image.copy(), cv2.COLOR_RGB2BGR)
            depth_vis = cv2.normalize(depth_mm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            overlay[mask] = cv2.addWeighted(overlay[mask], 0.0, depth_color[mask], 1.0, 0)
            self.rgbd_pub.publish(self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8"))
        except Exception:
            pass

        t_seg_start = time.perf_counter()
        result, _, imgs_crop, box_pts, mask_pts, track_ids, labels, labels_usd = self.segment.crop_img_w_bbox(
            self.rgb_image, conf=0.5, iou=0.2
        )
        seg_duration_ms = (time.perf_counter() - t_seg_start) * 1000.0
        self._publish_timing("realsense_cam_node", "segment", seg_duration_ms)

        img_result_msg = self.bridge.cv2_to_imgmsg(result, encoding="bgr8")
        self.get_logger().info("labels: %s" % labels)
        self.seg_pub.publish(img_result_msg)

        dim_x = self.rgb_image.shape[1]
        dim_y = self.rgb_image.shape[0]
        pad = 10
        odom_msg = Odometry()
        odom_msg.pose.pose.position.x = float(self.odom_info["t"][0])
        odom_msg.pose.pose.position.y = float(self.odom_info["t"][1])
        odom_msg.pose.pose.position.z = float(self.odom_info["t"][2])
        odom_msg.pose.pose.orientation.x = float(self.odom_info["q"][0])
        odom_msg.pose.pose.orientation.y = float(self.odom_info["q"][1])
        odom_msg.pose.pose.orientation.z = float(self.odom_info["q"][2])
        odom_msg.pose.pose.orientation.w = float(self.odom_info["q"][3])

        for ii in range(len(mask_pts)):
            crop_msg = CropImgDepthMsg()
            crop_msg.header.stamp = self.get_clock().now().to_msg()
            crop_msg.header.frame_id = "odom"
            bp = box_pts[ii]
            x_min = int(np.clip(bp[1, 0] - pad, 0, dim_x - 1))
            y_min = int(np.clip(bp[1, 1] - pad, 0, dim_y - 1))
            x_max = int(np.clip(bp[4, 0] + pad, 0, dim_x - 1))
            y_max = int(np.clip(bp[4, 1] + pad, 0, dim_y - 1))
            crop_msg.crop_bbox = [x_min, y_min, x_max, y_max]
            crop_msg.rgb_image = self.bridge.cv2_to_imgmsg(imgs_crop[ii], encoding="bgr8")
            crop_msg.rgb_image.header = crop_msg.header
            crop_msg.seg_points = mask_pts[ii].astype(int).flatten().tolist()
            crop_msg.depth_image = self.bridge.cv2_to_imgmsg(depth_mm, encoding="16UC1")
            crop_msg.depth_image.header = crop_msg.header
            camera_info_msg = CameraInfo()
            camera_info_msg.header = crop_msg.header
            camera_info_msg.k = self.cam_info["K"].flatten().tolist()
            camera_info_msg.p = self.cam_info["P"].flatten().tolist()
            camera_info_msg.width = self.cam_info["width"]
            camera_info_msg.height = self.cam_info["height"]
            crop_msg.camera_info = camera_info_msg
            crop_msg.odometry = odom_msg
            crop_msg.odometry.header = crop_msg.header
            crop_msg.track_id = track_ids[ii]
            crop_msg.label = labels_usd[ii]
            self.crop_rgb_depth_pub.publish(crop_msg)

        if imgs_crop:
            self.pub_debug_last_crop.publish(
                self.bridge.cv2_to_imgmsg(imgs_crop[-1], encoding="bgr8")
            )
        frame_duration_ms = (time.perf_counter() - t_frame_start) * 1000.0
        self._publish_timing("realsense_cam_node", "frame_total", frame_duration_ms)

    def _publish_timing(self, node_name: str, step_name: str, duration_ms: float):
        msg = PipelineStepTiming()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"
        msg.node_name = node_name
        msg.step_name = step_name
        msg.duration_ms = duration_ms
        msg.sequence_id = self._timing_sequence
        self._timing_sequence += 1
        self.pub_timing.publish(msg)

    def global_pcd_callback(self):
        global_pc_msg = point_cloud2.create_cloud_xyz32(
            header=Header(frame_id="odom"), points=self.points.get_points()
        )
        global_pc_msg.header.stamp = self.get_clock().now().to_msg()
        self.global_pcd_pub.publish(global_pc_msg)


def main():
    rclpy.init()
    node = RealsenseCamNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
