"""
RealSense-driven camera node that publishes the same CropImgDepthMsg and
global point cloud interface as lidar_cam_node, for use with RealSense bags
or live streams.

Subscribes to topics matching dataset_rs_recorder (aligned depth, color,
camera_info, robot pose). Depth is the rate limiter; each depth frame
triggers segmentation and CropImgDepthMsg publishing.

By default, registration uses local depth (like real2sam3d): each CropImgDepthMsg
carries the current frame's depth, and /global_lidar_points publishes the current
frame's unprojected points only (use_local_depth_for_global_pc=true). Downstream
registration and buffer then align against this frame's depth, not accumulated.
Set use_local_depth_for_global_pc:=false to accumulate depth and publish the
accumulated cloud on /global_lidar_points instead.

Tracking is optional: set use_tracking:=true and configure
config/tracking_filter.json to use a custom tracker (e.g. BoT-SORT).
"""

import time
import json
import numpy as np
import cv2
from pathlib import Path
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64

from custom_message.msg import CropImgDepthMsg

from scripts_r2u.segment_cls import Segmentation
from scripts_r2u.utils import ProjectionUtils
from scripts_r2u.utils import PointCloudBuffer


class RealsenseCamNode(Node):
    def __init__(self):
        super().__init__("realsense_cam_node")

        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("odom_topic", "/utlidar/robot_pose")
        self.declare_parameter("odom_type", "PoseStamped")  # "Odometry" or "PoseStamped"
        self.declare_parameter("camera_position_in_odom", [0.285, 0.0, 0.01])
        self.declare_parameter("use_yolo_pf", False)
        self.declare_parameter("use_tracking", False)
        default_realsense_min_depth_m = 0.2
        default_realsense_max_depth_m = 5.0
        self.declare_parameter("realsense_min_depth_m", default_realsense_min_depth_m)
        self.declare_parameter("realsense_max_depth_m", default_realsense_max_depth_m)
        # When true (default): /global_lidar_points = current frame only (local depth, like real2sam3d). When false: accumulated.
        self.declare_parameter("use_local_depth_for_global_pc", True)

        depth_topic = self.get_parameter("depth_topic").value
        image_topic = self.get_parameter("image_topic").value
        camera_info_topic = self.get_parameter("camera_info_topic").value
        odom_topic = self.get_parameter("odom_topic").value
        odom_type = self.get_parameter("odom_type").value
        T_cam = self.get_parameter("camera_position_in_odom").value
        use_yolo_pf = self.get_parameter("use_yolo_pf").value
        use_tracking = self.get_parameter("use_tracking").value
        self.realsense_min_depth_m = float(self.get_parameter("realsense_min_depth_m").value)
        self.realsense_max_depth_m = float(self.get_parameter("realsense_max_depth_m").value)
        self.use_local_depth_for_global_pc = self.get_parameter("use_local_depth_for_global_pc").value
        self.T_cam_in_odom = np.array(T_cam, dtype=np.float64)

        tracking_kwargs = {}
        if use_tracking:
            cfg = self._load_tracking_config()
            reliability_cfg = cfg.get("sensor_depth_reliability", {}) if isinstance(cfg, dict) else {}
            if (
                reliability_cfg.get("realsense_min_depth_m") is not None
                and abs(self.realsense_min_depth_m - default_realsense_min_depth_m) < 1e-9
            ):
                self.realsense_min_depth_m = float(reliability_cfg.get("realsense_min_depth_m"))
            if (
                reliability_cfg.get("realsense_max_depth_m") is not None
                and abs(self.realsense_max_depth_m - default_realsense_max_depth_m) < 1e-9
            ):
                self.realsense_max_depth_m = float(reliability_cfg.get("realsense_max_depth_m"))
            tracker_cfg = cfg.get("tracker", {}) if isinstance(cfg, dict) else {}
            if tracker_cfg.get("tracker_yaml"):
                pkg_share = Path(get_package_share_directory("real2usd"))
                tracking_kwargs["tracker"] = str((pkg_share / "config" / tracker_cfg["tracker_yaml"]).resolve())
            if tracker_cfg.get("track_conf") is not None:
                tracking_kwargs["conf"] = float(tracker_cfg["track_conf"])
            if tracker_cfg.get("track_iou") is not None:
                tracking_kwargs["iou"] = float(tracker_cfg["track_iou"])
            if tracker_cfg.get("retina_masks") is not None:
                tracking_kwargs["retina_masks"] = bool(tracker_cfg["retina_masks"])
            if tracker_cfg.get("imgsz") is not None:
                tracking_kwargs["imgsz"] = int(tracker_cfg["imgsz"])

        self.sub_depth = self.create_subscription(Image, depth_topic, self.depth_callback, 10)
        self.sub_rgb = self.create_subscription(Image, image_topic, self.rgb_callback, 10)
        self.sub_camera_info = self.create_subscription(CameraInfo, camera_info_topic, self.cam_info_callback, 10)
        if odom_type == "PoseStamped":
            self.sub_odom = self.create_subscription(PoseStamped, odom_topic, self.odom_from_pose_callback, 10)
        else:
            self.sub_odom = self.create_subscription(Odometry, odom_topic, self.odom_from_odom_callback, 10)

        self.crop_rgb_depth_pub = self.create_publisher(CropImgDepthMsg, "/usd/CropImgDepth", 10)
        self.seg_pub = self.create_publisher(Image, "/segment/image_segmented", 10)
        self.global_pcd_pub = self.create_publisher(PointCloud2, "/global_lidar_points", 10)
        self.depth_pub = self.create_publisher(Image, "/depth_image/lidar", 10)
        self.rgbd_pub = self.create_publisher(Image, "/depth_image/rgbd", 10)
        self.color_pc_pub = self.create_publisher(PointCloud2, "/segment/pointcloud_color", 10)
        self.pub_debug_last_crop = self.create_publisher(Image, "/debug/segment/last_crop", 10)
        self.timing_pub = self.create_publisher(Float64, "/timing/realsense_cam_node", 10)

        model_path = "models/yoloe-11l-seg-pf.pt" if use_yolo_pf else "models/yoloe-11l-seg.pt"
        self.segment = Segmentation(model_path, tracking_kwargs=tracking_kwargs)
        self.projection = ProjectionUtils(T=self.T_cam_in_odom)
        self.bridge = CvBridge()

        self.timer_global_pcd = self.create_timer(5.0, self.global_pcd_callback)
        self.cam_info = None
        self.rgb_image = None
        self.odom_info = {"t": None, "q": None, "eulerXYZ": None}
        self.points = PointCloudBuffer(max_points=1000000, voxel_size=0.01)
        self.points.clear()
        # When use_local_depth_for_global_pc: last frame's points (current-frame only on /global_lidar_points)
        self._last_frame_points = np.empty((0, 3))

        self.get_logger().info(
            "realsense_cam_node: depth=%s image=%s odom=%s (type=%s)"
            % (depth_topic, image_topic, odom_topic, odom_type)
        )
        self.get_logger().info(
            "Depth reliability: min=%.2fm, max=%.2fm" % (self.realsense_min_depth_m, self.realsense_max_depth_m)
        )
        self.get_logger().info(
            "/global_lidar_points: %s" % ("current frame only (local depth)" if self.use_local_depth_for_global_pc else "accumulated")
        )
        self.get_logger().info("Segmentation model: %s (use_yolo_pf=%s)" % (model_path, use_yolo_pf))
        if use_tracking:
            self.get_logger().info("Tracking config enabled: %s" % tracking_kwargs)

    def _load_tracking_config(self):
        pkg_share = Path(get_package_share_directory("real2usd"))
        cfg_path = pkg_share / "config" / "tracking_filter.json"
        try:
            with open(cfg_path) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            self.get_logger().warn("Failed to load tracking config %s: %s" % (cfg_path, e))
            return {}

    def odom_from_pose_callback(self, msg: PoseStamped):
        p = msg.pose.position
        o = msg.pose.orientation
        t = np.array([p.x, p.y, p.z], dtype=np.float64)
        q = np.array([o.x, o.y, o.z, o.w], dtype=np.float64)
        self.odom_info["t"] = t
        self.odom_info["q"] = q
        self.odom_info["eulerXYZ"] = R.from_quat(q).as_euler("xyz", degrees=True)

    def odom_from_odom_callback(self, msg: Odometry):
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        t = np.array([p.x, p.y, p.z], dtype=np.float64)
        q = np.array([o.x, o.y, o.z, o.w], dtype=np.float64)
        self.odom_info["t"] = t
        self.odom_info["q"] = q
        self.odom_info["eulerXYZ"] = R.from_quat(q).as_euler("xyz", degrees=True)

    def rgb_callback(self, msg):
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def cam_info_callback(self, msg):
        K = np.array(msg.k).reshape(3, 3)
        P = np.array(msg.p).reshape(3, 4)
        self.cam_info = {"K": K, "P": P, "width": msg.width, "height": msg.height}

    def depth_callback(self, msg: Image):
        t_frame_start = time.perf_counter()
        if self.cam_info is None or self.rgb_image is None or self.odom_info["t"] is None:
            return

        frame_stamp = msg.header.stamp
        frame_odom = {
            "t": np.array(self.odom_info["t"], dtype=np.float64),
            "q": np.array(self.odom_info["q"], dtype=np.float64),
            "eulerXYZ": np.array(self.odom_info["eulerXYZ"], dtype=np.float64) if self.odom_info.get("eulerXYZ") is not None else None,
        }
        frame_rgb = self.rgb_image.copy()
        frame_cam_info = {k: np.array(v, copy=True) if isinstance(v, np.ndarray) else v for k, v in self.cam_info.items()}

        depth_rs = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        if depth_rs is None or depth_rs.size == 0:
            return
        if depth_rs.dtype != np.uint16:
            depth_rs = np.asarray(depth_rs, dtype=np.uint16)

        depth_mm = depth_rs
        depth_msg = self.bridge.cv2_to_imgmsg(depth_mm, encoding="16UC1")
        depth_msg.header = msg.header
        self.depth_pub.publish(depth_msg)

        depth_m = depth_mm.astype(np.float64) / 1000.0
        valid_depth = (depth_m >= self.realsense_min_depth_m) & (depth_m <= self.realsense_max_depth_m)
        depth_m = np.where(valid_depth, depth_m, 0.0)
        depth_256 = (depth_m * 256.0).clip(0, 65535).astype(np.uint16)
        mask = depth_256 > 0
        if np.any(mask):
            _, color_pc_msg = self.projection.twoDtoThreeDColor(
                depth_256, frame_rgb, frame_cam_info, frame_odom
            )
            self.color_pc_pub.publish(color_pc_msg)
            v, u = np.where(depth_256 > 0)
            Z = depth_256[v, u].astype(np.float64) / 256.0
            uv1 = np.stack([u, v, np.ones_like(u, dtype=np.float64)], axis=0)
            xyz_cam = (np.linalg.inv(frame_cam_info["K"]) @ uv1) * Z
            points_cam = xyz_cam.T
            R_odom = R.from_quat(frame_odom["q"]).as_matrix()
            t_odom = frame_odom["t"]
            T_world_from_odom = np.eye(4)
            T_world_from_odom[:3, :3] = R_odom
            T_world_from_odom[:3, 3] = t_odom
            T_cam_from_world = np.linalg.inv(self.projection.T_odom_from_cam) @ np.linalg.inv(T_world_from_odom)
            T_world_from_cam = np.linalg.inv(T_cam_from_world)
            ones = np.ones((points_cam.shape[0], 1), dtype=np.float64)
            points_odom = (T_world_from_cam @ np.hstack([points_cam, ones]).T).T[:, :3]
            points_odom = points_odom[points_odom[:, 2] > 0]
            if len(points_odom) > 0:
                if self.use_local_depth_for_global_pc:
                    self._last_frame_points = np.asarray(points_odom, dtype=np.float64)
                else:
                    self.points.add_points(points_odom)

        try:
            overlay = cv2.cvtColor(frame_rgb.copy(), cv2.COLOR_RGB2BGR)
            depth_vis = cv2.normalize(depth_mm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            overlay[mask] = cv2.addWeighted(overlay[mask], 0.0, depth_color[mask], 1.0, 0)
            self.rgbd_pub.publish(self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8"))
        except Exception:
            pass

        result, _, imgs_crop, box_pts, mask_pts, track_ids, labels, labels_usd = self.segment.crop_img_w_bbox(
            frame_rgb, conf=0.5, iou=0.2
        )
        img_result_msg = self.bridge.cv2_to_imgmsg(result, encoding="bgr8")
        self.get_logger().info("labels: %s" % labels)
        self.seg_pub.publish(img_result_msg)

        dim_x = frame_rgb.shape[1]
        dim_y = frame_rgb.shape[0]
        pad = 10
        odom_msg = Odometry()
        odom_msg.header.stamp = frame_stamp
        odom_msg.header.frame_id = "odom"
        odom_msg.pose.pose.position.x = float(frame_odom["t"][0])
        odom_msg.pose.pose.position.y = float(frame_odom["t"][1])
        odom_msg.pose.pose.position.z = float(frame_odom["t"][2])
        odom_msg.pose.pose.orientation.x = float(frame_odom["q"][0])
        odom_msg.pose.pose.orientation.y = float(frame_odom["q"][1])
        odom_msg.pose.pose.orientation.z = float(frame_odom["q"][2])
        odom_msg.pose.pose.orientation.w = float(frame_odom["q"][3])

        for ii in range(len(mask_pts)):
            crop_msg = CropImgDepthMsg()
            crop_msg.header.stamp = frame_stamp
            crop_msg.header.frame_id = "odom"
            bp = box_pts[ii]
            x_min = int(np.clip(bp[1, 0] - pad, 0, dim_x - 1))
            y_min = int(np.clip(bp[1, 1] - pad, 0, dim_y - 1))
            x_max = int(np.clip(bp[4, 0] + pad, 0, dim_x - 1))
            y_max = int(np.clip(bp[4, 1] + pad, 0, dim_y - 1))
            crop_msg.crop_bbox = [x_min, y_min, x_max, y_max]
            unmasked_crop = frame_rgb[y_min:y_max, x_min:x_max]
            crop_msg.rgb_image = self.bridge.cv2_to_imgmsg(
                cv2.cvtColor(unmasked_crop, cv2.COLOR_RGB2BGR), encoding="bgr8"
            )
            crop_msg.rgb_image.header = crop_msg.header
            crop_msg.rgb_image_masked = self.bridge.cv2_to_imgmsg(imgs_crop[ii], encoding="bgr8")
            crop_msg.rgb_image_masked.header = crop_msg.header
            crop_msg.seg_points = mask_pts[ii].astype(int).flatten().tolist()
            crop_msg.depth_image = self.bridge.cv2_to_imgmsg(depth_mm, encoding="16UC1")
            crop_msg.depth_image.header = crop_msg.header
            camera_info_msg = CameraInfo()
            camera_info_msg.header = crop_msg.header
            camera_info_msg.k = frame_cam_info["K"].flatten().tolist()
            camera_info_msg.p = frame_cam_info["P"].flatten().tolist()
            camera_info_msg.width = frame_cam_info["width"]
            camera_info_msg.height = frame_cam_info["height"]
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
        frame_duration = time.perf_counter() - t_frame_start
        timing_msg = Float64()
        timing_msg.data = frame_duration
        self.timing_pub.publish(timing_msg)

    def global_pcd_callback(self):
        if self.use_local_depth_for_global_pc:
            points = self._last_frame_points
        else:
            points = self.points.get_points()
        global_pc_msg = point_cloud2.create_cloud_xyz32(
            header=Header(frame_id="odom"), points=points
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
