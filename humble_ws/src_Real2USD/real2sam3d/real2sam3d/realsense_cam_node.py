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
        self.declare_parameter("use_yolo_pf", False)
        self.declare_parameter("enable_pre_sam3d_quality_filter", False)
        self.declare_parameter("save_full_pointcloud", True)
        self.declare_parameter("pointcloud_save_period_sec", 5.0)
        self.declare_parameter("pointcloud_save_root_dir", "/data/sam3d_queue")
        default_realsense_min_depth_m = 0.2
        default_realsense_max_depth_m = 4.0
        self.declare_parameter("realsense_min_depth_m", default_realsense_min_depth_m)
        self.declare_parameter("realsense_max_depth_m", default_realsense_max_depth_m)
        self.declare_parameter("realsense_to_lidar_transform_json", "")
        self.declare_parameter("realsense_to_lidar_transform", [])

        depth_topic = self.get_parameter("depth_topic").value
        image_topic = self.get_parameter("image_topic").value
        camera_info_topic = self.get_parameter("camera_info_topic").value
        odom_topic = self.get_parameter("odom_topic").value
        T_cam = self.get_parameter("camera_position_in_odom").value
        use_yolo_pf = self.get_parameter("use_yolo_pf").value
        enable_pre_sam3d_quality_filter = self.get_parameter("enable_pre_sam3d_quality_filter").value
        self.save_full_pointcloud = bool(self.get_parameter("save_full_pointcloud").value)
        self.pointcloud_save_period_sec = float(self.get_parameter("pointcloud_save_period_sec").value)
        self.pointcloud_save_root_dir = str(self.get_parameter("pointcloud_save_root_dir").value)
        self.realsense_min_depth_m = float(self.get_parameter("realsense_min_depth_m").value)
        self.realsense_max_depth_m = float(self.get_parameter("realsense_max_depth_m").value)
        self.T_cam_in_odom = np.array(T_cam, dtype=np.float64)
        self._last_pointcloud_save_t = 0.0
        self.T_lidar_from_realsense = self._load_realsense_to_lidar_transform()

        tracking_kwargs = {}
        if enable_pre_sam3d_quality_filter:
            cfg = self._load_tracking_filter_config()
            reliability_cfg = cfg.get("sensor_depth_reliability", {}) if isinstance(cfg, dict) else {}
            # Launch/CLI param should win over config defaults.
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
            tracker_yaml = tracker_cfg.get("tracker_yaml")
            if tracker_yaml:
                pkg_share = Path(get_package_share_directory("real2sam3d"))
                tracking_kwargs["tracker"] = str((pkg_share / "config" / tracker_yaml).resolve())
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

        self.get_logger().info(
            "realsense_cam_node: depth=%s image=%s odom=%s"
            % (depth_topic, image_topic, odom_topic)
        )
        self.get_logger().info(
            f"Depth reliability gate: min={self.realsense_min_depth_m:.2f}m, max={self.realsense_max_depth_m:.2f}m"
        )
        if self.T_lidar_from_realsense is not None:
            t = self.T_lidar_from_realsense[:3, 3]
            self.get_logger().info(
                f"RealSense→lidar correction enabled: translation=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}] m"
            )
        self.get_logger().info(f"Segmentation model: {model_path} (use_yolo_pf={use_yolo_pf})")
        if enable_pre_sam3d_quality_filter:
            self.get_logger().info(f"Tracking config enabled: {tracking_kwargs}")
        if self.save_full_pointcloud:
            self.pointcloud_save_dir = (
                Path(self.pointcloud_save_root_dir) / "diagnostics" / "pointclouds" / "realsense"
            )
            self.pointcloud_save_dir.mkdir(parents=True, exist_ok=True)
            self.get_logger().info(
                f"Full pointcloud saver enabled: dir={self.pointcloud_save_dir}, period_sec={self.pointcloud_save_period_sec}"
            )
        else:
            self.pointcloud_save_dir = None

    def _load_tracking_filter_config(self):
        pkg_share = Path(get_package_share_directory("real2sam3d"))
        cfg_path = pkg_share / "config" / "tracking_pre_sam3d_filter.json"
        try:
            with open(cfg_path) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            self.get_logger().warn(f"Failed to load tracking filter config {cfg_path}: {e}")
            return {}

    def _load_realsense_to_lidar_transform(self):
        """Load 4x4 T_lidar_from_realsense from JSON path or from 16-float list. Applied to RealSense points in odom so they align with lidar."""
        json_path = self.get_parameter("realsense_to_lidar_transform_json").value
        list_param = self.get_parameter("realsense_to_lidar_transform").value
        if json_path and Path(json_path).is_file():
            try:
                with open(json_path) as f:
                    data = json.load(f)
                matrix = data.get("transform_lidar_from_realsense", {}).get("matrix_4x4")
                if matrix is None:
                    matrix = data.get("transform_lidar_from_realsense")
                if isinstance(matrix, list) and len(matrix) == 4 and all(len(row) == 4 for row in matrix):
                    T = np.array(matrix, dtype=np.float64)
                    return T
            except (OSError, json.JSONDecodeError, TypeError) as e:
                self.get_logger().warn(f"Failed to load realsense_to_lidar transform from {json_path}: {e}")
        if isinstance(list_param, (list, tuple)) and len(list_param) == 16:
            try:
                T = np.array(list_param, dtype=np.float64).reshape(4, 4)
                return T
            except (ValueError, TypeError):
                pass
        return None

    def _apply_lidar_correction(self, points_odom: np.ndarray) -> np.ndarray:
        """Transform RealSense odom points to lidar-aligned odom if correction is set."""
        if self.T_lidar_from_realsense is None or points_odom.size == 0:
            return points_odom
        ones = np.ones((points_odom.shape[0], 1), dtype=np.float64)
        hom = np.hstack([points_odom, ones])
        return (self.T_lidar_from_realsense @ hom.T).T[:, :3]

    def _transform_pointcloud_xyz(self, msg: PointCloud2) -> PointCloud2:
        """Apply T_lidar_from_realsense to x,y,z of a PointCloud2 (e.g. x,y,z,rgb); preserve other fields."""
        if self.T_lidar_from_realsense is None:
            return msg
        from sensor_msgs_py import point_cloud2 as pc2
        pts = np.array(pc2.read_points_list(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True))
        if pts.size == 0:
            return msg
        xyz = pts[:, :3].astype(np.float64)
        rgb = pts[:, 3].astype(np.float32).reshape(-1, 1)
        xyz_corr = self._apply_lidar_correction(xyz)
        data = np.hstack([xyz_corr.astype(np.float32), rgb])
        return pc2.create_cloud(msg.header, self.projection.fields, data)

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

        # Capture odom and stamp once for this frame so everything (depth unproject, accumulation, all CropImgDepth) uses the same pose. self.odom_info can be overwritten by odom_callback during the loop.
        frame_stamp = msg.header.stamp
        frame_odom = {
            "t": np.array(self.odom_info["t"], dtype=np.float64),
            "q": np.array(self.odom_info["q"], dtype=np.float64),
            "eulerXYZ": np.array(self.odom_info["eulerXYZ"], dtype=np.float64) if self.odom_info.get("eulerXYZ") is not None else None,
        }
        # Snapshot rgb and cam_info so they stay aligned with this depth frame even if callbacks overwrite self during slow processing.
        frame_rgb = self.rgb_image.copy()
        frame_cam_info = {k: np.array(v, copy=True) if isinstance(v, np.ndarray) else v for k, v in self.cam_info.items()}

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
        valid_depth = (depth_m >= self.realsense_min_depth_m) & (depth_m <= self.realsense_max_depth_m)
        depth_m = np.where(valid_depth, depth_m, 0.0)
        depth_256 = (depth_m * 256.0).clip(0, 65535).astype(np.uint16)
        mask = depth_256 > 0
        if np.any(mask):
            _, color_pc_msg = self.projection.twoDtoThreeDColor(
                depth_256, frame_rgb, frame_cam_info, frame_odom
            )
            if self.T_lidar_from_realsense is not None:
                color_pc_msg = self._transform_pointcloud_xyz(color_pc_msg)
            self.color_pc_pub.publish(color_pc_msg)
            # Accumulate points in odom for global cloud (same convention as lidar_cam)
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
            points_odom = self._apply_lidar_correction(points_odom)
            if len(points_odom) > 0:
                self.points.add_points(points_odom)

        try:
            overlay = cv2.cvtColor(frame_rgb.copy(), cv2.COLOR_RGB2BGR)
            depth_vis = cv2.normalize(depth_mm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            overlay[mask] = cv2.addWeighted(overlay[mask], 0.0, depth_color[mask], 1.0, 0)
            self.rgbd_pub.publish(self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8"))
        except Exception:
            pass

        t_seg_start = time.perf_counter()
        result, _, imgs_crop, box_pts, mask_pts, track_ids, labels, labels_usd = self.segment.crop_img_w_bbox(
            frame_rgb, conf=0.5, iou=0.2
        )
        seg_duration_ms = (time.perf_counter() - t_seg_start) * 1000.0
        self._publish_timing("realsense_cam_node", "segment", seg_duration_ms)

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
            crop_msg.rgb_image = self.bridge.cv2_to_imgmsg(imgs_crop[ii], encoding="bgr8")
            crop_msg.rgb_image.header = crop_msg.header
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
        self._maybe_save_pointcloud_periodic()

    def _maybe_save_pointcloud_periodic(self):
        if not self.save_full_pointcloud:
            return
        now = time.time()
        if (now - self._last_pointcloud_save_t) < max(0.1, self.pointcloud_save_period_sec):
            return
        self._save_full_pointcloud_snapshot("periodic")
        self._last_pointcloud_save_t = now

    def _save_full_pointcloud_snapshot(self, reason: str):
        if not self.save_full_pointcloud or self.pointcloud_save_dir is None:
            return
        pts = np.asarray(self.points.get_points(), dtype=np.float32)
        if pts.size == 0:
            return
        stamp_ns = int(self.get_clock().now().nanoseconds)
        out_path = self.pointcloud_save_dir / f"realsense_{reason}_{stamp_ns}.npy"
        np.save(str(out_path), pts)

    def destroy_node(self):
        try:
            self._save_full_pointcloud_snapshot("shutdown")
        except Exception as e:
            self.get_logger().warn(f"Failed saving shutdown realsense pointcloud snapshot: {e}")
        return super().destroy_node()


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
