import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from sensor_msgs_py import point_cloud2
from go2_interfaces.msg import Go2State, IMU
from nav_msgs.msg import Odometry
# from custom_message.msg import UsdStringIdPCMsg
from custom_message.msg import CropImgDepthMsg, PipelineStepTiming
from std_msgs.msg import Header, Int64
import cv2
from cv_bridge import CvBridge
import json, asyncio, time, pickle
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import open3d as o3d

from scripts_r2s3d.segment_cls import Segmentation
from scripts_r2s3d.utils import ProjectionUtils
from scripts_r2s3d.utils import PointCloudBuffer

"""
The idea is that lidar is the rate limiter so we use the lidar callback as the main driver callback.

The messages are based on go2_ros2_sdk repo which subscribes to webrtc topics from the unitree go2 robot directly.
"""

class LidarDriverNode(Node):
    def __init__(self):
        super().__init__("lidar_driver_node")
        self.declare_parameter("use_yolo_pf", False)
        self.declare_parameter("enable_pre_sam3d_quality_filter", False)
        self.declare_parameter("save_full_pointcloud", True)
        self.declare_parameter("pointcloud_save_period_sec", 5.0)
        self.declare_parameter("pointcloud_save_root_dir", "/data/sam3d_queue")
        self.declare_parameter("debug_verbose", False)
        default_lidar_min_range_m = 0.
        default_lidar_max_range_m = 1000.0
        self.declare_parameter("lidar_min_range_m", default_lidar_min_range_m)
        self.declare_parameter("lidar_max_range_m", default_lidar_max_range_m)
        use_yolo_pf = self.get_parameter("use_yolo_pf").value
        enable_pre_sam3d_quality_filter = self.get_parameter("enable_pre_sam3d_quality_filter").value
        self.save_full_pointcloud = bool(self.get_parameter("save_full_pointcloud").value)
        self.pointcloud_save_period_sec = float(self.get_parameter("pointcloud_save_period_sec").value)
        self.pointcloud_save_root_dir = str(self.get_parameter("pointcloud_save_root_dir").value)
        self.debug_verbose = bool(self.get_parameter("debug_verbose").value)
        self.lidar_min_range_m = float(self.get_parameter("lidar_min_range_m").value)
        self.lidar_max_range_m = float(self.get_parameter("lidar_max_range_m").value)
        self._last_pointcloud_save_t = 0.0
        self._debug_last_log_t = {}
        self._lidar_frame_count = 0
        self._rgb_msg_count = 0
        self._cam_info_msg_count = 0
        self._odom_msg_count = 0

        tracking_kwargs = {}
        if enable_pre_sam3d_quality_filter:
            cfg = self._load_tracking_filter_config()
            reliability_cfg = cfg.get("sensor_depth_reliability", {}) if isinstance(cfg, dict) else {}
            # Launch/CLI param should win over config defaults.
            if (
                reliability_cfg.get("lidar_min_range_m") is not None
                and abs(self.lidar_min_range_m - default_lidar_min_range_m) < 1e-9
            ):
                self.lidar_min_range_m = float(reliability_cfg.get("lidar_min_range_m"))
            if (
                reliability_cfg.get("lidar_max_range_m") is not None
                and abs(self.lidar_max_range_m - default_lidar_max_range_m) < 1e-9
            ):
                self.lidar_max_range_m = float(reliability_cfg.get("lidar_max_range_m"))
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

        # subscribers
        self.sub_rgb = self.create_subscription(Image, "/camera/image_raw", self.rgb_listener_callback, 10)
        self.sub_camera_info = self.create_subscription(CameraInfo, "/camera/camera_info", self.cam_info_listener_callback, 10)
        self.sub_lidar = self.create_subscription(PointCloud2, "/point_cloud2", self.lidar_listener_callback, 10) 
        self.sub_imu = self.create_subscription(IMU, "/imu", self.imu_listener_callback, 10)
        self.sub_states = self.create_subscription(Go2State, "/go2_states", self.states_listener_callback, 10)
        self.sub_odom = self.create_subscription(Odometry, "/odom", self.odom_listener_callback, 10)

        # main publishers
        self.crop_rgb_depth_pub = self.create_publisher(CropImgDepthMsg, "/usd/CropImgDepth", 10)

        # debugger publishers
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
        self.get_logger().info(f"Segmentation model: {model_path} (use_yolo_pf={use_yolo_pf})")
        if enable_pre_sam3d_quality_filter:
            self.get_logger().info(f"Tracking config enabled: {tracking_kwargs}")
        self.get_logger().info(
            f"Lidar reliability gate: min_range={self.lidar_min_range_m:.2f}m, max_range={self.lidar_max_range_m:.2f}m"
        )
        self.get_logger().info(f"Debug verbose logging: {self.debug_verbose}")
        if self.save_full_pointcloud:
            self.pointcloud_save_dir = Path(self.pointcloud_save_root_dir) / "diagnostics" / "pointclouds" / "lidar"
            self.pointcloud_save_dir.mkdir(parents=True, exist_ok=True)
            self.get_logger().info(
                f"Full pointcloud saver enabled: dir={self.pointcloud_save_dir}, period_sec={self.pointcloud_save_period_sec}"
            )
        else:
            self.pointcloud_save_dir = None

        # Unitree Go2 front camera extrinsics to odom body frame
        self.T_cam_in_odom = np.array([0.285, 0., 0.01])
        self.projection = ProjectionUtils(T=self.T_cam_in_odom)

        self.bridge = CvBridge()

        self.timer_global_pcd = self.create_timer(5.0, self.global_pcd_callback)
        self.cam_info = None
        self.rgb_image = None
        self.odom_info = {"t": None, "q": None, "eulerXYZ": None}
        self.odom_buffer = []

        # utlidar: ~8-10,000 points per second
        self.points = PointCloudBuffer(max_points=1000000, voxel_size=0.01)
        self.points.clear()

    def _load_tracking_filter_config(self):
        pkg_share = Path(get_package_share_directory("real2sam3d"))
        cfg_path = pkg_share / "config" / "tracking_pre_sam3d_filter.json"
        try:
            with open(cfg_path) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            self.get_logger().warn(f"Failed to load tracking filter config {cfg_path}: {e}")
            return {}

    def lidar_listener_callback(self, msg):
        """Extract the points from the PointCloud2 message
        return (N,3) array of lidar points from buffer
        """
        t_frame_start = time.perf_counter()
        self._lidar_frame_count += 1
        tnow = time.time()
        lidar_pts = np.asarray(
            point_cloud2.read_points_list(
                msg, field_names=("x", "y", "z"), skip_nans=True
            )
        )
        raw_count = int(lidar_pts.shape[0]) if lidar_pts.ndim == 2 else 0
        if lidar_pts.size == 0:
            self._debug_log(
                "lidar_empty",
                f"frame={self._lidar_frame_count}: lidar cloud empty",
                min_period_sec=2.0,
            )
            return
        ranges = np.linalg.norm(lidar_pts, axis=1)
        keep = (ranges >= self.lidar_min_range_m) & (ranges <= self.lidar_max_range_m)
        lidar_pts = lidar_pts[keep]
        kept_count = int(lidar_pts.shape[0]) if lidar_pts.ndim == 2 else 0
        if lidar_pts.size == 0:
            self._debug_log(
                "lidar_all_filtered",
                (
                    f"frame={self._lidar_frame_count}: all lidar points filtered by range gate "
                    f"[{self.lidar_min_range_m:.2f}, {self.lidar_max_range_m:.2f}]m; raw_count={raw_count}"
                ),
                min_period_sec=2.0,
            )
            return
        # add points to buffer
        self.points.add_points(lidar_pts)

        if lidar_pts.shape[0] > 0 and self.cam_info is not None and self.rgb_image is not None and self.odom_info["t"] is not None:
            # Process the lidar points into 2D camera frame and publish the depth image
            depth_image, depth_color, mask = self.projection.lidar2depth(lidar_pts, self.cam_info, self.odom_info)
            depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding="16UC1")
            # msg.header.stamp = lidar_msg.header.stamp
            self.depth_pub.publish(depth_msg)

            # overlay the depth image on the rgb image and publish the rgbd image
            try:
                overlay = cv2.cvtColor(self.rgb_image.copy(), cv2.COLOR_RGB2BGR)
                overlay[mask] = cv2.addWeighted(overlay[mask], 0.0, depth_color[mask], 1.0, 0)
                self.rgbd_pub.publish(self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8"))
            except:
                pass

            # publish the lidar points with color in the odom frame
            _, color_pc_msg = self.projection.twoDtoThreeDColor(depth_image, self.rgb_image, self.cam_info, self.odom_info)
            self.color_pc_pub.publish(color_pc_msg)

            # Segment the image and publish the segmentation output and tracks
            t_seg_start = time.perf_counter()
            result, _, imgs_crop, box_pts, mask_pts, track_ids, labels, labels_usd = self.segment.crop_img_w_bbox(self.rgb_image, conf=0.5, iou=0.2)
            seg_duration_ms = (time.perf_counter() - t_seg_start) * 1000.0
            self._publish_timing("lidar_cam_node", "segment", seg_duration_ms)
            self._debug_log(
                "seg_summary",
                (
                    f"frame={self._lidar_frame_count}: raw_pts={raw_count}, kept_pts={kept_count}, "
                    f"segments={len(mask_pts)}, seg_ms={seg_duration_ms:.1f}, "
                    f"rgb_msgs={self._rgb_msg_count}, cam_info_msgs={self._cam_info_msg_count}, odom_msgs={self._odom_msg_count}"
                ),
                min_period_sec=1.5,
            )
            # publish annotated segmented image
            img_result_msg = self.bridge.cv2_to_imgmsg(result, encoding="bgr8")
            self.get_logger().info(f"labels: {labels}")
            self.seg_pub.publish(img_result_msg)

            dim_x, dim_y = self.rgb_image.shape[1], self.rgb_image.shape[0]
            pad = 10  # same as segment_cls img_crop padding
            # publish the cropped image, depth image, and point cloud
            for ii in range(len(mask_pts)):
                # Create new CropImgDepthMsg
                crop_msg = CropImgDepthMsg()
                
                # Set header
                crop_msg.header.stamp = self.get_clock().now().to_msg()
                crop_msg.header.frame_id = "odom"
                
                # Crop bbox in full-image coords (same as img_crop: box + pad) for SAM3D job alignment
                bp = box_pts[ii]
                x_min = int(np.clip(bp[1, 0] - pad, 0, dim_x - 1))
                y_min = int(np.clip(bp[1, 1] - pad, 0, dim_y - 1))
                x_max = int(np.clip(bp[4, 0] + pad, 0, dim_x - 1))
                y_max = int(np.clip(bp[4, 1] + pad, 0, dim_y - 1))
                crop_msg.crop_bbox = [x_min, y_min, x_max, y_max]
                
                # Set RGB image
                crop_msg.rgb_image = self.bridge.cv2_to_imgmsg(imgs_crop[ii], encoding="bgr8")
                crop_msg.rgb_image.header = crop_msg.header
                
                # Set segmentation points as flat array
                crop_msg.seg_points = mask_pts[ii].astype(int).flatten().tolist()
                
                # Set depth image
                crop_msg.depth_image = self.bridge.cv2_to_imgmsg(depth_image, encoding="16UC1")
                crop_msg.depth_image.header = crop_msg.header
                
                # Set camera info
                camera_info_msg = CameraInfo()
                camera_info_msg.header = crop_msg.header
                camera_info_msg.k = self.cam_info["K"].flatten().tolist()
                camera_info_msg.p = self.cam_info["P"].flatten().tolist()
                camera_info_msg.width = self.cam_info["width"]
                camera_info_msg.height = self.cam_info["height"]
                crop_msg.camera_info = camera_info_msg
                
                # Set odometry info
                odom_msg = Odometry()
                odom_msg.header = crop_msg.header
                odom_msg.pose.pose.position.x = self.odom_info["t"][0]
                odom_msg.pose.pose.position.y = self.odom_info["t"][1]
                odom_msg.pose.pose.position.z = self.odom_info["t"][2]
                odom_msg.pose.pose.orientation.x = self.odom_info["q"][0]
                odom_msg.pose.pose.orientation.y = self.odom_info["q"][1]
                odom_msg.pose.pose.orientation.z = self.odom_info["q"][2]
                odom_msg.pose.pose.orientation.w = self.odom_info["q"][3]
                crop_msg.odometry = odom_msg
                
                # Set track ID and label
                crop_msg.track_id = track_ids[ii]
                crop_msg.label = labels_usd[ii]
                
                # Publish the message
                self.crop_rgb_depth_pub.publish(crop_msg)
            if imgs_crop:
                self.pub_debug_last_crop.publish(
                    self.bridge.cv2_to_imgmsg(imgs_crop[-1], encoding="bgr8")
                )
            frame_duration_ms = (time.perf_counter() - t_frame_start) * 1000.0
            self._publish_timing("lidar_cam_node", "frame_total", frame_duration_ms)
        else:
            missing = []
            if self.cam_info is None:
                missing.append("cam_info")
            if self.rgb_image is None:
                missing.append("rgb_image")
            if self.odom_info["t"] is None:
                missing.append("odom")
            self._debug_log(
                "waiting_inputs",
                (
                    f"frame={self._lidar_frame_count}: waiting for {','.join(missing)}; "
                    f"raw_pts={raw_count}, kept_pts={kept_count}, "
                    f"rgb_msgs={self._rgb_msg_count}, cam_info_msgs={self._cam_info_msg_count}, odom_msgs={self._odom_msg_count}"
                ),
                min_period_sec=1.5,
            )

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

    def rgb_listener_callback(self, msg):
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self._rgb_msg_count += 1

    def global_pcd_callback(self):
        global_pc_msg = point_cloud2.create_cloud_xyz32(header=Header(frame_id="odom"), points=self.points.get_points())
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
        out_path = self.pointcloud_save_dir / f"lidar_{reason}_{stamp_ns}.npy"
        np.save(str(out_path), pts)

    def destroy_node(self):
        try:
            self._save_full_pointcloud_snapshot("shutdown")
        except Exception as e:
            self.get_logger().warn(f"Failed saving shutdown lidar pointcloud snapshot: {e}")
        return super().destroy_node()

    def imu_listener_callback(self, msg):
        pass

    def states_listener_callback(self, msg):
        pass

    def odom_listener_callback(self, msg):
        """
        odom msg contains position (x,y,z) and orientation (quaternion, x,y,z,w)
        """
        t = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
        q = np.array([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
        eulerXYZ = R.from_quat(q).as_euler("xyz", degrees=True)
        self.odom_info["t"] = t
        self.odom_info["q"] = q
        self.odom_info["eulerXYZ"] = eulerXYZ
        self.odom_buffer.append({"t":t.copy(), "q":q.copy(), "eulerXYZ":eulerXYZ.copy()})
        self._odom_msg_count += 1

        #save the odom buffer as a pickle file
        # if len(self.odom_buffer) % 100 == 0:
        #     with open("odom_buffer.pkl", "wb") as f:
        #         pickle.dump(self.odom_buffer, f)

    def cam_info_listener_callback(self, msg):
        try:
            # sensor_msgs/CameraInfo arrays are plain sequences; convert before reshape.
            K = np.array(msg.k, dtype=np.float64).reshape(3, 3)  # intrinsics
            Rm = np.array(msg.r, dtype=np.float64).reshape(3, 3)  # rectification
            P = np.array(msg.p, dtype=np.float64).reshape(3, 4)  # projection matrix
            cam_info = {"K": K, "R": Rm, "P": P, "width": int(msg.width), "height": int(msg.height)}
            self.cam_info = cam_info
            self._cam_info_msg_count += 1
            self._debug_log(
                "cam_info_ok",
                f"camera_info parsed: width={msg.width}, height={msg.height}",
                min_period_sec=5.0,
            )
        except Exception as e:
            self.get_logger().warn(f"Failed parsing camera_info: {e}")

    def _debug_log(self, key: str, message: str, min_period_sec: float = 1.0):
        if not self.debug_verbose:
            return
        now = time.time()
        last = self._debug_last_log_t.get(key, 0.0)
        if (now - last) >= min_period_sec:
            self._debug_last_log_t[key] = now
            self.get_logger().info(f"[debug] {message}")
    

def main():
    rclpy.init()
    lidar_driver_node = LidarDriverNode()
    rclpy.spin(lidar_driver_node)
    lidar_driver_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
