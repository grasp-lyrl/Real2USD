import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from sensor_msgs_py import point_cloud2
# from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
import os
import json
import numpy as np
import time
from datetime import datetime
from scripts_r2u.utils import ProjectionUtils
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from numpy.fft import fft2, fftshift
from ipdb import set_trace as st



class DatasetRecorderNode(Node):
    """Saves grouped dataset per timestep when a PointCloud2 message arrives.

    For each step (one /point_cloud2 callback) the node will create a folder
    `base_dir/step_XXXX` containing:
      - `rgb.png`       : RGB image (if available)
      - `depth.png`     : depth image from lidar projection (16-bit) (if available)
      - `points.npy`    : raw lidar points Nx3
      - `points.ply`    : optional PLY point cloud (if open3d installed)
      - `camera_info.json` : intrinsics and projection
      - `odom.json`        : odometry (position, quaternion, euler)

    The node requires camera info, rgb image, and odom to produce the depth image.
    If any of those are missing it will still save the raw points and a minimal metadata file.
    """

    def __init__(self):
        super().__init__("dataset_recorder_node")

        # ROS parameters
        self.declare_parameter("base_dir", os.path.expanduser("~/dataset"))
        self.base_dir = self.get_parameter("base_dir").value
        os.makedirs(self.base_dir, exist_ok=True)

        # internal state
        self.bridge = CvBridge()
        self.cam_info = None
        self.rgb_image = None
        self.odom_info = {"t": None, "q": None, "eulerXYZ": None}

        self.T_cam_in_odom = np.array([0.285, 0., 0.01])
        self.projection = ProjectionUtils(T=self.T_cam_in_odom)

        # sequence index initialization
        self.step_index = self._compute_next_index()
        self.last_pos = None
        self.last_time = 0.0

        # subscriptions
        self.sub_depth = self.create_subscription(Image, "/camera/camera/aligned_depth_to_color/image_raw", self._depth_cb, 10)
        self.sub_image = self.create_subscription(Image, "/camera/camera/color/image_raw", self._image_cb, 10)
        self.sub_caminfo = self.create_subscription(CameraInfo, "/camera/camera/color/camera_info", self._caminfo_cb, 10)
        self.sub_odom = self.create_subscription(PoseStamped, "/utlidar/robot_pose", self._odom_cb, 10)

        self.get_logger().info(f"Dataset recorder saving to: {self.base_dir}")

    def is_blurry(self, image, threshold=170.0):
        """
        Checks if an image is blurry based on the variance of the Laplacian.
        Lower variance suggests higher blur.
        """
        # 2. Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 3. Apply the Laplacian operator and calculate the variance
        # Laplacian measures the rate of change in the intensity.
        # High variance means many sharp edges (details), low variance means few (blur).
        laplacian_variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # You may need to tune the threshold based on your specific dataset.
        if laplacian_variance < threshold:
            # 4. Check against the threshold
            # print(f"Laplacian Variance: {laplacian_variance:.2f} vs threshold {threshold}")
            return True # Image is considered blurry
        else:
            # print(f"Laplacian Variance: {laplacian_variance:.2f} vs threshold {threshold}")
            return False # Image is considered sharp

    def has_severe_artifacts(self, image, low_freq_ratio_threshold=0.0560):
        """
        Checks for structural corruption by analyzing the FFT spectrum.
        Severely corrupted images may have a less-defined, spread-out spectrum.

        This function filters out images that are pixelated due to I believe wifi connection issues, data loss.
        tune value for low_freq_size. found 100 to seperate data better. more pixelated images have ratio of 0.55 or lower
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 2. Compute the 2D FFT
        f = fft2(image)
        fshift = fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift))
        
        # 3. Analyze the magnitude spectrum
        rows, cols = image.shape
        center_row, center_col = rows // 2, cols // 2
        
        # Define a small region around the center (low-frequency area)
        # A sharp, normal image will have most energy clustered here.
        low_freq_size = 100
        low_freq_energy = np.sum(magnitude_spectrum[
            center_row - low_freq_size:center_row + low_freq_size,
            center_col - low_freq_size:center_col + low_freq_size
        ])
        
        # Total energy in the spectrum
        total_energy = np.sum(magnitude_spectrum)
        
        # Ratio of low-frequency energy to total energy
        low_freq_ratio = low_freq_energy / total_energy
        
        # If the ratio is very low, it means the energy is NOT clustered 
        # at the center (low frequencies), suggesting significant noise/corruption.

        if low_freq_ratio < low_freq_ratio_threshold:
            print(f"Low-Frequency Energy Ratio: {low_freq_ratio:.4f} vs threshold {low_freq_ratio_threshold}")        
            return True # Image has severe structural artifacts
        else:
            return False # Image is structurally sound

    def _compute_next_index(self):
        # find highest existing step_XXXX and return next index
        existing = [p for p in os.listdir(self.base_dir) if p.startswith("step_")]
        max_idx = 0
        for name in existing:
            try:
                idx = int(name.split("_")[-1])
                max_idx = max(max_idx, idx)
            except Exception:
                continue
        return max_idx + 1

    def _image_cb(self, msg: Image):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        # store RGB as RGB (not BGR) for saving
        self.rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

    def _caminfo_cb(self, msg: CameraInfo):
        K = np.array(msg.k).reshape(3, 3)
        P = np.array(msg.p).reshape(3, 4)
        self.cam_info = {"K": K, "P": P, "width": msg.width, "height": msg.height}

    def _odom_cb(self, msg: PoseStamped):
        t = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        q = np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
        # compute euler for convenience
        euler = R.from_quat(q).as_euler("xyz", degrees=True).tolist()
        self.odom_info = {"t": t.tolist(), "q": q.tolist(), "eulerXYZ": euler}

    def _depth_cb(self, msg: Image):
        """Main driver: triggered on aligned depth from rs. Save grouped data for this timestep."""
        # Snapshot once so all data for this frame stays consistent even if callbacks overwrite self during processing.
        frame_rgb = self.rgb_image.copy() if self.rgb_image is not None else None
        frame_cam_info = {k: np.array(v, copy=True) if isinstance(v, np.ndarray) else v for k, v in self.cam_info.items()} if self.cam_info is not None else None
        frame_odom = {k: (list(v) if isinstance(v, (list, np.ndarray)) else v) for k, v in self.odom_info.items()} if self.odom_info.get("t") is not None else None

        if frame_rgb is not None:
            if self.is_blurry(frame_rgb):
                self.get_logger().info("Image is blurry, skipping this frame.")
                return
            if self.has_severe_artifacts(frame_rgb):
                self.get_logger().info("Image has severe artifacts, skipping this frame.")
                return

        # determine should_process if more than 1m translation from last processed frame or more than 10s elapsed
        if frame_odom is None:
            return
        if self.last_pos is not None:
            dist_moved = np.linalg.norm(np.array(frame_odom["t"]) - np.array(self.last_pos))
            time_elapsed = self.get_clock().now().to_msg().sec - self.last_time

            # only process if moved more than threshold or enough time has passed
            should_process = (dist_moved > 1.0) or (time_elapsed > 10.0)
        else:
            self.last_pos = list(frame_odom["t"])
            self.last_time = self.get_clock().now().to_msg().sec
            should_process = True

        if should_process:
            self.last_pos = list(frame_odom["t"])
            self.last_time = self.get_clock().now().to_msg().sec

            if frame_cam_info is None or frame_rgb is None:
                return

            # create step folder
            step_name = f"step_{self.step_index:04d}"
            step_path = os.path.join(self.base_dir, step_name)
            os.makedirs(step_path, exist_ok=True)

            # save raw points
            # points_path = os.path.join(step_path, "points.npy")
            # np.save(points_path, lidar_pts)

            # save camera info if available
            caminfo_path = os.path.join(step_path, "camera_info.json")
            if frame_cam_info is not None:
                cam_json = {
                    "K": frame_cam_info["K"].tolist(),
                    "P": frame_cam_info["P"].tolist(),
                    "width": int(frame_cam_info["width"]),
                    "height": int(frame_cam_info["height"]),
                }
                with open(caminfo_path, "w") as f:
                    json.dump(cam_json, f, indent=2)

            # save odom (frame snapshot)
            odom_path = os.path.join(step_path, "odom.json")
            with open(odom_path, "w") as f:
                json.dump(frame_odom, f, indent=2)

            # save depth image
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

            # depth_image assumed to be uint16 (16UC1) in lidar_cam_node
            depth_path = os.path.join(step_path, "depth.png")
            # ensure dtype is uint16 before saving
            if depth_image.dtype != np.uint16:
                depth_to_save = depth_image.astype(np.uint16)
            else:
                depth_to_save = depth_image

            cv2.imwrite(depth_path, depth_to_save)

            # save RGB if present (frame snapshot)
            rgb_path = os.path.join(step_path, "rgb.png")
            # our frame_rgb is RGB; convert to BGR for OpenCV
            cv2.imwrite(rgb_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))


            # minimal metadata
            meta = {
                "timestamp": self.get_clock().now().to_msg().sec,
                "step": self.step_index,
            }
            with open(os.path.join(step_path, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)

            self.get_logger().info(f"Saved step {self.step_index:04d} -> {step_path}")
            self.step_index += 1


def main():
    rclpy.init()
    node = DatasetRecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down dataset recorder")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()