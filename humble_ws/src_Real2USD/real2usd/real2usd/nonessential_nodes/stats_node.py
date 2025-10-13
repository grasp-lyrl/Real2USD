import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from sensor_msgs_py import point_cloud2
from go2_interfaces.msg import Go2State, IMU
from nav_msgs.msg import Odometry
from custom_message.msg import UsdStringIdPCMsg
from std_msgs.msg import Header, Int64
import cv2
from cv_bridge import CvBridge
import json, asyncio, time, os
import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import matplotlib.pyplot as plt

from scripts_r2u.utils import ProjectionUtils

from ipdb import set_trace as st

"""
This node is used to collect statistics about the node and its messages, e.g. frequency of messages, timestamp of messages, etc.
"""

class CollectStatsNode(Node):
    def __init__(self):
        super().__init__("collect_stats")

        # Subscribe to incoming point cloud
        self.sub_rgb = self.create_subscription(Image, "/camera/image_raw", self.rgb_listener_callback, 10)
        self.sub_camera_info = self.create_subscription(CameraInfo, "/camera/camera_info", self.cam_info_listener_callback, 10)
        self.sub_lidar = self.create_subscription(PointCloud2, "/point_cloud2", self.lidar_listener_callback, 10) 
        self.sub_odom = self.create_subscription(Odometry, "/odom", self.odom_listener_callback, 10)
        
        self.T_cam_in_odom = np.array([0.285, 0., 0.01])
        self.projection = ProjectionUtils(T=self.T_cam_in_odom)
        
        self.timer = self.create_timer(0.001, self.timer_callback)
        self.depth_timer = self.create_timer(0.001, self.depth_callback)

        # Initialize message timing data structures using lists instead of deques
        self.message_timestamps = {
            'rgb': [],  # Store all timestamps
            'camera_info': [],
            'lidar': [],
            'odom': [],
            'depth': []
        }
        
        # Initialize message data
        self.rgb_image = None
        self.cam_info = None
        self.lidar_points = None
        self.odom_info = {"t": None, "q": None, "eulerXYZ": None}
        self.depth_image = None
        
        # Previous message data for comparison
        self.prev_rgb_image = None
        self.prev_cam_info = None
        self.prev_lidar_points = None
        self.prev_odom_info = {"t": None, "q": None, "eulerXYZ": None}
        self.prev_depth_image = None

        # Initialize bridge for image conversion
        self.bridge = CvBridge()
        
        # Create figures for both plots with proper height ratios
        self.fig, (self.ax_freq, self.ax_timeline) = plt.subplots(
            2, 1, 
            figsize=(12, 8),
            gridspec_kw={'height_ratios': [1, 2]}
        )
        plt.ion()  # Enable interactive mode
        
        # Set up timeline plot
        self.ax_timeline.set_ylabel('Message Type')
        self.ax_timeline.set_xlabel('Time (seconds)')
        self.ax_timeline.set_title('Message Timeline')
        self.ax_timeline.grid(True, alpha=0.3)
        
        # Store start time for timeline
        self.start_time = time.time()
        
        # Colors for different message types
        self.colors = {
            'rgb': 'red',
            'camera_info': 'blue',
            'lidar': 'green',
            'odom': 'purple',
            'depth': 'orange'
        }

    def calculate_frequency(self, timestamps):
        """Calculate frequency from a list of timestamps"""
        if len(timestamps) < 2:
            return 0.0
        # Calculate frequency using the last 100 messages (or all if less than 100)
        recent_timestamps = timestamps[-100:] if len(timestamps) > 100 else timestamps
        time_diffs = np.diff(recent_timestamps)
        return 1.0 / np.mean(time_diffs) if len(time_diffs) > 0 else 0.0

    def timer_callback(self):
        """
        Log message arrival times and calculate frequencies
        """
        current_time = time.time()
        
        # Check for new messages and log timestamps
        # For RGB image, we can use identity check since it's a new object each time
        if self.rgb_image is not None and self.rgb_image is not self.prev_rgb_image:
            self.message_timestamps['rgb'].append(current_time)
            self.rgb_image = None
            
        if self.cam_info is not None:
            self.message_timestamps['camera_info'].append(current_time)
            
        # For lidar points, compare the actual point cloud data
        if self.lidar_points is not None and (
            self.prev_lidar_points is None or 
            not np.array_equal(self.lidar_points, self.prev_lidar_points)
        ):
            self.message_timestamps['lidar'].append(current_time)
            self.prev_lidar_points = self.lidar_points.copy()
            
        # For odom info, compare each component (position, quaternion, euler angles)
        if self.odom_info["t"] is not None and (
            self.prev_odom_info["t"] is None or
            not np.array_equal(self.odom_info["t"], self.prev_odom_info["t"]) or
            not np.array_equal(self.odom_info["q"], self.prev_odom_info["q"]) or
            not np.array_equal(self.odom_info["eulerXYZ"], self.prev_odom_info["eulerXYZ"])
        ):
            self.message_timestamps['odom'].append(current_time)
            self.prev_odom_info = {
                "t": self.odom_info["t"].copy() if self.odom_info["t"] is not None else None,
                "q": self.odom_info["q"].copy() if self.odom_info["q"] is not None else None,
                "eulerXYZ": self.odom_info["eulerXYZ"].copy() if self.odom_info["eulerXYZ"] is not None else None
            }

        # Calculate frequencies
        frequencies = {
            topic: self.calculate_frequency(timestamps)
            for topic, timestamps in self.message_timestamps.items()
        }
        
        # Update plot every 2 seconds (more frequent updates for timeline)
        if int(current_time) % 2 == 0:
            self.update_plot(frequencies)

    def depth_callback(self):
        """
        Process depth image and log timestamps when it changes
        """
        current_time = time.time()
        
        # Only process if we have all required data
        if self.lidar_points is not None and self.cam_info is not None and self.odom_info["t"] is not None:
            # Generate new depth image
            depth_image, depth_color, mask = self.projection.lidar2depth(self.lidar_points, self.cam_info, self.odom_info)
            
            # Check if the depth image has actually changed
            if self.depth_image is None or not np.array_equal(depth_image, self.depth_image):
                # Update the depth image and log timestamp
                self.depth_image = depth_image.copy()  # Use copy to ensure we have a new array
                self.message_timestamps['depth'].append(current_time)
                # self.get_logger().info(f'New depth image generated at {current_time - self.start_time:.2f}s')

    def update_plot(self, frequencies):
        """Update both frequency and timeline plots"""
        # Update frequency plot
        self.ax_freq.clear()
        topics = list(frequencies.keys())
        freq_values = list(frequencies.values())
        
        bars = self.ax_freq.bar(topics, freq_values)
        self.ax_freq.set_ylabel('Frequency (Hz)')
        self.ax_freq.set_title('Message Frequencies')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            self.ax_freq.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f} Hz',
                        ha='center', va='bottom')
        
        # Update timeline plot
        self.ax_timeline.clear()
        self.ax_timeline.set_ylabel('Message Type')
        self.ax_timeline.set_xlabel('Time (seconds)')
        self.ax_timeline.set_title('Message Timeline')
        self.ax_timeline.grid(True, alpha=0.3)
        
        # Plot timeline for each message type
        y_positions = {topic: idx for idx, topic in enumerate(topics)}
        current_time = time.time()
        window_size = 30  # Show last 30 seconds
        
        for topic in topics:
            timestamps = self.message_timestamps[topic]  # Now using list directly
            if timestamps:
                # Convert timestamps to relative time from start
                relative_times = [t - self.start_time for t in timestamps]
                # Filter to show only last window_size seconds
                recent_times = [t for t in relative_times if t > current_time - self.start_time - window_size]
                if recent_times:
                    y_pos = [y_positions[topic]] * len(recent_times)
                    self.ax_timeline.scatter(recent_times, y_pos, 
                                          c=self.colors[topic], 
                                          label=topic, 
                                          alpha=0.6,
                                          marker='|',
                                          s=100)
        
        # Set x-axis limits to show last window_size seconds
        self.ax_timeline.set_xlim(current_time - self.start_time - window_size, 
                                current_time - self.start_time)
        self.ax_timeline.set_ylim(-0.5, len(topics) - 0.5)
        self.ax_timeline.set_yticks(range(len(topics)))
        self.ax_timeline.set_yticklabels(topics)
        self.ax_timeline.legend(loc='upper right')
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)

    def rgb_listener_callback(self, msg):
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def cam_info_listener_callback(self, msg):
        # intrinsics
        K = msg.k.reshape(3, 3)  # intrinsics
        R = msg.r  # Rectification Matrix, only used for stereo cameras
        P = msg.p.reshape(3, 4)  # projection matrix
        width = msg.width
        height = msg.height

        cam_info = {"K": K, "R": R, "P": P, "width": width, "height": height}
        self.cam_info = cam_info

    def lidar_listener_callback(self, msg):
        """Extract the points from the PointCloud2 message
        return (N,3) array of lidar points from buffer
        """
        points = np.asarray(
            point_cloud2.read_points_list(
                msg, field_names=("x", "y", "z"), skip_nans=True
            )
        )
        self.lidar_points = points

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

def main():
    rclpy.init()
    collect_stats = CollectStatsNode()
    rclpy.spin(collect_stats)
    collect_stats.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
