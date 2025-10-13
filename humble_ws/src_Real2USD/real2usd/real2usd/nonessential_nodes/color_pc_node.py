import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
import cv2
import open3d as o3d
from cv_bridge import CvBridge
from collections import deque
import numpy as np
from scipy.spatial.transform import Rotation as R
import time, asyncio, struct
from ipdb import set_trace as st
import os

class ColorPCNode(Node):
    def __init__(self):
        super().__init__("color_pc_node")

        # Subscribe to incoming point cloud
        self.create_subscription(PointCloud2, "/segment/pointcloud_color", self.lidar_callback, 10)
        
        # Store accumulated points
        self.accumulated_points = []
        self.accumulated_colors = []
        
        # Point cloud fields definition (needed for reading the incoming point cloud)
        self.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1)
        ]
        
        # Parameters
        self.declare_parameter('save_interval', 10)  # Save every 10 frames
        self.declare_parameter('output_dir', 'pointclouds')  # Directory to save PLY files
        self.declare_parameter('voxel_size', 0.01)  # Size of voxel for downsampling in meters
        self.frame_count = 0
        
        # Create output directory if it doesn't exist
        os.makedirs(self.get_parameter('output_dir').value, exist_ok=True)
        
        # Fixed output filename
        self.output_file = f"{self.get_parameter('output_dir').value}/accumulated_pointcloud.ply"

    def save_to_ply(self):
        if len(self.accumulated_points) == 0:
            return
            
        # Convert to Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(self.accumulated_points))
        
        # Convert colors from float32 back to RGB
        colors_uint32 = np.array([struct.unpack('I', struct.pack('f', val))[0] for val in self.accumulated_colors], dtype=np.uint32)
        colors = np.zeros((len(colors_uint32), 3), dtype=np.uint8)
        colors[:, 0] = (colors_uint32 >> 16) & 0xFF  # R
        colors[:, 1] = (colors_uint32 >> 8) & 0xFF   # G
        colors[:, 2] = colors_uint32 & 0xFF          # B
        
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
        
        # Downsample the point cloud
        voxel_size = self.get_parameter('voxel_size').value
        downsampled_pcd = pcd.voxel_down_sample(voxel_size)
        
        # Save to PLY file
        o3d.io.write_point_cloud(self.output_file, downsampled_pcd)
        self.get_logger().info(f"Updated point cloud in {self.output_file} with {len(downsampled_pcd.points)} points (downsampled from {len(pcd.points)})")

    def lidar_callback(self, msg: PointCloud2):
        # Read points from incoming message
        pc_data = point_cloud2.read_points_list(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True)
        
        # Convert to numpy array and extract points and colors
        points = np.array([[p.x, p.y, p.z] for p in pc_data])
        colors = np.array([p.rgb for p in pc_data])
        
        # Add to accumulated data
        self.accumulated_points.extend(points)
        self.accumulated_colors.extend(colors)
        
        # Check if it's time to save
        self.frame_count += 1
        if self.frame_count >= self.get_parameter('save_interval').value:
            self.save_to_ply()
            self.frame_count = 0

def main():
    rclpy.init()
    color_pc_node = ColorPCNode()
    rclpy.spin(color_pc_node)
    color_pc_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
