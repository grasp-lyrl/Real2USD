import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from custom_message.msg import UsdStringIdPCMsg, UsdStringIdSrcTargMsg
import numpy as np
import pickle
import json
from pathlib import Path
from typing import Optional, Dict, List

"""
Modified isaac_lidar_node that uses pre-processed point cloud data
instead of real-time simulation.

This node loads pre-processed point cloud data from files and publishes
them in the same format as the original isaac_lidar_node, allowing
the full pipeline to work without real-time simulation.

***IMPORTANT***
Update the data_dir to the correct path. based on your directory structure.
See Readme/USD_PREPROCESS_README.md for more details.

"""

class IsaacLidarNodePreprocessed(Node):
    def __init__(self):
        super().__init__("sim_lidar2world_node_preprocessed")
        
        # Subscribe to all data from past nodes
        self.sub = self.create_subscription(UsdStringIdPCMsg, "/usd/StringIdPC", self.substitute_prim_callback, 10)

        # Publisher of src and target together
        self.publisher = self.create_publisher(UsdStringIdSrcTargMsg, "/usd/StringIdSrcTarg", 10)

        # Debugging publishers
        self.pub_lidar_src = self.create_publisher(PointCloud2, "/isaac/src_point_cloud", 10)
        self.pub_lidar_target = self.create_publisher(PointCloud2, "/isaac/target_point_cloud", 10)
        
        # Initialize point cloud lookup
        self.data_dir = Path("/data/preprocessed_usd_data")
        self.cache = {}
        self.load_metadata()
        
        self.get_logger().info("IsaacLidarNodePreprocessed initialized - using pre-processed data")

    def load_metadata(self):
        """Load metadata from processing results"""
        results_file = self.data_dir / "processing_results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                self.metadata = json.load(f)
            self.get_logger().info(f"Loaded metadata for {len(self.metadata)} pre-processed objects")
            
            # Log available USD paths for debugging
            available_paths = [obj['usd_path'] for obj in self.metadata if obj['status'] == 'success']
            self.get_logger().info(f"Available USD paths: {len(available_paths)}")
            for i, path in enumerate(available_paths[:5]):  # Show first 5
                self.get_logger().info(f"  {i}: {path}")
            if len(available_paths) > 5:
                self.get_logger().info(f"  ... and {len(available_paths) - 5} more")
        else:
            self.metadata = []
            self.get_logger().warning(f"No processing results found at {results_file}")

    def get_point_cloud(self, usd_path: str, object_id: int) -> Optional[np.ndarray]:
        """Get point cloud data for a specific USD path"""
        # Check cache first
        cache_key = usd_path
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Create the same folder structure as the USD path
        usd_path_obj = Path(usd_path)
        
        # Find the common base directory (e.g., /data/SimReadyAssets)
        path_parts = usd_path_obj.parts
        if len(path_parts) >= 3:
            # Skip the first two parts (e.g., /data/SimReadyAssets)
            relative_path = Path(*path_parts[3:])
        else:
            # Fallback to just the filename
            relative_path = usd_path_obj.name
        
        # Look for saved data in the same folder structure
        data_file = self.data_dir / relative_path.parent / f"{relative_path.stem}.pkl"
        
        if data_file.exists():
            try:
                with open(data_file, 'rb') as f:
                    data = pickle.load(f)
                
                point_cloud = data['point_cloud']
                self.cache[cache_key] = point_cloud
                self.get_logger().info(f"Loaded point cloud for {usd_path} - shape: {point_cloud.shape}")
                return point_cloud
                
            except Exception as e:
                self.get_logger().error(f"Error loading point cloud data from {data_file}: {e}")
                return None
        else:
            self.get_logger().warning(f"Point cloud data not found for {usd_path} at {data_file}")
            return None

    def substitute_prim_callback(self, msg):
        # Get pre-processed point cloud data
        usd_path = msg.data_path
        object_id = msg.id
        
        point_cloud = self.get_point_cloud(usd_path, object_id)
        
        if point_cloud is not None:
            # Create source point cloud message
            src_msg = point_cloud2.create_cloud_xyz32(
                header=Header(frame_id="sim_lidar"), 
                points=point_cloud
            )
            src_msg.header.stamp = self.get_clock().now().to_msg()
            
            # Create registration message
            registration_msg = UsdStringIdSrcTargMsg()
            registration_msg.header = msg.header
            registration_msg.data_path = msg.data_path
            registration_msg.id = msg.id
            registration_msg.src_pc = src_msg
            registration_msg.targ_pc = msg.pc
            
            # Publish messages
            self.publisher.publish(registration_msg)
            self.pub_lidar_src.publish(src_msg)
            self.pub_lidar_target.publish(msg.pc)
            
            self.get_logger().info(f"Published pre-processed point cloud for {usd_path}")
        else:
            self.get_logger().warning(f"No pre-processed data available for {usd_path} (ID: {object_id})")


def main():
    rclpy.init()
    sim_lidar2world_node = IsaacLidarNodePreprocessed()
    rclpy.spin(sim_lidar2world_node)
    sim_lidar2world_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main() 