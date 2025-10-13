# Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

"""
Pre-process USD dataset to generate and save point cloud data
Created by: Christopher Hsu, chsu8@seas.upenn.edu
Date: 2/17/25

This script loads USD objects into Isaac Sim, generates point cloud data using LiDAR sensors,
and saves the data to files for later use in the full pipeline without real-time simulation.
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse

from isaacsim import SimulationApp

# Initialize Isaac Sim
simulation_app = SimulationApp({"renderer": "RaytracedLighting", "headless": True})

import omni
from isaacsim.core.api import World
from isaacsim.core.utils import stage as stage_utils
import omni.isaac.core.utils.prims as prims_utils
import omni.replicator.core as rep
from isaacsim.core.utils.extensions import enable_extension
from pxr import Gf, Sdf, Usd, UsdGeom

# Enable ROS2 bridge extension
enable_extension("isaacsim.ros2.bridge")

simulation_app.update()

import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header


class USDPreprocessor:
    """
    Pre-processes USD objects to generate and save point cloud data
    """
    
    def __init__(self, output_dir: str = "/data/preprocessed_usd_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Isaac Sim world
        self.timeline = omni.timeline.get_timeline_interface()
        self.ros_world = World(physics_dt=1.0, rendering_dt=1.0, stage_units_in_meters=1.0)
        
        # LiDAR sensor positions (same as in multi_rtx_lidar_standalone_node.py)
        self.translations = [(0, 4.0, 1.0), 
                             (0, -4.0, 1.0),
                             (-4.0, 0, 1.0),
                             (4.0, 0, 1.0)]
        
        self.sensors = []
        self.hydra_textures = []
        
        # Setup LiDAR sensors
        for ii, translate in enumerate(self.translations):
            self.add_lidar(translate, ii)
        
        self.ros_world.reset()
        
        # Data storage
        self.point_clouds = {i: None for i in range(len(self.translations))}
        self.point_cloud_received = {i: False for i in range(len(self.translations))}
        
        # Initialize ROS2 node for point cloud reception
        rclpy.init()
        self.node = rclpy.create_node("usd_preprocessor")
        
        # Subscribe to point cloud topics
        for i in range(len(self.translations)):
            self.node.create_subscription(
                PointCloud2, 
                f"/isaac/point_cloud_{i}", 
                lambda msg, idx=i: self.point_cloud_callback(msg, idx), 
                10
            )
    
    def add_lidar(self, translation: Tuple[float, float, float], itr: int):
        """Add LiDAR sensor to the simulation"""
        try:
            orientation = Gf.Quatd(1.0, 0.0, 0.0, 0.0)
            
            _, sensor = omni.kit.commands.execute(
                "IsaacSensorCreateRtxLidar",
                path=f"/sensor_{itr}",
                parent=None,
                config="Hesai_XT32_SD10",
                translation=translation,
                orientation=orientation,
            )
            self.sensors.append(sensor)
            
            # RTX sensors are cameras and must be assigned to their own render product
            hydra_texture = rep.create.render_product(sensor.GetPath(), [1, 1], name="Isaac")
            self.hydra_textures.append(hydra_texture)
            
            # Create Point cloud publisher pipeline in the post process graph
            writer = rep.writers.get("RtxLidar" + "ROS2PublishPointCloud")
            writer.initialize(topicName=f"/isaac/point_cloud_{itr}", frameId="sim_lidar")
            writer.attach([hydra_texture])
            
        except Exception as e:
            print(f"Error creating LiDAR sensor {itr}: {e}")
            raise
    
    def point_cloud_callback(self, msg: PointCloud2, sensor_idx: int):
        """Callback for receiving point cloud data from LiDAR sensors"""
        try:
            points = np.asarray(
                point_cloud2.read_points_list(
                    msg, field_names=("x", "y", "z"), skip_nans=True
                )
            )
            
            # Transform points to world coordinates
            world_points = points + np.array(self.translations[sensor_idx])
            self.point_clouds[sensor_idx] = world_points
            self.point_cloud_received[sensor_idx] = True
            
        except Exception as e:
            print(f"Error processing point cloud from sensor {sensor_idx}: {e}")
    
    def load_usd_object(self, usd_path: str) -> bool:
        """Load a USD object into the simulation"""
        try:
            # Clear existing objects
            world_prim = stage_utils.get_current_stage().GetPrimAtPath("/World")
            if world_prim:
                for child in world_prim.GetChildren():
                    prims_utils.delete_prim(child.GetPath())
            
            # Add the reference to the stage
            meters_per_unit = 1.0
            try:
                prim = stage_utils.add_reference_to_stage(usd_path, "/World/object")
                # Get meters per unit from the USD file
                prim_stage = Usd.Stage.Open(usd_path)
                meters_per_unit = UsdGeom.GetStageMetersPerUnit(prim_stage)
            except Exception as e:
                print(f"Error loading USD file {usd_path}: {e}")
                return False
            
            # Handle unit conversion if needed
            if meters_per_unit != 1.0:
                try:
                    xformOpOrderAttr = prim.GetAttribute("xformOpOrder")
                    xformOpOrder = list(xformOpOrderAttr.Get()) if xformOpOrderAttr.IsValid() else []
                except:
                    # Create xformOpOrder if it doesn't exist
                    xform = UsdGeom.Xformable(prim)
                    xform.SetXformOpOrder([])
                    xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
                    xform.AddRotateXYZOp().Set(Gf.Vec3d(0, 0, 0))
                    xform.AddScaleOp().Set(Gf.Vec3d(1, 1, 1))
                    
                    xformOpOrderAttr = prim.GetAttribute("xformOpOrder")
                    xformOpOrder = list(xformOpOrderAttr.Get()) if xformOpOrderAttr.IsValid() else []
                
                # Add units resolve transform
                if "xformOp:scale:unitsResolve" not in xformOpOrder:
                    xformOpOrder.append("xformOp:scale:unitsResolve")
                    xformOpOrderAttr.Set(xformOpOrder)
                    units_resolve_attr = prim.CreateAttribute("xformOp:scale:unitsResolve", Sdf.ValueTypeNames.Double3)
                    units_resolve_attr.Set(Gf.Vec3d(meters_per_unit, meters_per_unit, meters_per_unit))
            
            return True
            
        except Exception as e:
            print(f"Error loading USD object {usd_path}: {e}")
            return False
    
    def generate_point_cloud_data(self, usd_path: str, timeout: float = 10.0) -> Optional[np.ndarray]:
        """Generate point cloud data for a USD object"""
        # Reset point cloud data
        for i in range(len(self.translations)):
            self.point_clouds[i] = None
            self.point_cloud_received[i] = False
        
        # Load the USD object
        if not self.load_usd_object(usd_path):
            return None
        
        # Start simulation
        self.timeline.play()
        self.ros_world.reset()
        
        # Wait for point cloud data
        start_time = time.time()
        all_received = False
        
        while time.time() - start_time < timeout:
            try:
                self.ros_world.step(render=True)
                rclpy.spin_once(self.node, timeout_sec=0.01)
                
                # Check if all sensors have received data
                all_received = all(self.point_cloud_received.values())
                if all_received:
                    break
                    
            except Exception as e:
                print(f"Error in simulation step: {e}")
                break
        
        # Stop simulation
        self.timeline.stop()
        
        if all_received:
            # Combine all point clouds
            combined_points = []
            for i in range(len(self.translations)):
                if self.point_clouds[i] is not None:
                    combined_points.append(self.point_clouds[i])
            
            if combined_points:
                return np.vstack(combined_points)
        
        return None
    
    def save_point_cloud_data(self, usd_path: str, point_cloud: np.ndarray, object_id: int):
        """Save point cloud data to file"""
        # Create relative path structure based on USD path
        usd_path_obj = Path(usd_path)
        
        # Find the common base directory (e.g., /data/SimReadyAssets)
        # We'll assume the base is the first two levels of the path
        path_parts = usd_path_obj.parts
        if len(path_parts) >= 3:
            # Skip the first two parts (e.g., /data/SimReadyAssets)
            relative_path = Path(*path_parts[3:])
        else:
            # Fallback to just the filename
            relative_path = usd_path_obj.name
        
        # Create the output directory structure
        output_subdir = self.output_dir / relative_path.parent
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        # Create filename without object ID, just use the original name with .pkl extension
        usd_filename = relative_path.stem
        output_file = output_subdir / f"{usd_filename}.pkl"
        
        # Save data
        data = {
            'usd_path': usd_path,
            'object_id': object_id,
            'point_cloud': point_cloud,
            'sensor_translations': self.translations,
            'timestamp': time.time()
        }
        
        with open(output_file, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Saved point cloud data to {output_file}")
        return str(output_file)
    
    def process_usd_dataset(self, usd_paths: List[str], start_id: int = 0):
        """Process a list of USD files and generate point cloud data"""
        results = []
        
        for i, usd_path in enumerate(usd_paths):
            object_id = start_id + i
            print(f"Processing {usd_path} (ID: {object_id})")
            
            # Generate point cloud data
            point_cloud = self.generate_point_cloud_data(usd_path)
            
            if point_cloud is not None:
                # Save data
                output_file = self.save_point_cloud_data(usd_path, point_cloud, object_id)
                
                results.append({
                    'usd_path': usd_path,
                    'object_id': object_id,
                    'output_file': output_file,
                    'point_cloud_shape': point_cloud.shape,
                    'status': 'success'
                })
                
                print(f"Successfully processed {usd_path} - Point cloud shape: {point_cloud.shape}")
            else:
                results.append({
                    'usd_path': usd_path,
                    'object_id': object_id,
                    'status': 'failed',
                    'error': 'Failed to generate point cloud data'
                })
                
                print(f"Failed to process {usd_path}")
        
        # Save processing results
        results_file = self.output_dir / "processing_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Processing complete. Results saved to {results_file}")
        return results
    
    def cleanup(self):
        """Clean up resources"""
        # Clean up sensors
        for sensor in self.sensors:
            try:
                if sensor.IsValid():
                    omni.kit.commands.execute("DeletePrim", path=sensor.GetPath())
            except Exception as e:
                print(f"Error cleaning up sensor: {e}")
        
        for texture in self.hydra_textures:
            try:
                if texture.IsValid():
                    rep.delete.render_product(texture)
            except Exception as e:
                print(f"Error cleaning up texture: {e}")
        
        # Clean up ROS2
        self.node.destroy_node()
        rclpy.shutdown()
        
        # Close simulation
        self.timeline.stop()
        simulation_app.close()


class PointCloudLookup:
    """
    Provides lookup functionality for pre-processed point cloud data
    """
    
    def __init__(self, data_dir: str = "/data/preprocessed_usd_data"):
        self.data_dir = Path(data_dir)
        self.cache = {}
        self.load_metadata()
    
    def load_metadata(self):
        """Load metadata from processing results"""
        results_file = self.data_dir / "processing_results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = []
            print(f"Warning: No processing results found at {results_file}")
    
    def get_point_cloud(self, usd_path: str, object_id: int) -> Optional[np.ndarray]:
        """Get point cloud data for a specific USD path and object ID"""
        # Check cache first
        cache_key = f"{usd_path}_{object_id}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Look for saved data
        usd_filename = Path(usd_path).stem
        data_file = self.data_dir / f"{usd_filename}_id_{object_id}.pkl"
        
        if data_file.exists():
            try:
                with open(data_file, 'rb') as f:
                    data = pickle.load(f)
                
                point_cloud = data['point_cloud']
                self.cache[cache_key] = point_cloud
                return point_cloud
                
            except Exception as e:
                print(f"Error loading point cloud data from {data_file}: {e}")
                return None
        else:
            print(f"Point cloud data not found for {usd_path} (ID: {object_id})")
            return None
    
    def get_available_objects(self) -> List[Dict]:
        """Get list of available pre-processed objects"""
        return self.metadata.copy()


def main():
    parser = argparse.ArgumentParser(description="Pre-process USD dataset to generate point cloud data")
    parser.add_argument("--usd_list", type=str, required=True, 
                       help="Path to file containing list of USD paths (one per line)")
    parser.add_argument("--output_dir", type=str, default="/data/preprocessed_usd_data",
                       help="Output directory for pre-processed data")
    parser.add_argument("--start_id", type=int, default=0,
                       help="Starting object ID for the dataset")
    
    args = parser.parse_args()
    
    # Load USD paths
    with open(args.usd_list, 'r') as f:
        usd_paths = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(usd_paths)} USD files to process")
    
    # Initialize preprocessor
    preprocessor = USDPreprocessor(args.output_dir)
    
    try:
        # Process dataset
        results = preprocessor.process_usd_dataset(usd_paths, args.start_id)
        
        # Print summary
        successful = sum(1 for r in results if r['status'] == 'success')
        failed = len(results) - successful
        
        print(f"\nProcessing Summary:")
        print(f"Total files: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        
    finally:
        preprocessor.cleanup()


if __name__ == "__main__":
    main() 