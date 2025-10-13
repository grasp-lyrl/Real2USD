# Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

"""
edited by christopher hsu, chsu8@seas.upenn.edu
Date: 6/6/25
from ~/isaacsim/standalone/api/isaacsim.ros2.bridge/subscriber.py
"""
from isaacsim import SimulationApp
import argparse
import json
import os

simulation_app = SimulationApp({"renderer": "RaytracedLighting", "headless": False})

import omni
import carb
import sys, os
from isaacsim.core.api import World
from isaacsim.core.utils import stage as stage_utils
import isaacsim.core.utils.prims as prims_utils
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.prims import SingleGeometryPrim, RigidPrim
from pxr import Gf, Sdf, Usd, UsdGeom
import torch as th
from isaacsim.storage.native import get_assets_root_path

# enable ROS2 bridge extension
enable_extension("isaacsim.ros2.bridge")

simulation_app.update()

import time

# Note that this is not the system level rclpy, but one compiled for omniverse
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from custom_message.msg import UsdBufferPoseMsg


"""
this script will take in a USD file path and add it to the stage
~/isaacsim/python.sh humble_ws/src_whatchanged/scripts_isaacsim/usd_builder.py --buffer-file /data/SimIsaacData/buffer/matched_buffer_20250612_180831.json 
"""


class UsdBuilder(Node):
    def __init__(self, buffer_file=None):
        super().__init__("usd_builder")

        # Initialize tracking dictionaries first
        self.prim_paths = []
        self.usd_paths = []
        self.obj_ids = []  # This will store cluster ids
        self.positions = []
        self.orientations = []
        self.id_to_prim_path = {}  # Map from cluster id to prim path
        self.object_buffer = {}  # Buffer for incoming objects

        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')

        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            simulation_app.close()
            sys.exit()

        self.timeline = omni.timeline.get_timeline_interface()
        self.ros_world = World(physics_dt=1.0/60, rendering_dt=1.0/60, stage_units_in_meters=1.0, backend="torch")
        self.stage = simulation_app.context.get_stage()
        self.ros_world.scene.add_default_ground_plane()
        
        # Only subscribe to ROS topic if not loading from buffer file
        if buffer_file is None:
            self.sub = self.create_subscription(UsdBufferPoseMsg, "/usd/SimUsdPoseBuffer", self.listener_callback, 10)
        else:
            self.load_from_buffer_file(buffer_file)

        self.ros_world.reset()

        self.itr = 0
        # Hybrid approach additions
        self.update_frequency = 10  # Update scene every N simulation steps
        self.save_frequency = 50  # Save every N simulation steps
        self.sim_step_counter = 0
        self.step_counter = 0
        self.total_steps = 0
        self.save_counter = 0

    def listener_callback(self, msg):
        # msg is UsdBufferPoseMsg
        current_ids = set()
        for obj_msg in msg.objects:
            obj_id = obj_msg.id
            current_ids.add(obj_id)
            usd_file_path = obj_msg.data_path
            t = [obj_msg.pose.position.x, obj_msg.pose.position.y, obj_msg.pose.position.z]
            quatWXYZ = [obj_msg.pose.orientation.w, obj_msg.pose.orientation.x, obj_msg.pose.orientation.y, obj_msg.pose.orientation.z]
            prim_path = f"/World/object_{obj_id}"
            self.id_to_prim_path[obj_id] = prim_path
            self.object_buffer[obj_id] = {
                'usd': usd_file_path,
                'position': t,
                'quatWXYZ': quatWXYZ,
                'prim_path': prim_path
            }

        # Remove objects not in the current message
        to_remove = [obj_id for obj_id in self.object_buffer if obj_id not in current_ids]
        for obj_id in to_remove:
            prim_path = self.id_to_prim_path.get(obj_id)
            if prim_path:
                try:
                    prims_utils.delete_prim(prim_path)
                except Exception as e:
                    print(f"Error deleting prim {prim_path}: {e}")
            self.object_buffer.pop(obj_id)
            self.id_to_prim_path.pop(obj_id, None)

        self.step_counter += 1
        self.total_steps += 1
        self.save_counter += 1

        if self.step_counter >= self.update_frequency:
            self.update_scene()
            self.step_counter = 0

        if self.save_counter >= self.save_frequency:
            self.save_usda()
            self.save_counter = 0

    def update_scene(self):
        self.ros_world.stop()

        # Remove objects not in the current buffer
        current_obj_ids = set(self.object_buffer.keys())
        paths_to_remove = []
        for prim_path in self.prim_paths:
            try:
                obj_id = int(prim_path.split('_')[-1])
                if obj_id not in current_obj_ids:
                    world_prim = self.stage.GetPrimAtPath(prim_path)
                    if world_prim:
                        try:
                            prims_utils.delete_prim(prim_path)
                        except Exception as e:
                            print(f"Error deleting prim {prim_path}: {e}")
                        paths_to_remove.append(prim_path)
            except ValueError:
                continue

        # Update tracking lists
        indices_to_keep = [i for i, path in enumerate(self.prim_paths) if path not in paths_to_remove]
        self.prim_paths = [self.prim_paths[i] for i in indices_to_keep]
        self.usd_paths = [self.usd_paths[i] for i in indices_to_keep]
        self.obj_ids = [self.obj_ids[i] for i in indices_to_keep]
        self.positions = [self.positions[i] for i in indices_to_keep]
        self.orientations = [self.orientations[i] for i in indices_to_keep]

        # Add or update all objects in the buffer
        for obj_id, obj in self.object_buffer.items():
            self.get_logger().info(f"Adding object {obj_id} with usd path {obj['usd']}")
            self.add_reference(obj['prim_path'], obj['usd'], obj['position'], obj['quatWXYZ'], obj_id)

        self.ros_world.reset()
        self.ros_world.play()

    def set_pose(self):
        """
        Set poses for all prims in the scene.
        """
        if len(self.prim_paths) > 0:  # Changed from != 0 to > 0 for clarity
            prims = RigidPrim(
                prim_paths_expr=self.prim_paths,
                name="rigid_prim",
            )

            # Convert stored positions and orientations to tensors
            positions_tensor = th.tensor(self.positions, device=self.device, dtype=th.float32)
            orientations_tensor = th.tensor(self.orientations, device=self.device, dtype=th.float32)

            # Set the poses to the stored values
            prims.set_world_poses(
                positions=positions_tensor,
                orientations=orientations_tensor,
            )
            prims.set_velocities(
                velocities=th.zeros((len(self.prim_paths),6), device=self.device),
            )

    def add_reference(self, prim_path, usd_file_path, t, quatWXYZ, obj_id):
        # check if the prim already exists and delete it
        world_prim = stage_utils.get_current_stage().GetPrimAtPath(prim_path)
        if world_prim:
            prims_utils.delete_prim(prim_path)

        # store data to lists if new else edit
        if prim_path not in self.prim_paths:
            self.prim_paths.append(prim_path)
            self.usd_paths.append(usd_file_path)
            self.obj_ids.append(obj_id)
            self.positions.append(t)
            self.orientations.append(quatWXYZ)
        else:
            idx = self.prim_paths.index(prim_path)
            self.usd_paths[idx] = usd_file_path
            self.obj_ids[idx] = obj_id
            self.positions[idx] = t
            self.orientations[idx] = quatWXYZ
        
        # create the prim, which is the base object with just a transform
        label = self.get_semantic_label(usd_file_path)
        
        # Create the prim with initial scale
        prim = prims_utils.create_prim(
            prim_path=prim_path,
            position=t,
            orientation=quatWXYZ,
            scale=Gf.Vec3d(1.0, 1.0, 1.0),  # Use Gf.Vec3d for double precision
            usd_path=usd_file_path,
            semantic_label=label,
        )

        # wrap the prim in a RigidPrim object with colliders
        geo_prim = SingleGeometryPrim(
            prim_path=prim_path,
            name=prim_path,
            collision=True)
        geo_prim.set_collision_approximation("convexHull")
        
        # add to isaac sim registry - always add since we're creating a new prim
        self.ros_world.scene.add(geo_prim)

        # add the reference to the stage and return the prim
        meters_per_unit = 1.0
        try:
            prim = stage_utils.add_reference_to_stage(usd_file_path, prim_path)
            # open only the prim to get the meters per unit
            prim_stage = Usd.Stage.Open(usd_file_path)
            meters_per_unit = UsdGeom.GetStageMetersPerUnit(prim_stage)
        except:
            print("Not a valid USD file path")

        if meters_per_unit != 1.0:
            # Get the current xformOpOrder
            try:
                xformOpOrderAttr = prim.GetAttribute("xformOpOrder")
                xformOpOrder = list(xformOpOrderAttr.Get()) if xformOpOrderAttr.IsValid() else []
            except:
                ## if we can't find the xformOpOrder, create it
                xform = UsdGeom.Xformable(prim)
                xform.SetXformOpOrder([])
                xformOpOrderAttr = prim.GetAttribute("xformOpOrder")
                xformOpOrder = list(xformOpOrderAttr.Get()) if xformOpOrderAttr.IsValid() else []

            # create a new xformOp
            if "xformOp:scale:unitsResolve" not in xformOpOrder:
                xformOpOrder.append("xformOp:scale:unitsResolve")
                xformOpOrderAttr.Set(xformOpOrder)
                # set type attribute as a double3
                units_resolve_attr = prim.CreateAttribute("xformOp:scale:unitsResolve", Sdf.ValueTypeNames.Double3)
                # set the value
                units_resolve_attr.Set(Gf.Vec3d(meters_per_unit, meters_per_unit, meters_per_unit))

        # add bounding box metadata
        self.add_bounding_box(prim)

    def add_bounding_box(self, prim):
        """
        this function is called when we want to add a bounding box to the prim
        """
        current_time = Usd.TimeCode.Default()
        included_purposes = [UsdGeom.Tokens.default_]
        bbox_cache = UsdGeom.BBoxCache(current_time, included_purposes)
        bound = bbox_cache.ComputeWorldBound(prim)
        bound_range = bound.ComputeAlignedBox()

        if bound_range.GetMin() and bound_range.GetMax():
            min_corner = bound_range.GetMin()
            max_corner = bound_range.GetMax()

            min_attr = prim.CreateAttribute("boundingBoxMin", Sdf.ValueTypeNames.Float3)
            max_attr = prim.CreateAttribute("boundingBoxMax", Sdf.ValueTypeNames.Float3)


            min_attr.Set(min_corner)
            max_attr.Set(max_corner)

            # print(f"Bounding box metadata added to {prim.GetPath()} using BBoxCache")
        else:
            # print(f"Failed to compute bounding box for {prim.GetPath()} using BBoxCache")
            pass


    def get_semantic_label(self, usd_file_path):
        parts = usd_file_path.split(os.sep)

        if len(parts) >= 2:  # Ensure there are at least 3 parts (directory/directory/file.usd)
            return str(os.path.join(parts[-2], os.path.splitext(parts[-1])[0]))
        else:
            return "object"  # Not enough directories


    def save_usda(self):
        """
        Save the current stage state to a USD file with incrementing number.
        Files are named as 'state_YYYYMMDD_NN.usda' where NN is a two-digit number.
        If a file with the same name exists, it will be overwritten.
        """
        base_path = "/data/SimIsaacData/usda"
        os.makedirs(base_path, exist_ok=True)
        
        # Get current date
        current_date = time.strftime("%Y%m%d")
        
        # Create filename with current counter
        file_name = f"state_{current_date}.usda"
        file_path = os.path.join(base_path, file_name)
        
        # If the file exists, remove it first to ensure clean overwrite
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                self.get_logger().info(f"Removed existing file: {file_path}")
            except Exception as e:
                self.get_logger().warn(f"Failed to remove existing file: {e}")
        
        # Export the stage as USDA format (ASCII)
        try:
            self.stage.GetRootLayer().Export(file_path)
            self.get_logger().info(f"Saved stage state to {file_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to save stage state: {e}")

    def load_from_buffer_file(self, buffer_file):
        """Load objects from a saved buffer file"""
        try:
            with open(buffer_file, 'r') as f:
                buffer_data = json.load(f)
            
            self.get_logger().info(f"Loading buffer from {buffer_file} (timestamp: {buffer_data['timestamp']})")
            
            # Process each object in the buffer
            for obj_data in buffer_data['objects']:
                obj_id = obj_data['cluster_id']
                usd_file_path = obj_data['usd_path']
                t = obj_data['position']
                quatWXYZ = obj_data['quatWXYZ']
                prim_path = f"/World/object_{obj_id}"
                
                self.id_to_prim_path[obj_id] = prim_path
                self.object_buffer[obj_id] = {
                    'usd': usd_file_path,
                    'position': t,
                    'quatWXYZ': quatWXYZ,
                    'prim_path': prim_path
                }
            
            # Update the scene immediately
            self.update_scene()
            
            # Ensure poses are set after loading
            if len(self.prim_paths) > 0:
                self.set_pose()
                self.get_logger().info(f"Set poses for {len(self.prim_paths)} objects loaded from buffer")
            
        except Exception as e:
            self.get_logger().error(f"Failed to load buffer file {buffer_file}: {e}")
            simulation_app.close()
            sys.exit(1)

    def run_simulation(self):
        self.timeline.play()
        reset_needed = False
        while simulation_app.is_running():
            self.ros_world.step(render=True)
            for __ in range(10):
                rclpy.spin_once(self, timeout_sec=0.0)
            if self.ros_world.is_stopped() and not reset_needed:
                reset_needed = True
            if self.ros_world.is_playing():
                if reset_needed:
                    self.ros_world.reset()
                    reset_needed = False
                self.set_pose()

        # Cleanup
        self.timeline.stop()
        self.destroy_node()
        simulation_app.close()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='USD Builder for Isaac Sim')
    parser.add_argument('--buffer-file', type=str, help='Path to a saved buffer file to load')
    args = parser.parse_args()

    rclpy.init()
    usd_builder = UsdBuilder(buffer_file=args.buffer_file)
    usd_builder.run_simulation()


if __name__ == "__main__":
    main()