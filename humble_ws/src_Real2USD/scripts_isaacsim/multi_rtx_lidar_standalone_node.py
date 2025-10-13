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
Date: 2/17/25
from ~/isaacsim/standalone/api/isaacsim.ros2.bridge/subscriber.py
"""
from isaacsim import SimulationApp

simulation_app = SimulationApp({"renderer": "RaytracedLighting", "headless": False})

import omni
from isaacsim.core.api import World
from isaacsim.core.utils import stage as stage_utils
import omni.isaac.core.utils.prims as prims_utils
import omni.replicator.core as rep
from isaacsim.core.utils.extensions import enable_extension
from pxr import Gf, Sdf, Usd, UsdGeom

# enable ROS2 bridge extension
enable_extension("isaacsim.ros2.bridge")

simulation_app.update()

import time

# Note that this is not the system level rclpy, but one compiled for omniverse
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from custom_message.msg import UsdStringIdPCMsg


"""
ros2 topic pub /usd_path std_msgs/msg/String "{data: '/data/SimReadyPacks/Commercial_NVD/Assets/ArchVis/Commercial/Seating/Steelbook.usd'}"

the script will subscribe to topic /usd_path and load the usd file into the world
generate lidars based on translation data
and publish the lidar data to ros2 topics
"""


class IsaacLidarSimNode(Node):
    def __init__(self):
        super().__init__("multi_rtx_lidar_standalone_node")

        # setting up the world with a cube
        self.timeline = omni.timeline.get_timeline_interface()
        self.ros_world = World(physics_dt=1.0, rendering_dt=1.0, stage_units_in_meters=1.0)
        
        self.sub = self.create_subscription(UsdStringIdPCMsg, "/usd/StringIdPC", self.substitute_prim_callback, 10)
    
        # in xyz world frame
        self.translations = [(0, 4.0, 1.0), 
                             (0, -4.0, 1.0),
                             (-4.0, 0, 1.0),
                             (4.0, 0, 1.0)]
        self.sensors = []  # Store sensor references
        self.hydra_textures = []  # Store hydra texture references
        for ii, translate in enumerate(self.translations):
            self.add_lidar(translate, ii)

        self.ros_world.reset()
        self.msg_str = None

    def cleanup_sensors(self):
        """Clean up sensor resources"""
        for sensor in self.sensors:
            try:
                if sensor.IsValid():
                    omni.kit.commands.execute("DeletePrim", path=sensor.GetPath())
            except Exception as e:
                self.get_logger().error(f"Error cleaning up sensor: {e}")
        
        for texture in self.hydra_textures:
            try:
                if texture.IsValid():
                    rep.delete.render_product(texture)
            except Exception as e:
                self.get_logger().error(f"Error cleaning up texture: {e}")
        
        self.sensors = []
        self.hydra_textures = []

    def substitute_prim_callback(self, msg):
        msg_str = msg.data_path
        # only add reference if the message is different than the previous one
        if msg_str != self.msg_str:
            self.add_reference("/World/object", msg_str)
        self.msg_str = msg_str


    def add_reference(self, prim_path, usd_file_path):
        # clear out /World
        world_prim = stage_utils.get_current_stage().GetPrimAtPath("/World")
        if world_prim:
            for child in world_prim.GetChildren():
                prims_utils.delete_prim(child.GetPath())

        # add the reference to the stage and return the prim
        meters_per_unit = 1.0
        try:
            prim = stage_utils.add_reference_to_stage(usd_file_path, prim_path)
            # open only the prim to get the meters per unit
            prim_stage = Usd.Stage.Open(usd_file_path)
            meters_per_unit = UsdGeom.GetStageMetersPerUnit(prim_stage)
            # print(meters_per_unit)
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
                xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
                xform.AddRotateXYZOp().Set(Gf.Vec3d(0, 0, 0))
                xform.AddScaleOp().Set(Gf.Vec3d(1, 1, 1))

                xformOpOrderAttr = prim.GetAttribute("xformOpOrder")
                xformOpOrder = list(xformOpOrderAttr.Get()) if xformOpOrderAttr.IsValid() else []

            # create a new xformOp
            if "xformOp:scale:unitsResolve" not in xformOpOrder:
                xformOpOrder.append("xformOp:scale:unitsResolve")
                xformOpOrderAttr.Set(xformOpOrder)
                # set type attribute as a double3
                units_resolve_attr = prim.CreateAttribute("xformOp:scale:unitsResolve", Sdf.ValueTypeNames.Double3)
                # set the value
                units_resolve_attr.Set(Gf.Vec3d(meters_per_unit, meters_per_unit, meters_per_unit))  # Example values for cm-to-m conversion



    def add_lidar(self, translation, itr):
        try:
            orientation = Gf.Quatd(1.0, 0.0, 0.0, 0.0)

            _, sensor = omni.kit.commands.execute(
                "IsaacSensorCreateRtxLidar",
                path="/sensor_"+str(itr),
                parent=None,
                config="Hesai_XT32_SD10",
                translation=translation,
                orientation=orientation,  # Gf.Quatd is w,i,j,k
            )
            self.sensors.append(sensor)  # Store sensor reference

            # RTX sensors are cameras and must be assigned to their own render product
            hydra_texture = rep.create.render_product(sensor.GetPath(), [1, 1], name="Isaac")
            self.hydra_textures.append(hydra_texture)  # Store texture reference

            # Create Point cloud publisher pipeline in the post process graph
            writer = rep.writers.get("RtxLidar" + "ROS2PublishPointCloud")
            writer.initialize(topicName="/isaac/point_cloud_"+str(itr), frameId="sim_lidar")
            writer.attach([hydra_texture])

            """Create the debug draw pipeline in the post process graph
            This add visualization of the point cloud in isaacsim"""
            # writer = rep.writers.get("RtxLidar" + "DebugDrawPointCloud")
            # writer.attach([hydra_texture])

        except Exception as e:
            self.get_logger().error(f"Error creating LiDAR sensor {itr}: {e}")
            raise

    def run_simulation(self):
        try:
            self.timeline.play()
            reset_needed = False
            while simulation_app.is_running():
                try:
                    self.ros_world.step(render=True)
                    rclpy.spin_once(self, timeout_sec=0.0)
                    if self.ros_world.is_stopped() and not reset_needed:
                        reset_needed = True
                    if self.ros_world.is_playing():
                        if reset_needed:
                            self.ros_world.reset()
                            reset_needed = False
                    
                    # Add a small delay to prevent overwhelming the GPU
                    time.sleep(0.001)  # 1ms delay
                    
                except Exception as e:
                    self.get_logger().error(f"Error in simulation loop: {e}")
                    break

        finally:
            # Cleanup
            self.cleanup_sensors()
            self.timeline.stop()
            self.destroy_node()
            simulation_app.close()


if __name__ == "__main__":
    rclpy.init()
    isaac_lidar_sim_node = IsaacLidarSimNode()
    isaac_lidar_sim_node.run_simulation()