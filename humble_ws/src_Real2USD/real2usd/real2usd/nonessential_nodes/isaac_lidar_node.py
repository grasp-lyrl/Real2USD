import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from message_filters import ApproximateTimeSynchronizer, Subscriber
from custom_message.msg import UsdStringIdPCMsg, UsdStringIdSrcTargMsg
import numpy as np

from ipdb import set_trace as st

"""
/home/hsu/repos/whatchanged/humble_ws/src_whatchanged/scripts_isaacsim/multi_rtx_lidar_standalone.py

this node works with the standalone script where it publishes point cloud data generated from lidars in isaac sim
make sure the sensor locations/translations in the world frame are aligned between this file and the standalone script
"""

class IsaacLidarNode(Node):
    def __init__(self):
        super().__init__("sim_lidar2world_node")
        
        self.translations = [(0, 4.0, 1.0), 
                             (0, -4.0, 1.0),
                             (-4.0, 0, 1.0),
                             (4.0, 0, 1.0)]
        # subscribe to all data from isaac and past nodes
        self.sub = self.create_subscription(UsdStringIdPCMsg, "/usd/StringIdPC", self.substitute_prim_callback, 10)

        self.sub_pc_0 = self.create_subscription(PointCloud2, "/isaac/point_cloud_0", self.callback_0, 10)  
        self.sub_pc_1 = self.create_subscription(PointCloud2, "/isaac/point_cloud_1", self.callback_1, 10)
        self.sub_pc_2 = self.create_subscription(PointCloud2, "/isaac/point_cloud_2", self.callback_2, 10)
        self.sub_pc_3 = self.create_subscription(PointCloud2, "/isaac/point_cloud_3", self.callback_3, 10)

        # publisher of src and target together
        self.publisher = self.create_publisher(UsdStringIdSrcTargMsg, "/usd/StringIdSrcTarg", 10)

        # debugging publishers
        self.pub_lidar_src = self.create_publisher(PointCloud2, "/isaac/src_point_cloud", 10)
        self.pub_lidar_target = self.create_publisher(PointCloud2, "/isaac/target_point_cloud", 10)

        self.timer = self.create_timer(0.1, self.timer_callback)

        self.StringIdPcMsg = None
    def callback_0(self, msg_0):
        points_0 = np.asarray(
            point_cloud2.read_points_list(
                msg_0, field_names=("x", "y", "z"), skip_nans=True
            )
        )
        try:
            self.point_0 = points_0 + self.translations[0]
        except:
            # print("points 0 not found")
            pass

    def callback_1(self, msg_1):
        points_1 = np.asarray(
            point_cloud2.read_points_list(
                msg_1, field_names=("x", "y", "z"), skip_nans=True
            )
        )
        try:
            self.point_1 = points_1 + self.translations[1]
        except:
            # print("points 1 not found")
            pass
    
    def callback_2(self, msg_2):
        points_2 = np.asarray(
            point_cloud2.read_points_list(
                msg_2, field_names=("x", "y", "z"), skip_nans=True
            )
        )
        try:
            self.point_2 = points_2 + self.translations[2]
        except:
            # print("points 2 not found")
            pass
    
    def callback_3(self, msg_3):
        points_3 = np.asarray(
            point_cloud2.read_points_list(
                msg_3, field_names=("x", "y", "z"), skip_nans=True
            )
        )
        try:
            self.point_3 = points_3 + self.translations[3]
        except:
            # print("points 3 not found")
            pass

    def substitute_prim_callback(self, msg):
        self.StringIdPcMsg = msg

    def timer_callback(self):
        try:
            points = np.vstack([self.point_0, self.point_1, self.point_2, self.point_3])
            src_msg = point_cloud2.create_cloud_xyz32(header=Header(frame_id="sim_lidar"), points=points)
            src_msg.header.stamp = self.get_clock().now().to_msg()
        except:
            points = None
    
        if self.StringIdPcMsg is not None and points is not None:
            registration_msg = UsdStringIdSrcTargMsg()
            registration_msg.header = self.StringIdPcMsg.header
            registration_msg.data_path = self.StringIdPcMsg.data_path
            registration_msg.id = self.StringIdPcMsg.id
            registration_msg.src_pc = src_msg
            registration_msg.targ_pc = self.StringIdPcMsg.pc
            
            # publish target point cloud
            targ_msg = self.StringIdPcMsg.pc
            targ_msg.header.frame_id = "sim_lidar"
            self.publisher.publish(registration_msg)
            self.pub_lidar_src.publish(src_msg)
            self.pub_lidar_target.publish(targ_msg)



def main():
    rclpy.init()
    sim_lidar2world_node = IsaacLidarNode()
    rclpy.spin(sim_lidar2world_node)
    sim_lidar2world_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
