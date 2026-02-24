import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from sensor_msgs_py import point_cloud2
from go2_interfaces.msg import Go2State, IMU
from nav_msgs.msg import Odometry
# from custom_message.msg import UsdStringIdPCMsg
from custom_message.msg import CropImgDepthMsg
from std_msgs.msg import Header, Int64, Float64
import cv2
from cv_bridge import CvBridge
import json, asyncio, time, pickle
import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d

from scripts_r2u.segment_cls import Segmentation
from scripts_r2u.utils import ProjectionUtils
from scripts_r2u.utils import PointCloudBuffer

from ipdb import set_trace as st

"""
The idea is that lidar is the rate limiter so we use the lidar callback as the main driver callback.

The messages are based on go2_ros2_sdk repo which subscribes to webrtc topics from the unitree go2 robot directly.
"""

class LidarDriverNode(Node):
    def __init__(self):
        super().__init__("lidar_driver_node")

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
        self.timing_pub = self.create_publisher(Float64, "/timing/lidar_cam_node", 10)

        # prompted model
        model_path = "models/yoloe-11l-seg.pt"
        # prompt free model
        # model_path = "models/yoloe-11l-seg-pf.pt"
        self.segment = Segmentation(model_path)

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

    def lidar_listener_callback(self, msg):
        """Extract the points from the PointCloud2 message
        return (N,3) array of lidar points from buffer
        """
        lidar_pts = np.asarray(
            point_cloud2.read_points_list(
                msg, field_names=("x", "y", "z"), skip_nans=True
            )
        )
        # add points to buffer
        self.points.add_points(lidar_pts)


        if lidar_pts.shape[0] > 0 and self.cam_info is not None and self.rgb_image is not None and self.odom_info["t"] is not None:    
            # Snapshot once so all data for this frame stays consistent even if callbacks overwrite self during processing.
            frame_stamp = msg.header.stamp
            frame_odom = {
                "t": np.array(self.odom_info["t"], dtype=np.float64),
                "q": np.array(self.odom_info["q"], dtype=np.float64),
                "eulerXYZ": np.array(self.odom_info["eulerXYZ"], dtype=np.float64) if self.odom_info.get("eulerXYZ") is not None else None,
            }
            frame_rgb = self.rgb_image.copy()
            frame_cam_info = {k: np.array(v, copy=True) if isinstance(v, np.ndarray) else v for k, v in self.cam_info.items()}

            t_start = time.perf_counter()
            # Process the lidar points into 2D camera frame and publish the depth image       
            depth_image, depth_color, mask = self.projection.lidar2depth(lidar_pts, frame_cam_info, frame_odom)
            depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding="16UC1")
            # msg.header.stamp = lidar_msg.header.stamp
            self.depth_pub.publish(depth_msg)

            # overlay the depth image on the rgb image and publish the rgbd image
            try:
                overlay = cv2.cvtColor(frame_rgb.copy(), cv2.COLOR_RGB2BGR)
                overlay[mask] = cv2.addWeighted(overlay[mask], 0.0, depth_color[mask], 1.0, 0)
                self.rgbd_pub.publish(self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8"))
            except:
                pass

            # publish the lidar points with color in the odom frame
            _, color_pc_msg = self.projection.twoDtoThreeDColor(depth_image, frame_rgb, frame_cam_info, frame_odom)
            self.color_pc_pub.publish(color_pc_msg)

            # Segment the image and publish the segmentation output and tracks
            result, _, imgs_crop, _, mask_pts, track_ids, labels, labels_usd = self.segment.crop_img_w_bbox(frame_rgb, conf=0.5, iou=0.2)
            # publish annotated segmented image
            img_result_msg = self.bridge.cv2_to_imgmsg(result, encoding="bgr8")
            self.get_logger().info(f"labels: {labels}")
            self.seg_pub.publish(img_result_msg)

            # One stamp per frame so timing_node can measure time-per-frame (CropImgDepth → last StringIdPose for this frame)
            # publish the cropped image, depth image, and point cloud (same stamp and odom for whole frame)
            for ii in range(len(mask_pts)):
                # Create new CropImgDepthMsg
                crop_msg = CropImgDepthMsg()
                
                # Set header (same stamp for all objects in this frame)
                crop_msg.header.stamp = frame_stamp
                crop_msg.header.frame_id = "odom"
                
                # Set RGB image
                crop_msg.rgb_image = self.bridge.cv2_to_imgmsg(imgs_crop[ii], encoding="bgr8")
                crop_msg.rgb_image.header = crop_msg.header
                
                # Set segmentation points as flat array
                crop_msg.seg_points = mask_pts[ii].astype(int).flatten().tolist()
                
                # Set depth image
                crop_msg.depth_image = self.bridge.cv2_to_imgmsg(depth_image, encoding="16UC1")
                crop_msg.depth_image.header = crop_msg.header
                
                # Set camera info (frame snapshot)
                camera_info_msg = CameraInfo()
                camera_info_msg.header = crop_msg.header
                camera_info_msg.k = frame_cam_info["K"].flatten().tolist()
                camera_info_msg.p = frame_cam_info["P"].flatten().tolist()
                camera_info_msg.width = frame_cam_info["width"]
                camera_info_msg.height = frame_cam_info["height"]
                crop_msg.camera_info = camera_info_msg
                
                # Set odometry info (frame snapshot)
                odom_msg = Odometry()
                odom_msg.header = crop_msg.header
                odom_msg.pose.pose.position.x = float(frame_odom["t"][0])
                odom_msg.pose.pose.position.y = float(frame_odom["t"][1])
                odom_msg.pose.pose.position.z = float(frame_odom["t"][2])
                odom_msg.pose.pose.orientation.x = float(frame_odom["q"][0])
                odom_msg.pose.pose.orientation.y = float(frame_odom["q"][1])
                odom_msg.pose.pose.orientation.z = float(frame_odom["q"][2])
                odom_msg.pose.pose.orientation.w = float(frame_odom["q"][3])
                crop_msg.odometry = odom_msg
                
                # Set track ID and label
                crop_msg.track_id = track_ids[ii]
                crop_msg.label = labels_usd[ii]
                
                # Publish the message
                self.crop_rgb_depth_pub.publish(crop_msg)
            t_end = time.perf_counter()
            timing_msg = Float64()
            timing_msg.data = t_end - t_start
            self.timing_pub.publish(timing_msg)

    def rgb_listener_callback(self, msg):
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def global_pcd_callback(self):
        global_pc_msg = point_cloud2.create_cloud_xyz32(header=Header(frame_id="odom"), points=self.points.get_points())
        global_pc_msg.header.stamp = self.get_clock().now().to_msg()
        self.global_pcd_pub.publish(global_pc_msg)

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

        #save the odom buffer as a pickle file
        # if len(self.odom_buffer) % 100 == 0:
        #     with open("odom_buffer.pkl", "wb") as f:
        #         pickle.dump(self.odom_buffer, f)

    def cam_info_listener_callback(self, msg):
        # intrinsics
        K = msg.k.reshape(3, 3)  # intrinsics
        R = msg.r  # Rectification Matrix, only used for stereo cameras
        P = msg.p.reshape(3, 4)  # projection matrix
        width = msg.width
        height = msg.height

        cam_info = {"K": K, "R": R, "P": P, "width": width, "height": height}
        self.cam_info = cam_info
    

def main():
    rclpy.init()
    lidar_driver_node = LidarDriverNode()
    rclpy.spin(lidar_driver_node)
    lidar_driver_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
