import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
import struct
from sensor_msgs.msg import PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
import open3d as o3d
import time

class ProjectionUtils:
    def __init__(self, T):

        self.T_cam_in_odom = T

        # odom to camera frame
        static_cam_offset_euler = np.radians([-90, 0, 90]) # Example: X-90, Y-0, Z+90 (XYZ order)
        self.R_static_odom_to_cam = R.from_euler('xyz', static_cam_offset_euler, degrees=False).as_matrix()
        # additional rotation to make the camera frame point forward
        R_additional_rotation = R.from_euler('xyz', np.radians([0, 0, 180]), degrees=False).as_matrix()
        self.R_static_odom_to_cam = R_additional_rotation @ self.R_static_odom_to_cam 

        self.T_odom_from_cam = self._create_homogeneous_matrix(self.R_static_odom_to_cam, self.T_cam_in_odom)

        self.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1)
        ]

    def _create_homogeneous_matrix(self, rotation_obj, translation_vec):
        """Helper to create a 4x4 homogeneous transformation matrix."""
        matrix = np.eye(4)
        matrix[:3, :3] = rotation_obj
        matrix[:3, 3] = translation_vec
        return matrix

    def lidar2depth(self, lidar_pts, cam_info, odom_info):
        """
        convert lidar points in world into the camera frame
        """
        img_height, img_width = (
            cam_info["height"],
            cam_info["width"],
        )  # Replace with your camera resolution
        depth_image = np.zeros((img_height, img_width), dtype=np.uint16)
        # 1. Get Odometry Frame's Pose in World Frame (T_world_from_odom)
        # This is the pose of the Odom frame's origin (t) and orientation (R) within the World frame.
        R_world_from_odom = R.from_quat(odom_info["q"]).as_matrix()
        t_world_from_odom = odom_info["t"]
        T_world_from_odom = self._create_homogeneous_matrix(R_world_from_odom, t_world_from_odom)

        T_cam_from_world = np.linalg.inv(self.T_odom_from_cam) @ np.linalg.inv(T_world_from_odom)

        # 4. Transform Lidar Points from World to Camera Frame
        lidar_pts_homogeneous = np.hstack((lidar_pts, np.ones((lidar_pts.shape[0], 1))))

        # Apply the transformation
        lidar_cam_homogeneous = (T_cam_from_world @ lidar_pts_homogeneous.T).T
        lidar_cam = lidar_cam_homogeneous[:, :3] # Remove the homogeneous coordinate

        # --- Intrinsics (This part is generally correct) ---
        K = cam_info["K"]

        # Filter out points behind the camera (Z > 0)
        lidar_cam = lidar_cam[lidar_cam[:, 2] > 0]
        if lidar_cam.shape[0] == 0:
            return depth_image, None, None # No points in front of camera

        """Intrinsics"""
        K = cam_info["K"]
        # filter out points behind the camera
        lidar_cam = lidar_cam[lidar_cam[:, 2] > 0]
        # project to image plane
        uvw = K @ lidar_cam.T
        # normalize by w
        uv = uvw[:2] / uvw[2]

        # where Z is the distance from the image plane and scaling by 256.0 for int16
        uvz = np.vstack((uv, lidar_cam[:, 2] * 256.0)).T
        # filter out points outside the image
        uvz_in = uvz[
            (0 <= uvz[:, 0])
            & (uvz[:, 0] < img_width)
            & (0 <= uvz[:, 1])
            & (uvz[:, 1] < img_height)
        ]
        # put valid points in image
        depth_image[uvz_in[:, 1].astype(int), uvz_in[:, 0].astype(int)] = uvz_in[:, 2]

        # debugging overlay
        depth_norm = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_8bit = depth_norm.astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)
        mask = depth_image > 0

        return depth_image.astype(np.uint16), depth_color, mask

    def twoDtoThreeD(self, twoD_pts, depth_img, cam_info, odom_info):
        """ 
        given 2d points in camera frame and project them into the world frame

        """
        u = np.clip(twoD_pts[:, 0].astype(int), 0, depth_img.shape[1] - 1)
        v = np.clip(twoD_pts[:, 1].astype(int), 0, depth_img.shape[0] - 1)
        Z = (
            depth_img[v.astype(int), u.astype(int)] / 256.0
        )  # d/256 converts from uint16 to meters
        uv1 = np.vstack((u, v, np.ones_like(u)))

        # filter out points with zero depth
        mask = Z > 0
        uv1 = uv1[:,mask]
        Z = Z[mask]

        # for degbugging this function, just try to project all non zero depth points
        # v, u = np.nonzero(depth_img)
        # Z = depth_img[v, u] / 256.0
        # uv1 = np.vstack((u, v, np.ones_like(u)))

        Kinv = np.linalg.inv(cam_info["K"])
        xyz = Kinv @ uv1 * Z
        points = np.array([xyz[0], xyz[1], xyz[2]]).T

        R_world_from_odom = R.from_quat(odom_info["q"]).as_matrix()
        t_world_from_odom = odom_info["t"]
        T_world_from_odom = self._create_homogeneous_matrix(R_world_from_odom, t_world_from_odom)

        T_cam_from_world = np.linalg.inv(self.T_odom_from_cam) @ np.linalg.inv(T_world_from_odom)
        T_world_from_cam = np.linalg.inv(T_cam_from_world)

        h_points = np.hstack((points, np.ones((points.shape[0], 1))))
        # rotate/translate points from camera frame to world frame
        lidar_odom = (T_world_from_cam @ h_points.T).T
        lidar_odom = lidar_odom[:, :3]/lidar_odom[:, 3, np.newaxis]

        # filter out points that have -Z
        lidar_odom = lidar_odom[lidar_odom[:, 2] > 0]
        
        return lidar_odom

    def twoDtoThreeDColor(self, depth_img, rgb_img, cam_info, odom_info):
        """ 
        given 2d points in camera frame and project them into the world frame with the color from the image.
        For multiple 3D points projecting to the same pixel:
        - Surface point (closest): Original color
        - All other points: Light grey
        """
        # Get all non-zero depth points and their coordinates
        v, u = np.nonzero(depth_img)
        
        # Return empty array if no points found
        if len(v) == 0 or len(u) == 0:
            color_pc_msg = point_cloud2.create_cloud(header=Header(frame_id="odom"), fields=self.fields, points=np.empty((0, 4), dtype=np.float32))
            return np.empty((0, 4), dtype=np.float32), color_pc_msg
            
        Z = depth_img[v, u] / 256.0  # Convert from uint16 to meters
        
        # Create homogeneous coordinates
        uv1 = np.vstack((u, v, np.ones_like(u)))
        
        # Get the color of the points
        rgb = rgb_img[v.astype(int), u.astype(int), :]
        
        # Convert rgb [0,255] to uint32 and then to float
        rgb_uint32 = (rgb[:,0].astype(np.uint32) << 16) | (rgb[:,1].astype(np.uint32) << 8) | rgb[:,2].astype(np.uint32)
        rgb_float = np.array([struct.unpack('f', struct.pack('I', val))[0] for val in rgb_uint32], dtype=np.float32)
        
        # Project to 3D
        Kinv = np.linalg.inv(cam_info["K"])
        xyz = Kinv @ uv1 * Z
        points = np.array([xyz[0], xyz[1], xyz[2]]).T
        
        R_world_from_odom = R.from_quat(odom_info["q"]).as_matrix()
        t_world_from_odom = odom_info["t"]
        T_world_from_odom = self._create_homogeneous_matrix(R_world_from_odom, t_world_from_odom)

        T_cam_from_world = np.linalg.inv(self.T_odom_from_cam) @ np.linalg.inv(T_world_from_odom)
        T_world_from_cam = np.linalg.inv(T_cam_from_world)

        h_points = np.hstack((points, np.ones((points.shape[0], 1))))
        # rotate/translate points from camera frame to world frame
        lidar_color = (T_world_from_cam @ h_points.T).T
        lidar_color = lidar_color[:, :3]/lidar_color[:, 3, np.newaxis]

        # Filter out points with negative Z
        valid_mask = lidar_color[:, 2] > 0
        if not np.any(valid_mask):
            color_pc_msg = point_cloud2.create_cloud(header=Header(frame_id="odom"), fields=self.fields, points=np.empty((0, 4), dtype=np.float32))
            return np.empty((0, 4), dtype=np.float32), color_pc_msg
            
        lidar_color = lidar_color[valid_mask]
        rgb_float = rgb_float[valid_mask]
        u = u[valid_mask]
        v = v[valid_mask]
        Z = Z[valid_mask]
        
        # Create a unique identifier for each pixel (u,v)
        pixel_ids = v * depth_img.shape[1] + u
        
        # Create a structured array for sorting
        dtype = [('pixel_id', int), ('distance', float), ('point', float, 3), ('color', float)]
        structured_array = np.zeros(len(pixel_ids), dtype=dtype)
        structured_array['pixel_id'] = pixel_ids
        structured_array['distance'] = Z  # Distance from camera
        structured_array['point'] = lidar_color
        structured_array['color'] = rgb_float
        
        # Sort by pixel_id and distance (ascending)
        sorted_indices = np.argsort(structured_array, order=('pixel_id', 'distance'))
        sorted_array = structured_array[sorted_indices]
        
        # Find the first occurrence of each pixel_id (closest point)
        unique_pixel_mask = np.concatenate(([True], sorted_array['pixel_id'][1:] != sorted_array['pixel_id'][:-1]))
        
        # Create light grey color
        light_grey_rgb = (192 << 16) | (192 << 8) | 192  # Light grey
        light_grey_float = struct.unpack('f', struct.pack('I', light_grey_rgb))[0]
        
        # Initialize all points to light grey
        all_colors = np.full(len(sorted_array), light_grey_float, dtype=np.float32)
        
        # Set surface points to their original colors
        all_colors[unique_pixel_mask] = sorted_array['color'][unique_pixel_mask]
        
        # Combine points and colors
        lidar_color = np.hstack((sorted_array['point'], all_colors.reshape(-1,1)))
    
        color_pc_msg = point_cloud2.create_cloud(header=Header(frame_id="odom"), fields=self.fields, points=lidar_color)

        return lidar_color, color_pc_msg
        


class PointCloudBuffer:
    def __init__(self, max_points=10000, voxel_size=0.01):
        """
        double ended queue.
        if set to a fix length the oldest points are removed
        """
        self.max_points = max_points
        # self.buffer = deque(maxlen=max_points)
        self.buffer = np.empty((0,3))
        self.pcd = o3d.geometry.PointCloud()
        self.voxel_size = voxel_size
        self.last_time = time.time()

    def add_points(self, points):
        # self.buffer.extend(points)
        self.buffer = np.concatenate((self.buffer, points), axis=0)
        self.pcd.points = o3d.utility.Vector3dVector(np.array(self.buffer))
        # Downsample the point cloud
        downsampled_pcd = self.pcd.voxel_down_sample(self.voxel_size)
        self.buffer = np.array(downsampled_pcd.points)

    def get_points(self):
        return np.array(self.buffer)

    def clear(self):
        self.buffer = np.empty((0,3))
        self.pcd = o3d.geometry.PointCloud()