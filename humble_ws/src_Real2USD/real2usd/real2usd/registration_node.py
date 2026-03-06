import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from geometry_msgs.msg import Pose
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from custom_message.msg import UsdStringIdPoseMsg, UsdStringIdSrcTargMsg
from std_msgs.msg import Float64
import time


class RegistrationNode(Node):
    def __init__(self):
        super().__init__("registration_node")

        # subscribers
        self.sub_src_targ = self.create_subscription(UsdStringIdSrcTargMsg, "/usd/StringIdSrcTarg", self.callback_src_targ, 10)

        # publishers
        self.pub_pose = self.create_publisher(UsdStringIdPoseMsg, "/usd/StringIdPose", 10)

        # debugging publisher
        self.pub_odom_pc_seg = self.create_publisher(PointCloud2, "/registration/pc", 10)

        # Debug visualization publishers
        self.pub_best_match = self.create_publisher(PointCloud2, "/registration/best_match", 10)
        self.timing_pub = self.create_publisher(Float64, "/timing/registration_node", 10)

        # Minimum points after downsampling for FGR (avoids Open3D UniformIntGenerator(0,0) when empty)
        self.min_points_for_fgr = 32

        # Create fields for XYZ and RGB
        self.fields = [
            point_cloud2.PointField(name='x', offset=0, datatype=point_cloud2.PointField.FLOAT32, count=1),
            point_cloud2.PointField(name='y', offset=4, datatype=point_cloud2.PointField.FLOAT32, count=1),
            point_cloud2.PointField(name='z', offset=8, datatype=point_cloud2.PointField.FLOAT32, count=1),
            point_cloud2.PointField(name='rgb', offset=12, datatype=point_cloud2.PointField.FLOAT32, count=1)
        ]
        self.debug_visualization = True

    def callback_src_targ(self, msg):
        t_start = time.perf_counter()
        points_src = np.asarray(
            point_cloud2.read_points_list(
                msg.src_pc, field_names=("x", "y", "z"), skip_nans=True
            )
        )
        points_targ = np.asarray(
            point_cloud2.read_points_list(
                msg.targ_pc, field_names=("x", "y", "z"), skip_nans=True
            )
        )
        usd_url = msg.data_path
        trackId = msg.id

        if len(points_src) != 0 and len(points_targ) != 0:
            t, quatXYZW, points_transformed = self.get_pose(points_src, points_targ)
            if points_transformed is not None:
                self.pub_odom_pc_seg.publish(
                    point_cloud2.create_cloud_xyz32(
                        header=Header(frame_id="odom"), points=np.asarray(points_transformed)
                    )
                )

            if t is not None and quatXYZW is not None:
                # print(quatXYZW)
                # publish data_path, id, and pose as a single message (propagate header.stamp for e2e timing)
                pose_msg = UsdStringIdPoseMsg()
                pose_msg.header = msg.header
                pose_msg.header.frame_id = "odom"
                pose_msg.data_path = usd_url
                pose_msg.id = trackId
                pose = Pose()
                pose.position.x = t[0]
                pose.position.y = t[1]
                pose.position.z = t[2]
                pose.orientation.w = quatXYZW[3]
                pose.orientation.x = quatXYZW[0]
                pose.orientation.y = quatXYZW[1]
                pose.orientation.z = quatXYZW[2]
                pose_msg.pose = pose
                self.pub_pose.publish(pose_msg)
        timing_msg = Float64()
        timing_msg.data = time.perf_counter() - t_start
        self.timing_pub.publish(timing_msg)

    def preprocess_point_cloud(self, pcd, voxel_size, skip_downsample=False):
        # Minimal logging for preprocessing
        if skip_downsample:
            pcd_down = pcd
        else:
            pcd_down = pcd.voxel_down_sample(voxel_size)
        radius_normal = voxel_size * 2
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        radius_feature = voxel_size * 5
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_down, pcd_fpfh

    def visualize_best_match(self, src_points_transformed, targ_points, transformation=None):
        """Visualize the best match between source and target points. src_points_transformed should be the downsampled and yaw-rotated source after registration."""
        if not self.debug_visualization:
            return
        
        # src_points_transformed is already transformed, so just plot as-is
        src_colors = np.tile([1.0, 0.0, 0.0], (len(src_points_transformed), 1))  # Red
        targ_colors = np.tile([0.0, 1.0, 1.0], (len(targ_points), 1))  # Cyan
        
        # Combine points and colors
        all_points = np.vstack((src_points_transformed, targ_points))
        all_colors = np.vstack((src_colors, targ_colors))
        
        # Convert colors to uint32 format for RGB field
        colors_uint32 = np.zeros(len(all_points), dtype=np.uint32)
        for i in range(len(all_points)):
            r, g, b = all_colors[i]
            colors_uint32[i] = (int(r * 255) << 16) | (int(g * 255) << 8) | int(b * 255)
        
        # Combine points and colors into a structured array
        points_with_colors = np.zeros(len(all_points), dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('rgb', np.float32)
        ])
        points_with_colors['x'] = all_points[:, 0]
        points_with_colors['y'] = all_points[:, 1]
        points_with_colors['z'] = all_points[:, 2]
        points_with_colors['rgb'] = colors_uint32.view(np.float32)
        
        msg = point_cloud2.create_cloud(
            header=Header(frame_id="odom"),
            fields=self.fields,
            points=points_with_colors
        )
        self.pub_best_match.publish(msg)

    def get_pose(self, points_src, points_targ, render=False):
        """Register source to full target (no clustering). Try multiple yaw inits; FGR+ICP with min-point guard."""
        voxel_size = 0.01
        target_center = np.mean(points_targ, axis=0)

        src = o3d.geometry.PointCloud()
        src.points = o3d.utility.Vector3dVector(points_src)
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(points_targ)

        skip_downsample_src = len(points_src) < 30
        src_down_orig, source_fpfh_orig = self.preprocess_point_cloud(src, voxel_size=voxel_size, skip_downsample=skip_downsample_src)
        target_down, target_fpfh = self.preprocess_point_cloud(target, voxel_size=voxel_size)

        n_src = len(src_down_orig.points)
        n_targ = len(target_down.points)
        if n_src < 2 or n_targ < 2:
            self.get_logger().warn("Registration skipped: too few points after preprocessing (src=%d, targ=%d)" % (n_src, n_targ))
            return None, None, None

        best_fitness = float('-inf')
        best_translation = None
        best_quaternion = None
        best_points_transformed = None

        yaw_angles = [0, np.pi/2, np.pi, 3*np.pi/2]
        for yaw in yaw_angles:
            rot_matrix = np.array([
                [np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw),  np.cos(yaw), 0],
                [0, 0, 1]
            ])
            src_down = o3d.geometry.PointCloud()
            src_down.points = o3d.utility.Vector3dVector(np.dot(np.asarray(src_down_orig.points), rot_matrix.T))
            source_fpfh = source_fpfh_orig

            trans_init = np.eye(4)
            trans_init[:3, 3] = target_center
            trans_init[:3, :3] = np.eye(3)

            # FGR needs enough points (avoids Open3D UniformIntGenerator(0,0))
            use_fgr = n_src >= self.min_points_for_fgr and n_targ >= self.min_points_for_fgr
            if use_fgr:
                try:
                    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
                        src_down, target_down, source_fpfh, target_fpfh,
                        o3d.pipelines.registration.FastGlobalRegistrationOption(
                            maximum_correspondence_distance=voxel_size * 0.5,
                            iteration_number=64,
                            tuple_scale=0.95,
                            maximum_tuple_count=500))
                    trans_init = result.transformation
                    distance_threshold = 0.03
                except Exception as e:
                    self.get_logger().warn("FGR failed: %s" % str(e))
                    distance_threshold = 0.02
            else:
                distance_threshold = 0.02

            icp_result = o3d.pipelines.registration.registration_icp(
                src_down, target_down, distance_threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))

            if icp_result.fitness > best_fitness:
                best_fitness = icp_result.fitness
                rotation = np.array(icp_result.transformation[0:3, 0:3])
                yaw_final = np.arctan2(rotation[1, 0], rotation[0, 0]) + yaw
                matrix = np.array([
                    [np.cos(yaw_final), -np.sin(yaw_final), 0],
                    [np.sin(yaw_final), np.cos(yaw_final), 0],
                    [0, 0, 1]
                ])
                quaternion = R.from_matrix(matrix).as_quat()
                translation = np.array(icp_result.transformation[0:3, 3])

                src_copy = o3d.geometry.PointCloud()
                src_copy.points = src_down.points
                src_copy.transform(icp_result.transformation)
                best_points_transformed = np.asarray(src_copy.points)
                best_translation = translation
                best_quaternion = quaternion

                self.get_logger().info(
                    "Best match: yaw_init: %.1f deg, fitness: %.3f, translation: %s, yaw: %.1f deg"
                    % (np.degrees(yaw), icp_result.fitness, translation, np.degrees(yaw_final))
                )

        if best_points_transformed is not None:
            self.visualize_best_match(best_points_transformed, points_targ)
            self.get_logger().info("Final best fitness: %.3f" % best_fitness)

        return best_translation, best_quaternion, best_points_transformed


def main():
    rclpy.init()
    registration_node = RegistrationNode()
    rclpy.spin(registration_node)
    registration_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
