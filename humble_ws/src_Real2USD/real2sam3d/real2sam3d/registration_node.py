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
from ipdb import set_trace as st
from sklearn.cluster import DBSCAN
import random


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
        self.pub_clusters = self.create_publisher(PointCloud2, "/registration/clusters", 10)
        self.pub_best_match = self.create_publisher(PointCloud2, "/registration/best_match", 10)
        
        # Debug visualization parameters
        self.debug_visualization = True  # Set to False to disable debug visualization

        # Tukey loss function, huber loss with a flat
        sigma = 0.05  # Reduced from 0.1 to be more sensitive to small errors
        self.loss = o3d.pipelines.registration.TukeyLoss(k=sigma)

        # DBSCAN parameters - adjusted for larger scale
        self.eps = 0.2  # Increased to handle large objects like tables/couches
        self.min_samples = 20  # Increased for robustness, fits large objects
        
        # Minimum points required for FGR
        self.min_points_for_fgr = 30  # Minimum points needed for FGR to work
        
        # Weight parameters for point weighting
        self.max_weight_distance = 2.0  # Reduced from 5.0 to match smaller scale
        self.min_weight = 0.2  # Increased from 0.1 to give more weight to far points

        # Create fields for XYZ and RGB
        self.fields = [
            point_cloud2.PointField(name='x', offset=0, datatype=point_cloud2.PointField.FLOAT32, count=1),
            point_cloud2.PointField(name='y', offset=4, datatype=point_cloud2.PointField.FLOAT32, count=1),
            point_cloud2.PointField(name='z', offset=8, datatype=point_cloud2.PointField.FLOAT32, count=1),
            point_cloud2.PointField(name='rgb', offset=12, datatype=point_cloud2.PointField.FLOAT32, count=1)
        ]

        self.cluster_colors = {}  # Store colors for each cluster

    def callback_src_targ(self, msg):
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
                # publish data_path, id, and pose as a single message
                msg = UsdStringIdPoseMsg()
                msg.header = Header(frame_id="odom")
                msg.data_path = usd_url
                msg.id = trackId
                pose = Pose()
                pose.position.x = t[0]
                pose.position.y = t[1]
                pose.position.z = t[2]
                pose.orientation.w = quatXYZW[3]
                pose.orientation.x = quatXYZW[0]
                pose.orientation.y = quatXYZW[1]
                pose.orientation.z = quatXYZW[2]
                msg.pose = pose
                self.pub_pose.publish(msg)

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

    def cluster_target_points(self, points):
        """Cluster target points using DBSCAN and return cluster labels."""
        # Only minimal logging for clustering
        eps = self.eps
        clustering = DBSCAN(eps=eps, min_samples=self.min_samples).fit(points)
        labels = clustering.labels_
        unique_labels = np.unique(labels)
        # self.get_logger().info(f"Found {len(unique_labels)} clusters (including noise)")
        for label in unique_labels:
            if label == -1:
                continue
            # self.get_logger().info(f"Cluster {label} size: {np.sum(labels == label)}")
        return labels

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

    def visualize_best_cluster(self, points, labels, best_cluster_id):
        """Visualize all clusters, but color the best cluster in red."""
        if not self.debug_visualization:
            return
        colors = np.zeros((len(points), 3))
        for i, label in enumerate(labels):
            if label == best_cluster_id:
                colors[i] = [1.0, 0.0, 0.0]  # Red for best cluster
            elif label == -1:
                colors[i] = [0.5, 0.5, 0.5]  # Gray for noise
            else:
                # Use stored color if available, else random
                colors[i] = self.cluster_colors.get(label, [0.0, 1.0, 0.0])
        # Convert colors to uint32 format for RGB field
        colors_uint32 = np.zeros(len(points), dtype=np.uint32)
        for i in range(len(points)):
            r, g, b = colors[i]
            colors_uint32[i] = (int(r * 255) << 16) | (int(g * 255) << 8) | int(b * 255)
        points_with_colors = np.zeros(len(points), dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('rgb', np.float32)
        ])
        points_with_colors['x'] = points[:, 0]
        points_with_colors['y'] = points[:, 1]
        points_with_colors['z'] = points[:, 2]
        points_with_colors['rgb'] = colors_uint32.view(np.float32)
        msg = point_cloud2.create_cloud(
            header=Header(frame_id="odom"),
            fields=self.fields,
            points=points_with_colors
        )
        self.pub_clusters.publish(msg)

    def get_pose(self, points_src, points_targ, render=False):
        cluster_labels = self.cluster_target_points(points_targ)
        unique_clusters = np.unique(cluster_labels)
        
        best_fitness = float('-inf')
        best_translation = None
        best_quaternion = None
        best_points_transformed = None
        best_transformation = None
        best_cluster_id = None
        
        # Try registering against each cluster
        for cluster_id in unique_clusters:
            if cluster_id == -1:  # Skip noise points
                continue
                
            # Get points for this cluster
            cluster_mask = cluster_labels == cluster_id
            cluster_points = points_targ[cluster_mask]
            
            if len(cluster_points) < self.min_samples:
                # Only warn if skipping due to too few points
                # self.get_logger().warn(f"Skipping cluster {cluster_id} - too few points: {len(cluster_points)}")
                continue
                
            # Create point clouds
            src = o3d.geometry.PointCloud()
            src.points = o3d.utility.Vector3dVector(points_src)
            
            target = o3d.geometry.PointCloud()
            target.points = o3d.utility.Vector3dVector(cluster_points)
            
            # Initial transformation using cluster centroid
            target_center = np.mean(cluster_points, axis=0)
            
            voxel_size = 0.01  # Fixed voxel size for both source and target
            # Preprocess source: skip downsampling if already small
            skip_downsample_src = len(points_src) < 30
            src_down_orig, source_fpfh_orig = self.preprocess_point_cloud(src, voxel_size=voxel_size, skip_downsample=skip_downsample_src)
            target_down, target_fpfh = self.preprocess_point_cloud(target, voxel_size=voxel_size)

            # Skip registration if source cloud is too small after preprocessing
            if len(src_down_orig.points) < 30:
                # self.get_logger().warn(f"Skipping registration for cluster {cluster_id}: source cloud too small after downsampling ({len(src_down_orig.points)} points)")
                continue

            # Try multiple initial yaws
            yaw_angles = [0, np.pi/2, np.pi, 3*np.pi/2]
            for yaw in yaw_angles:
                # Apply yaw rotation to source
                rot_matrix = np.array([
                    [np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw),  np.cos(yaw), 0],
                    [0, 0, 1]
                ])
                src_down = o3d.geometry.PointCloud()
                src_down.points = o3d.utility.Vector3dVector(np.dot(np.asarray(src_down_orig.points), rot_matrix.T))
                source_fpfh = source_fpfh_orig  # FPFH is rotation invariant

                # Initial transformation (translation only)
                trans_init = np.eye(4)
                trans_init[:3, 3] = target_center
                trans_init[:3, :3] = np.eye(3)

                # Skip FGR for very small clusters
                if len(cluster_points) < 30:
                    # self.get_logger().warn(f"Cluster {cluster_id} too small for FGR, using direct ICP")
                    distance_threshold = 0.02
                else:
                    try:
                        result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
                            src_down, target_down, source_fpfh, target_fpfh,
                            o3d.pipelines.registration.FastGlobalRegistrationOption(
                                maximum_correspondence_distance=voxel_size * 0.5,
                                iteration_number=64,
                                tuple_scale=0.95,
                                maximum_tuple_count=500))
                        trans_init = result.transformation
                        # Only log FGR success if needed
                        distance_threshold = 0.03
                    except Exception as e:
                        self.get_logger().warn(f"FGR failed for cluster {cluster_id}: {str(e)}")
                        distance_threshold = 0.02

                # Perform ICP with fixed parameters
                icp_result = o3d.pipelines.registration.registration_icp(
                    src_down, target_down, distance_threshold, trans_init,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))

                # Only log best match summary
                # Check if this is the best result so far
                if icp_result.fitness > best_fitness:
                    best_fitness = icp_result.fitness
                    rotation = np.array(icp_result.transformation[0:3, 0:3])
                    yaw_final = np.arctan2(rotation[1,0], rotation[0,0]) + yaw  # Add initial yaw
                    matrix = np.array([
                        [np.cos(yaw_final), -np.sin(yaw_final), 0],
                        [np.sin(yaw_final), np.cos(yaw_final), 0],
                        [0, 0, 1]
                    ])
                    quaternion = R.from_matrix(matrix).as_quat()
                    translation = np.array(icp_result.transformation[0:3, 3])

                    # Create transformation matrix
                    transformation = np.eye(4)
                    transformation[0:3,3] = translation
                    transformation[0:3,0:3] = matrix

                    # Transform source points for visualization
                    src_copy = o3d.geometry.PointCloud()
                    src_copy.points = src_down.points
                    src_copy.transform(transformation)
                    best_points_transformed = np.asarray(src_copy.points)
                    best_translation = translation
                    best_quaternion = quaternion
                    best_transformation = transformation
                    best_cluster_id = cluster_id

                    self.get_logger().info(f"New best match: Cluster {cluster_id}, yaw_init: {np.degrees(yaw):.1f} deg, fitness: {icp_result.fitness:.3f}, translation: {translation}, yaw: {np.degrees(yaw_final):.1f} deg")
        
        if best_points_transformed is not None:
            # Visualize clusters again, highlighting the best cluster in red
            self.visualize_best_cluster(points_targ, cluster_labels, best_cluster_id)
            self.visualize_best_match(best_points_transformed, points_targ)
            self.get_logger().info(f"Final best fitness: {best_fitness:.3f}")
        else:
            # self.get_logger().warn("No valid registration found!")
            pass
        
        return best_translation, best_quaternion, best_points_transformed


def main():
    rclpy.init()
    registration_node = RegistrationNode()
    rclpy.spin(registration_node)
    registration_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
