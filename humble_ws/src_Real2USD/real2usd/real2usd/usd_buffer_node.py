import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from geometry_msgs.msg import Pose
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull, Delaunay, KDTree
from pxr import Usd, UsdGeom, Gf
from custom_message.msg import UsdStringIdPoseMsg, UsdBufferPoseMsg
import struct
import json
import os
from datetime import datetime
from collections import defaultdict

class USDBufferNode(Node):
    """
    A ROS2 node that manages a buffer of USD objects and matches them with point cloud data.
    
    This node:
    1. Receives USD object poses and point cloud data
    2. Clusters objects based on spatial proximity and semantic labels
    3. Matches objects with point cloud data using occupancy scoring
    4. Publishes matched objects and visualization data
    5. Periodically saves the matched buffer to disk
    """
    
    def __init__(self):
        super().__init__("usd_buffer_node")
        
        # Initialize subscribers and publishers
        self._init_ros_interface()
        
        # Initialize data structures
        self._init_data_structures()
        
        # Initialize timers
        self._init_timers()
        
        # Initialize configuration
        self._init_config()

    def _init_ros_interface(self):
        """Initialize ROS subscribers and publishers"""
        # Subscribers
        self.usd_sub = self.create_subscription(
            UsdStringIdPoseMsg, 
            "/usd/StringIdPose", 
            self.usd_buffer_callback, 
            10
        )
        self.global_pcd_sub = self.create_subscription(
            PointCloud2, 
            "/global_lidar_points", 
            self.global_pcd_callback, 
            10
        )

        # Publishers
        self.pub_pcd = self.create_publisher(PointCloud2, "/usd/pcd_occupied", 10)
        self.pub_usd_pose = self.create_publisher(UsdBufferPoseMsg, "/usd/SimUsdPoseBuffer", 10)

    def _init_data_structures(self):
        """Initialize data structures for object tracking and point cloud processing"""
        # Object buffer and spatial tracking
        self.object_buffer = []  # List of object dictionaries
        self.object_positions = np.empty((0, 3))  # Array of object positions
        self.object_tree = None  # KD-tree for object positions
        self.object_clusters = defaultdict(list)  # Maps cluster_id to list of object indices
        self.cluster_centers = {}  # Maps cluster_id to center position
        self.next_cluster_id = 0
        self.last_processed_buffer_size = 0
        
        # Point cloud data structures
        self.point_cloud_points = np.empty((0, 3))
        self.point_cloud_tree = None
        self.occupied_points = set()
        self.used_point_indices_by_cluster = {}  # Track which points have been used by each cluster
        
        # Point cloud message fields (defined once)
        self.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1)
        ]
        
        # USD cache
        self.usd_cache = {}  # Cache for USD vertices and their bounding boxes

    def _init_timers(self):
        """Initialize ROS timers for periodic tasks"""
        self.timer_association = self.create_timer(0.1, self.check_for_new_objects)
        self.timer_publisher = self.create_timer(1.0, self.publish_matched_objects)
        self.timer_save_buffer = self.create_timer(30.0, self.save_matched_buffer)

    def _init_config(self):
        """Initialize configuration parameters"""
        # Spatial clustering parameters
        self.position_tolerance = 5.0  # Distance threshold for considering objects as part of same cluster
        
        # Point cloud processing parameters
        self.point_tolerance = 0.01
        self.ground_plane_height_threshold = 0.1  # Points below this height are considered ground
        
        # Scoring parameters
        self.expected_point_density = 1000  # points per cubic meter
        self.voxel_size = 0.1  # 10cm voxels for distribution analysis
        self.density_weight = 0.6  # weight for density score
        self.distribution_weight = 0.4  # weight for distribution score
        self.min_score_threshold = 0.3  # minimum combined score to consider a match valid
        self.min_points_threshold = 10  # minimum number of points required for a match
        
        # Label extraction configuration
        self.use_second_to_last_label = False
        self.use_third_to_last_label = True
        
        # Save directory configuration
        self.save_dir = "/data/SimIsaacData/buffer"
        os.makedirs(self.save_dir, exist_ok=True)

    def usd_buffer_callback(self, msg):
        """
        Handle incoming USD object messages and update the buffer.
        Processes new objects and updates spatial clustering.
        """
        self.update_object_buffer(
            msg.id,
            msg.data_path,
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            [msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z]
        )

    def update_object_buffer(self, obj_id, usd_file_path, position, quatWXYZ):
        """
        Add new object to buffer and update spatial clustering.
        
        Args:
            obj_id: Unique identifier for the object
            usd_file_path: Path to the USD file
            position: [x, y, z] position in world coordinates
            quatWXYZ: [w, x, y, z] quaternion orientation
        """
        euler = R.from_quat([quatWXYZ[1], quatWXYZ[2], quatWXYZ[3], quatWXYZ[0]]).as_euler('xyz')
        position = np.array(position)

        # Get semantic label for the new object
        object_label = self.get_object_label(usd_file_path)

        # Create new object entry
        new_obj = {
            'usd': usd_file_path,
            'position': position,
            'quatWXYZ': np.array(quatWXYZ),
            'orientation': np.array(euler),
            'cluster_id': None,  # Will be set by spatial clustering
            'matched': False,
            'points_in_hull': None,
            'occupancy_score': 0.0,
            'label': object_label
        }

        # Add to object buffer
        obj_idx = len(self.object_buffer)
        self.object_buffer.append(new_obj)

        # Update spatial tracking
        self.object_positions = np.vstack((self.object_positions, position))
        self.object_tree = KDTree(self.object_positions)

        # Find nearby objects and update clustering
        if len(self.object_positions) > 1:
            self._update_clustering(obj_idx, position, object_label)
        else:
            # No existing clusters, create new cluster
            self._create_new_cluster(obj_idx, position)

    def _update_clustering(self, obj_idx, position, object_label):
        """
        Update clustering for a new object.
        
        Args:
            obj_idx: Index of the new object in object_buffer
            position: Position of the new object
            object_label: Semantic label of the new object
        """
        # Find nearest neighbor
        distances, indices = self.object_tree.query(position, k=2)  # k=2 because first result is self
        if len(distances) > 1 and distances[1] < self.position_tolerance:
            # Find the cluster of the nearest object
            nearest_idx = indices[1]
            nearest_obj = self.object_buffer[nearest_idx]
            nearest_cluster = nearest_obj['cluster_id']
            
            # Only merge if labels match
            if nearest_obj['label'] == object_label:
                if nearest_cluster is not None:
                    # Add to existing cluster
                    self.object_buffer[obj_idx]['cluster_id'] = nearest_cluster
                    self.object_clusters[nearest_cluster].append(obj_idx)
                    # Update cluster center
                    self.cluster_centers[nearest_cluster] = position
                else:
                    # Create new cluster with both objects
                    self._create_new_cluster_with_objects(obj_idx, nearest_idx, position)
            else:
                # Labels don't match, create new cluster
                self._create_new_cluster(obj_idx, position)
        else:
            # No nearby clusters, create new cluster
            self._create_new_cluster(obj_idx, position)

    def _create_new_cluster_with_objects(self, obj_idx1, obj_idx2, position):
        """Create a new cluster containing two objects"""
        new_cluster_id = self.next_cluster_id
        self.next_cluster_id += 1
        
        # Add both objects to the new cluster
        self.object_buffer[obj_idx1]['cluster_id'] = new_cluster_id
        self.object_buffer[obj_idx2]['cluster_id'] = new_cluster_id
        self.object_clusters[new_cluster_id] = [obj_idx1, obj_idx2]
        self.cluster_centers[new_cluster_id] = position
        
        self.get_logger().info(
            f"Created new cluster {new_cluster_id} for objects "
            f"{self.object_buffer[obj_idx1]['usd']} and {self.object_buffer[obj_idx2]['usd']}"
        )

    def _create_new_cluster(self, obj_idx, position):
        """Create a new cluster for a single object"""
        new_cluster_id = self.next_cluster_id
        self.next_cluster_id += 1
        self.object_buffer[obj_idx]['cluster_id'] = new_cluster_id
        self.object_clusters[new_cluster_id] = [obj_idx]
        self.cluster_centers[new_cluster_id] = position
        self.get_logger().info(
            f"Created new cluster {new_cluster_id} for object {self.object_buffer[obj_idx]['usd']}"
        )

    def filter_ground_plane(self, points):
        """
        Filter out points that are below the ground plane height threshold.
        Returns points that are above the ground plane.
        """
        # Simple height-based filtering
        return points[points[:, 2] > self.ground_plane_height_threshold]

    def global_pcd_callback(self, msg):
        """
        Handle incoming point cloud messages and update our point cloud data structures.
        Uses KD-tree for efficient spatial queries and a set for tracking occupied points.
        Filters out ground plane points.
        """
        # Convert incoming message to numpy array
        new_points = np.asarray(
            point_cloud2.read_points_list(
                msg, field_names=("x", "y", "z"), skip_nans=True
            )
        )
        
        if len(new_points) == 0:
            return

        # Filter out ground plane points
        new_points = new_points[new_points[:, 2] > self.ground_plane_height_threshold]
        if len(new_points) == 0:
            return

        # If this is our first point cloud, initialize everything
        if len(self.point_cloud_points) == 0:
            self.point_cloud_points = new_points
            self.point_cloud_tree = KDTree(new_points)
            self._publish_point_cloud(new_points, np.ones((len(new_points), 3)))
            return

        # Find new points that aren't already in our point cloud
        if self.point_cloud_tree is not None:
            # Query KD-tree to find distances to nearest neighbors
            distances, _ = self.point_cloud_tree.query(new_points, k=1, distance_upper_bound=self.point_tolerance)
            
            # Points are new if they're not within tolerance of any existing point
            new_point_mask = distances > self.point_tolerance
            new_points = new_points[new_point_mask]

        if len(new_points) > 0:
            # Add new points to our point cloud
            self.point_cloud_points = np.vstack((self.point_cloud_points, new_points))
            
            # Rebuild KD-tree with all points
            self.point_cloud_tree = KDTree(self.point_cloud_points)
            
            # Update visualization
            self._update_visualization()

    def _update_visualization(self):
        """Update point cloud colors based on object matches and their quality scores"""
        if len(self.point_cloud_points) == 0:
            return

        # Initialize all points as white (unassigned)
        colors = np.ones((len(self.point_cloud_points), 3))
        
        # Color points for each matched object based on match quality
        for obj in self.object_buffer:
            if obj['matched'] and obj['point_indices'] is not None:
                # Validate point indices are still valid
                valid_indices = [idx for idx in obj['point_indices'] if idx < len(self.point_cloud_points)]
                if len(valid_indices) != len(obj['point_indices']):
                    self.get_logger().warn(
                        f"Object {obj['usd']} has {len(obj['point_indices']) - len(valid_indices)} "
                        f"invalid point indices"
                    )
                    obj['point_indices'] = valid_indices
                
                # Color based on density score (green) and distribution score (blue)
                density_score = obj.get('density_score', 0)
                distribution_score = obj.get('distribution_score', 0)
                colors[valid_indices] = [0, density_score, distribution_score]
        
        self._publish_point_cloud(self.point_cloud_points, colors)

    def _publish_point_cloud(self, points, colors):
        """Publish point cloud with colors"""
        rgb_uint8 = (colors * 255).astype(np.uint8)
        rgb_uint32 = (rgb_uint8[:,0].astype(np.uint32) << 16) | \
                    (rgb_uint8[:,1].astype(np.uint32) << 8) | \
                    rgb_uint8[:,2].astype(np.uint32)
        rgb_float = np.array([struct.unpack('f', struct.pack('I', val))[0] for val in rgb_uint32], dtype=np.float32)
        pc_color = np.hstack((points, rgb_float.reshape(-1,1)))
        pcd_msg = point_cloud2.create_cloud(header=Header(frame_id="odom"), fields=self.fields, points=pc_color)
        self.pub_pcd.publish(pcd_msg)

    def get_usd_vertices(self, usd_path, position, orientation):
        """
        Get vertices from USD file and transform them to world coordinates.
        Uses caching to avoid repeated USD file reads.
        
        Returns:
            tuple: (transformed_vertices, hull, delaunay, (world_center, world_extents))
        """
        if usd_path not in self.usd_cache:
            # Load and cache USD data
            cached_data = self._load_usd_data(usd_path)
            if cached_data is None:
                return None, None, None, (None, None)
            self.usd_cache[usd_path] = cached_data

        # Get cached data
        cached_data = self.usd_cache[usd_path]
        vertices = cached_data['vertices']
        local_center = cached_data['local_center']
        local_extents = cached_data['local_extents']

        # Transform vertices to world coordinates
        r = R.from_quat([orientation[1], orientation[2], orientation[3], orientation[0]])
        rot_matrix = r.as_matrix()
        
        # First rotate around origin, then translate
        transformed_verts = np.dot(rot_matrix, vertices.T).T
        transformed_verts = transformed_verts + position

        # Transform OBB to world coordinates
        world_center = np.dot(rot_matrix, local_center) + position
        # For extents, we only need to rotate (no translation)
        world_extents = np.abs(np.dot(rot_matrix, local_extents))

        # Create convex hull
        try:
            hull = ConvexHull(transformed_verts)
            delaunay = Delaunay(transformed_verts[hull.vertices])
            return transformed_verts, hull, delaunay, (world_center, world_extents)
        except Exception as e:
            self.get_logger().error(f"Error creating convex hull for {usd_path}: {e}")
            return None, None, None, (world_center, world_extents)

    def _load_usd_data(self, usd_path):
        """Load USD data and compute local bounds"""
        stage = Usd.Stage.Open(usd_path)
        if not stage:
            self.get_logger().error(f"Could not open USD stage at {usd_path}")
            return None

        meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
        vertices = []
        
        # Get all mesh vertices
        for prim in stage.TraverseAll():
            if prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(prim)
                points_attr = mesh.GetPointsAttr()
                if points_attr:
                    points = points_attr.Get()
                    if points:
                        xform = UsdGeom.Xformable(prim)
                        matrix = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                        for p in points:
                            transformed_p = matrix.Transform(p)
                            scaled_p = transformed_p * meters_per_unit
                            vertices.append(scaled_p)

        vertices = np.array(vertices)
        if len(vertices) == 0:
            self.get_logger().error(f"No vertices found in USD file: {usd_path}")
            return None

        # Calculate oriented bounding box in local coordinates
        min_bounds = np.min(vertices, axis=0)
        max_bounds = np.max(vertices, axis=0)
        local_center = (min_bounds + max_bounds) / 2.0
        local_extents = (max_bounds - min_bounds) / 2.0

        return {
            'vertices': vertices,
            'local_center': local_center,
            'local_extents': local_extents
        }

    def is_point_in_obb(self, point, center, extents):
        """
        Check if a point is inside an oriented bounding box.
        The OBB is defined by its center and extents (half-widths) in world coordinates.
        """
        # Check if point is within extents in each dimension
        return np.all(np.abs(point - center) <= extents)

    def find_points_in_hull(self, hull_vertices, delaunay, center, obb_params, points, point_tree, cluster_id):
        """
        Efficiently find points within a convex hull using OBB filtering.
        Returns points inside hull and their indices, excluding points used by other clusters.
        Points can be shared within the same cluster.
        """
        if hull_vertices is None or delaunay is None:
            return [], []

        world_center, world_extents = obb_params

        # First use KD-tree to find points within a rough radius (diagonal of OBB)
        # This is just to get a smaller set of points to check
        rough_radius = np.linalg.norm(world_extents) * 2
        indices = point_tree.query_ball_point(world_center, rough_radius)
        if not indices:
            return [], []

        # Convert to numpy array and get candidate points
        indices = np.array(indices)
        
        # Filter out points that have been used by other clusters
        used_by_other_clusters = set()
        for other_cluster_id, used_points in self.used_point_indices_by_cluster.items():
            if other_cluster_id != cluster_id:
                used_by_other_clusters.update(used_points)
        
        unused_indices = indices[~np.isin(indices, list(used_by_other_clusters))]
        if len(unused_indices) == 0:
            return [], []
            
        candidate_points = points[unused_indices]
        
        # First filter by OBB (faster than hull check)
        is_in_obb = np.array([self.is_point_in_obb(p, world_center, world_extents) for p in candidate_points])
        obb_indices = unused_indices[is_in_obb]
        obb_points = candidate_points[is_in_obb]
        
        if len(obb_points) == 0:
            return [], []

        # Then use Delaunay triangulation to check which points are inside hull
        is_inside = delaunay.find_simplex(obb_points) >= 0
        
        # Get points and indices that are inside
        inside_indices = obb_indices[is_inside]
        inside_points = obb_points[is_inside]

        return inside_points, inside_indices

    def process_cluster(self, cluster_id):
        """
        Process a cluster of objects to find which one best fits the point cloud.
        Uses improved scoring system that considers point density and distribution.
        
        Args:
            cluster_id: ID of the cluster to process
            
        Returns:
            tuple: (best_object_index, best_score) or (None, 0.0) if no valid match
        """
        if not self.point_cloud_tree or len(self.point_cloud_points) == 0:
            return None, 0.0

        cluster_objects = self.object_clusters[cluster_id]
        
        # Find the currently matched object in this cluster
        current_matched_idx = next(
            (obj_idx for obj_idx in cluster_objects if self.object_buffer[obj_idx]['matched']),
            None
        )

        # Determine which objects to process
        if current_matched_idx is None:
            best_score = 0.0
            best_obj_idx = None
            objects_to_process = cluster_objects
        else:
            # Only process new objects and compare against current matched object
            current_score = self.object_buffer[current_matched_idx]['occupancy_score']
            best_score = current_score
            best_obj_idx = current_matched_idx
            # Get only new objects in this cluster
            objects_to_process = [
                obj_idx for obj_idx in cluster_objects 
                if obj_idx >= self.last_processed_buffer_size
            ]
            
            if not objects_to_process:
                # Check if current matched object still meets threshold
                if current_score >= self.min_score_threshold:
                    return current_matched_idx, current_score
                else:
                    # Current match no longer meets threshold, clear it
                    self._clear_cluster_match(cluster_id, current_matched_idx)
                    return None, 0.0

        # Initialize or clear used points for this cluster
        self.used_point_indices_by_cluster[cluster_id] = set()

        # Process each object in the cluster
        for obj_idx in objects_to_process:
            score = self._evaluate_object_match(obj_idx, cluster_id)
            if score > best_score:
                best_score = score
                best_obj_idx = obj_idx

        # Check if best score meets threshold
        if best_score < self.min_score_threshold:
            self.get_logger().info(
                f"Rejecting cluster {cluster_id} as best score {best_score:.2f} "
                f"below threshold {self.min_score_threshold}"
            )
            if current_matched_idx is not None:
                self._clear_cluster_match(cluster_id, current_matched_idx)
            return None, 0.0

        # Update matched object if it changed
        if best_obj_idx != current_matched_idx:
            self._update_cluster_match(cluster_id, current_matched_idx, best_obj_idx, best_score)

        return best_obj_idx, best_score

    def _evaluate_object_match(self, obj_idx, cluster_id):
        """
        Evaluate how well an object matches the point cloud data.
        
        Args:
            obj_idx: Index of the object to evaluate
            cluster_id: ID of the cluster containing the object
            
        Returns:
            float: Combined score for the match, or 0.0 if invalid
        """
        obj = self.object_buffer[obj_idx]
        
        # Get vertices and create hull
        vertices, hull, delaunay, obb_params = self.get_usd_vertices(
            obj['usd'], 
            obj['position'], 
            obj['quatWXYZ']
        )
        
        if vertices is None or hull is None:
            return 0.0

        # Find points in hull
        points_in_hull, point_indices = self.find_points_in_hull(
            hull.points[hull.vertices],
            delaunay,
            obj['position'],
            obb_params,
            self.point_cloud_points,
            self.point_cloud_tree,
            cluster_id
        )

        if len(points_in_hull) < self.min_points_threshold:
            return 0.0

        # Calculate scores
        density_score = self._calculate_density_score(points_in_hull, hull.volume)
        distribution_score = self._calculate_distribution_score(points_in_hull, hull)
        
        # Combine scores
        score = (density_score * self.density_weight + 
                distribution_score * self.distribution_weight) * len(points_in_hull)
        
        # Update object data
        obj['points_in_hull'] = points_in_hull
        obj['point_indices'] = point_indices
        obj['occupancy_score'] = score
        obj['density_score'] = density_score
        obj['distribution_score'] = distribution_score
        
        return score

    def _calculate_density_score(self, points, volume):
        """Calculate how well the points fill the hull volume"""
        point_density = len(points) / max(volume, 1e-6)
        return min(point_density / self.expected_point_density, 1.0)

    def _calculate_distribution_score(self, points, hull):
        """Calculate how well the points are distributed in the hull"""
        if len(points) <= 1:
            return 0.0
            
        # Divide hull into voxels and check point distribution
        min_bounds = np.min(hull.points[hull.vertices], axis=0)
        max_bounds = np.max(hull.points[hull.vertices], axis=0)
        voxel_counts = np.zeros(np.ceil((max_bounds - min_bounds) / self.voxel_size).astype(int))
        
        for point in points:
            voxel_idx = np.floor((point - min_bounds) / self.voxel_size).astype(int)
            if np.all(voxel_idx >= 0) and np.all(voxel_idx < voxel_counts.shape):
                voxel_counts[tuple(voxel_idx)] += 1
        
        # Calculate distribution score (lower variance is better)
        non_empty_voxels = voxel_counts[voxel_counts > 0]
        if len(non_empty_voxels) > 0:
            return 1.0 - min(np.std(non_empty_voxels) / np.mean(non_empty_voxels), 1.0)
        return 0.0

    def _clear_cluster_match(self, cluster_id, obj_idx):
        """Clear a cluster's matched object and its points"""
        self.object_buffer[obj_idx]['matched'] = False
        self.object_buffer[obj_idx]['points_in_hull'] = None
        self.object_buffer[obj_idx]['point_indices'] = None
        self.occupied_points -= set(self.used_point_indices_by_cluster.get(cluster_id, set()))
        if cluster_id in self.used_point_indices_by_cluster:
            del self.used_point_indices_by_cluster[cluster_id]
        self._update_visualization()

    def _update_cluster_match(self, cluster_id, old_idx, new_idx, score):
        """Update a cluster's matched object"""
        # Clear old match if it exists
        if old_idx is not None:
            self._clear_cluster_match(cluster_id, old_idx)
        
        # Update points for the new best object
        obj = self.object_buffer[new_idx]
        self.used_point_indices_by_cluster[cluster_id].update(obj['point_indices'])
        self.occupied_points.update(obj['point_indices'])
        obj['matched'] = True
        
        # Update cluster center
        self.cluster_centers[cluster_id] = obj['position']
        
        self.get_logger().info(
            f"Updated cluster {cluster_id} center to position of matched object "
            f"{obj['usd']} (score: {score:.2f}, "
            f"density_score: {obj['density_score']:.2f}, "
            f"distribution_score: {obj['distribution_score']:.2f})"
        )
        
        self._update_visualization()

    def extract_labels_from_usd_path(self, usd_path):
        """
        Extract semantic labels from USD file path.
        
        Args:
            usd_path: Path to USD file
            
        Returns:
            tuple: (second_to_last_label, third_to_last_label) or (None, None) if extraction fails
        """
        try:
            parts = usd_path.split('/')
            if len(parts) >= 3:
                return parts[-2], parts[-3]
        except Exception as e:
            self.get_logger().warn(f"Failed to extract labels from path {usd_path}: {e}")
        return None, None

    def get_object_label(self, usd_path):
        """
        Get the semantic label for an object based on configuration.
        
        Args:
            usd_path: Path to USD file
            
        Returns:
            str: Label to use for matching, or None if no valid label found
        """
        second_label, third_label = self.extract_labels_from_usd_path(usd_path)
        
        if self.use_second_to_last_label and second_label:
            return second_label
        elif self.use_third_to_last_label and third_label:
            return third_label
        return None

    def check_for_new_objects(self):
        """Check if we have new objects to process and update matches"""
        if len(self.object_buffer) <= self.last_processed_buffer_size:
            return

        # Get new objects
        new_objects = self.object_buffer[self.last_processed_buffer_size:]
        
        # Process each cluster that has new objects
        clusters_to_process = set(obj['cluster_id'] for obj in new_objects)
        
        # Clear used points for clusters we're about to process
        self._clear_clusters_for_processing(clusters_to_process)
        
        # Process each cluster
        for cluster_id in clusters_to_process:
            best_obj_idx, best_score = self.process_cluster(cluster_id)
            if best_obj_idx is not None:
                current_matched = any(
                    self.object_buffer[obj_idx]['matched'] 
                    for obj_idx in self.object_clusters[cluster_id]
                )
                self.get_logger().info(
                    f"Cluster {cluster_id}: Best object {self.object_buffer[best_obj_idx]['usd']} "
                    f"with score {best_score:.2f} out of {len(self.object_clusters[cluster_id])} objects "
                    f"in the cluster (replaced existing match: {current_matched})"
                )
        
        self.last_processed_buffer_size = len(self.object_buffer)
        self._update_visualization()

    def _clear_clusters_for_processing(self, cluster_ids):
        """Clear points and matches for clusters that are about to be processed"""
        for cluster_id in cluster_ids:
            if cluster_id in self.used_point_indices_by_cluster:
                # Remove this cluster's points from occupied_points
                self.occupied_points -= self.used_point_indices_by_cluster[cluster_id]
                # Clear points from all objects in this cluster
                for obj_idx in self.object_clusters[cluster_id]:
                    self.object_buffer[obj_idx]['points_in_hull'] = None
                    self.object_buffer[obj_idx]['point_indices'] = None
                    self.object_buffer[obj_idx]['matched'] = False
                del self.used_point_indices_by_cluster[cluster_id]
        
        # Validate that all points in occupied_points still exist in point_cloud_points
        if len(self.occupied_points) > 0:
            valid_indices = set(range(len(self.point_cloud_points)))
            self.occupied_points = self.occupied_points.intersection(valid_indices)

    def publish_matched_objects(self):
        """Publish currently matched objects as a ROS message"""
        msg = UsdBufferPoseMsg()
        for obj in self.object_buffer:
            if obj.get('matched'):
                obj_msg = UsdStringIdPoseMsg()
                obj_msg.data_path = obj['usd']
                obj_msg.id = int(obj['cluster_id']) if obj.get('cluster_id') is not None else 0
                obj_msg.pose.position.x = float(obj['position'][0])
                obj_msg.pose.position.y = float(obj['position'][1])
                obj_msg.pose.position.z = float(obj['position'][2])
                obj_msg.pose.orientation.w = float(obj['quatWXYZ'][0])
                obj_msg.pose.orientation.x = float(obj['quatWXYZ'][1])
                obj_msg.pose.orientation.y = float(obj['quatWXYZ'][2])
                obj_msg.pose.orientation.z = float(obj['quatWXYZ'][3])
                msg.objects.append(obj_msg)
        self.pub_usd_pose.publish(msg)

    def save_matched_buffer(self):
        """Save the current matched buffer to a JSON file"""
        # Get current timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"matched_buffer_{timestamp}.json"
        filepath = os.path.join(self.save_dir, filename)

        # Create buffer data structure
        buffer_data = {
            "timestamp": timestamp,
            "objects": [
                {
                    "usd_path": obj['usd'],
                    "cluster_id": int(obj['cluster_id']) if obj.get('cluster_id') is not None else 0,
                    "position": obj['position'].tolist(),
                    "quatWXYZ": obj['quatWXYZ'].tolist(),
                    "label": obj.get('label'),
                    "occupancy_score": float(obj.get('occupancy_score', 0.0))
                }
                for obj in self.object_buffer
                if obj.get('matched')
            ]
        }

        # Save to file
        try:
            with open(filepath, 'w') as f:
                json.dump(buffer_data, f, indent=2)
            self.get_logger().info(f"Saved matched buffer to {filepath}")
            
            # Keep only the last 5 buffer files
            self._cleanup_old_buffer_files()
        except Exception as e:
            self.get_logger().error(f"Failed to save matched buffer: {e}")

    def _cleanup_old_buffer_files(self):
        """Remove old buffer files, keeping only the 5 most recent ones"""
        buffer_files = sorted([f for f in os.listdir(self.save_dir) if f.startswith("matched_buffer_")])
        if len(buffer_files) > 5:
            for old_file in buffer_files[:-5]:
                try:
                    os.remove(os.path.join(self.save_dir, old_file))
                    self.get_logger().info(f"Removed old buffer file: {old_file}")
                except Exception as e:
                    self.get_logger().warn(f"Failed to remove old buffer file {old_file}: {e}")

def main():
    """Main entry point for the USD buffer node"""
    rclpy.init()
    usd_buffer_node = USDBufferNode()
    rclpy.spin(usd_buffer_node)
    usd_buffer_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
