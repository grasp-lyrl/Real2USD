import traceback
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from sklearn.cluster import DBSCAN

from custom_message.msg import PipelineStepTiming, UsdStringIdPoseMsg, UsdStringIdSrcTargMsg
from geometry_msgs.msg import Pose
from std_msgs.msg import Header


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
        self.pub_debug_src = self.create_publisher(PointCloud2, "/debug/registration/source_pc", 10)
        self.pub_debug_targ = self.create_publisher(PointCloud2, "/debug/registration/target_pc", 10)
        self.pub_overlay = self.create_publisher(PointCloud2, "/registration/overlay", 10)
        self.pub_timing = self.create_publisher(PipelineStepTiming, "/pipeline/timings", 10)
        self._timing_sequence = 0

        self.declare_parameter("yaw_only_registration", True)
        self.yaw_only_registration = self.get_parameter("yaw_only_registration").value
        if isinstance(self.yaw_only_registration, str):
            self.yaw_only_registration = self.yaw_only_registration.lower() == "true"
        self.declare_parameter("registration_min_fitness", 0.10)
        self.declare_parameter("registration_icp_distance_threshold_m", 0.03)
        self.declare_parameter("registration_icp_max_iteration", 50)
        self.declare_parameter("registration_min_target_points", 30)
        self.declare_parameter("registration_max_translation_delta_m", 2.5)
        self.declare_parameter("registration_metrics_path", "")
        self.registration_min_fitness = float(self.get_parameter("registration_min_fitness").value)
        self.registration_icp_distance_threshold_m = float(self.get_parameter("registration_icp_distance_threshold_m").value)
        self.registration_icp_max_iteration = int(self.get_parameter("registration_icp_max_iteration").value)
        self.registration_min_target_points = int(self.get_parameter("registration_min_target_points").value)
        self.registration_max_translation_delta_m = float(self.get_parameter("registration_max_translation_delta_m").value)
        metrics_path = str(self.get_parameter("registration_metrics_path").value or "").strip()
        self.registration_metrics_path = Path(metrics_path) if metrics_path else None
        if self.registration_metrics_path is not None:
            self.registration_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self._last_reg_debug = {}

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

    def _log_registration_event(self, *, job_id: str, track_id: int, data_path: str, success: bool, duration_ms: float):
        if self.registration_metrics_path is None:
            return
        payload = {
            "job_id": job_id,
            "track_id": int(track_id),
            "data_path": data_path,
            "success": bool(success),
            "duration_ms": float(duration_ms),
            "debug": self._last_reg_debug,
        }
        try:
            with open(self.registration_metrics_path, "a") as f:
                f.write(json.dumps(payload) + "\n")
        except Exception:
            pass

    def _quat_to_yaw_only(self, quat_xyzw: np.ndarray) -> np.ndarray:
        """Project orientation to yaw-only (rotation about Z). quat_xyzw is (x,y,z,w). Returns (x,y,z,w)."""
        q = np.asarray(quat_xyzw, dtype=np.float64).ravel()
        if q.shape[0] < 4:
            return quat_xyzw
        R_full = R.from_quat([q[0], q[1], q[2], q[3]]).as_matrix()
        yaw = np.arctan2(R_full[1, 0], R_full[0, 0])
        R_z = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1],
        ], dtype=np.float64)
        return np.asarray(R.from_matrix(R_z).as_quat(), dtype=np.float64)

    def callback_src_targ(self, msg):
        try:
            # Writable copy: ROS PointCloud2 buffer is read-only; Open3D may write to arrays.
            # Vectorized: structured array from list, then column_stack + copy (no Python loop over points).
            pts_src = point_cloud2.read_points_list(
                msg.src_pc, field_names=("x", "y", "z"), skip_nans=True
            )
            pts_targ = point_cloud2.read_points_list(
                msg.targ_pc, field_names=("x", "y", "z"), skip_nans=True
            )
            _xyz_dtype = [("x", np.float64), ("y", np.float64), ("z", np.float64)]
            arr_src = np.array(pts_src, dtype=_xyz_dtype)
            arr_targ = np.array(pts_targ, dtype=_xyz_dtype)
            points_src = np.column_stack((arr_src["x"], arr_src["y"], arr_src["z"])).copy()
            points_targ = np.column_stack((arr_targ["x"], arr_targ["y"], arr_targ["z"])).copy()
            # Force writable C-contiguous (ROS/Open3D can hand back read-only views)
            points_src = np.require(points_src, dtype=np.float64, requirements=["C_CONTIGUOUS", "OWNDATA", "W"])
            points_targ = np.require(points_targ, dtype=np.float64, requirements=["C_CONTIGUOUS", "OWNDATA", "W"])
        except Exception as e:
            self.get_logger().warn(f"Failed to read point clouds: {e}")
            self.get_logger().warn(f"Traceback:\n{traceback.format_exc()}")
            return
        usd_url = msg.data_path
        trackId = msg.id

        if len(points_src) == 0 or len(points_targ) == 0:
            self.get_logger().warn(
                f"Registration skipped id={trackId} job_id={getattr(msg, 'job_id', '')}: empty clouds (src={len(points_src)}, targ={len(points_targ)}). No pose published."
            )
            # When we have initial_pose, publish it anyway so the slot appears in scene_graph.json (object at SAM3D pose when ICP target is empty).
            initial_pose = getattr(msg, "initial_pose", None)
            if initial_pose is not None and len(points_src) > 0:
                p = initial_pose.position
                q = initial_pose.orientation
                t = np.array([p.x, p.y, p.z], dtype=np.float64)
                quatXYZW = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
                if self.yaw_only_registration:
                    quatXYZW = self._quat_to_yaw_only(quatXYZW)
                pose_msg = UsdStringIdPoseMsg()
                pose_msg.header = Header(frame_id="odom")
                pose_msg.data_path = usd_url
                pose_msg.id = trackId
                pose_msg.job_id = getattr(msg, "job_id", "") or ""
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
                self.get_logger().info(
                    f"Published initial pose for id={trackId} (empty target); slot will appear in scene_graph.json"
                )
            return
        try:
            self.pub_debug_src.publish(
                point_cloud2.create_cloud_xyz32(header=Header(frame_id="odom"), points=points_src.astype(np.float32))
            )
            self.pub_debug_targ.publish(
                point_cloud2.create_cloud_xyz32(header=Header(frame_id="odom"), points=points_targ.astype(np.float32))
            )
        except Exception:
            pass
        t_start = self.get_clock().now()
        # Use initial pose from SAM3D+go2 when provided (global target; ICP from this init)
        initial_pose = getattr(msg, "initial_pose", None)
        use_initial = (
            initial_pose is not None
            and (abs(initial_pose.position.x) > 1e-9 or abs(initial_pose.position.y) > 1e-9 or abs(initial_pose.position.z) > 1e-9)
        )
        try:
            t, quatXYZW, points_transformed = self.get_pose(
                points_src, points_targ, initial_pose=initial_pose if use_initial else None
            )
        except KeyboardInterrupt:
            raise
        except Exception as e:
            self.get_logger().warn(f"Registration failed for {usd_url}: {e}")
            tb = traceback.format_exc()
            self.get_logger().warn(f"Traceback:\n{tb}")
            return
        t_elapsed = (self.get_clock().now() - t_start).nanoseconds * 1e-6
        timing_msg = PipelineStepTiming()
        timing_msg.header.stamp = self.get_clock().now().to_msg()
        timing_msg.header.frame_id = "odom"
        timing_msg.node_name = "registration_node"
        timing_msg.step_name = "get_pose"
        timing_msg.duration_ms = t_elapsed
        timing_msg.sequence_id = self._timing_sequence
        self._timing_sequence += 1
        self.pub_timing.publish(timing_msg)
        if points_transformed is not None:
            try:
                self.pub_odom_pc_seg.publish(
                    point_cloud2.create_cloud_xyz32(
                        header=Header(frame_id="odom"),
                        points=np.asarray(points_transformed, dtype=np.float32).copy(),
                    )
                )
                self._publish_overlay(points_transformed, points_targ)
            except Exception:
                pass
        if t is not None and quatXYZW is not None:
            pose_msg = UsdStringIdPoseMsg()
            pose_msg.header = Header(frame_id="odom")
            pose_msg.data_path = usd_url
            pose_msg.id = trackId
            pose_msg.job_id = getattr(msg, "job_id", "") or ""
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
            self._log_registration_event(
                job_id=getattr(msg, "job_id", "") or "",
                track_id=trackId,
                data_path=usd_url,
                success=True,
                duration_ms=t_elapsed,
            )
        else:
            self._log_registration_event(
                job_id=getattr(msg, "job_id", "") or "",
                track_id=trackId,
                data_path=usd_url,
                success=False,
                duration_ms=t_elapsed,
            )

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

    def _publish_overlay(self, src_points_transformed, targ_points):
        """Publish combined point cloud: registered object (red) + target (cyan) in odom frame for visualization."""
        if src_points_transformed is None or len(src_points_transformed) == 0:
            return
        src_colors = np.tile([1.0, 0.0, 0.0], (len(src_points_transformed), 1))
        targ_colors = np.tile([0.0, 1.0, 1.0], (len(targ_points), 1))
        all_points = np.vstack((np.asarray(src_points_transformed, dtype=np.float64), np.asarray(targ_points, dtype=np.float64)))
        all_colors = np.vstack((src_colors, targ_colors))
        colors_uint32 = np.zeros(len(all_points), dtype=np.uint32)
        for i in range(len(all_points)):
            r, g, b = all_colors[i]
            colors_uint32[i] = (int(r * 255) << 16) | (int(g * 255) << 8) | int(b * 255)
        points_with_colors = np.zeros(len(all_points), dtype=[
            ("x", np.float32), ("y", np.float32), ("z", np.float32), ("rgb", np.float32),
        ])
        points_with_colors["x"] = all_points[:, 0].astype(np.float32)
        points_with_colors["y"] = all_points[:, 1].astype(np.float32)
        points_with_colors["z"] = all_points[:, 2].astype(np.float32)
        points_with_colors["rgb"] = colors_uint32.view(np.float32)
        try:
            msg = point_cloud2.create_cloud(
                header=Header(frame_id="odom"),
                fields=self.fields,
                points=points_with_colors,
            )
            self.pub_overlay.publish(msg)
        except Exception:
            pass

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

    def _get_pose_from_initial(self, points_src, points_targ, initial_pose):
        """Register source to full target using initial pose (from SAM3D+go2) as trans_init. No clustering."""
        p = initial_pose.position
        q = initial_pose.orientation
        trans_init = np.eye(4)
        trans_init[:3, :3] = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        trans_init[:3, 3] = [p.x, p.y, p.z]

        src = o3d.geometry.PointCloud()
        src.points = o3d.utility.Vector3dVector(np.asarray(points_src, dtype=np.float64).copy())
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(np.asarray(points_targ, dtype=np.float64).copy())

        voxel_size = 0.01
        skip_src = len(points_src) < 30
        src_down, _ = self.preprocess_point_cloud(src, voxel_size=voxel_size, skip_downsample=skip_src)
        target_down, _ = self.preprocess_point_cloud(target, voxel_size=voxel_size)
        n_src = len(np.asarray(src_down.points))
        n_targ = len(np.asarray(target_down.points))
        if n_src < self.registration_min_target_points or n_targ < self.registration_min_target_points:
            self._last_reg_debug = {
                "mode": "initial_pose",
                "reason": "too_few_points_after_preprocess",
                "n_src": int(n_src),
                "n_targ": int(n_targ),
            }
            self.get_logger().warn("Initial-pose registration: too few points after preprocessing; returning initial pose")
            quat = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
            if self.yaw_only_registration:
                quat = self._quat_to_yaw_only(quat)
            return np.array([p.x, p.y, p.z]), quat, None

        distance_threshold = self.registration_icp_distance_threshold_m
        icp_result = o3d.pipelines.registration.registration_icp(
            src_down, target_down, distance_threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self.registration_icp_max_iteration),
        )
        # Robustness: if fitness low or translation jumps too far, fall back to initial pose.
        if icp_result.fitness < self.registration_min_fitness:
            self._last_reg_debug = {
                "mode": "initial_pose",
                "reason": "low_fitness_fallback_to_initial",
                "fitness": float(icp_result.fitness),
                "min_fitness": float(self.registration_min_fitness),
            }
            self.get_logger().warn(
                f"Initial-pose registration: low fitness {icp_result.fitness:.3f} < {self.registration_min_fitness:.3f}; using initial pose as result"
            )
            quat = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
            if self.yaw_only_registration:
                quat = self._quat_to_yaw_only(quat)
            return np.array([p.x, p.y, p.z]), quat, None
        T = np.asarray(icp_result.transformation, dtype=np.float64).copy()
        translation = T[0:3, 3].copy()
        init_translation = np.array([p.x, p.y, p.z], dtype=np.float64)
        translation_delta = float(np.linalg.norm(translation - init_translation))
        if translation_delta > self.registration_max_translation_delta_m:
            self._last_reg_debug = {
                "mode": "initial_pose",
                "reason": "translation_jump_fallback_to_initial",
                "fitness": float(icp_result.fitness),
                "translation_delta_m": float(translation_delta),
                "max_translation_delta_m": float(self.registration_max_translation_delta_m),
            }
            quat = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
            if self.yaw_only_registration:
                quat = self._quat_to_yaw_only(quat)
            return init_translation, quat, None
        R_full = T[0:3, 0:3]
        # Objects are z-up; constrain to yaw-only (rotate about Z) so we don't tip the object
        if self.yaw_only_registration:
            yaw = np.arctan2(R_full[1, 0], R_full[0, 0])
            R_z = np.array([
                [np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1],
            ], dtype=np.float64)
            quat = R.from_matrix(R_z).as_quat()
            T_final = np.eye(4, dtype=np.float64)
            T_final[:3, :3] = R_z
            T_final[:3, 3] = translation
        else:
            quat = R.from_matrix(R_full.copy()).as_quat()
            T_final = T
        src_copy = o3d.geometry.PointCloud()
        src_copy.points = src_down.points
        src_copy.transform(T_final)
        points_transformed = np.asarray(src_copy.points, dtype=np.float64).copy()
        self.get_logger().info(
            f"Initial-pose registration: fitness={icp_result.fitness:.3f}, translation={translation}, yaw_only={self.yaw_only_registration}"
        )
        self._last_reg_debug = {
            "mode": "initial_pose",
            "reason": "ok",
            "fitness": float(icp_result.fitness),
            "inlier_rmse": float(icp_result.inlier_rmse),
            "translation_delta_m": float(translation_delta),
            "n_src": int(n_src),
            "n_targ": int(n_targ),
        }
        return translation, quat, points_transformed

    def get_pose(self, points_src, points_targ, render=False, initial_pose=None):
        # Single registration from initial pose (SAM3D+go2) to full target (e.g. global PC)
        if initial_pose is not None:
            return self._get_pose_from_initial(points_src, points_targ, initial_pose)

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
                
            # Get points for this cluster (copy so Open3D gets writable buffer)
            cluster_mask = cluster_labels == cluster_id
            cluster_points = np.asarray(points_targ[cluster_mask], dtype=np.float64).copy()

            if len(cluster_points) < self.registration_min_target_points:
                # Only warn if skipping due to too few points
                # self.get_logger().warn(f"Skipping cluster {cluster_id} - too few points: {len(cluster_points)}")
                continue
                
            # Create point clouds (ensure writable arrays for Open3D)
            src = o3d.geometry.PointCloud()
            src.points = o3d.utility.Vector3dVector(np.asarray(points_src, dtype=np.float64).copy())
            target = o3d.geometry.PointCloud()
            target.points = o3d.utility.Vector3dVector(cluster_points)
            
            # Initial transformation using cluster centroid
            target_center = np.mean(cluster_points, axis=0)
            
            voxel_size = 0.01  # Fixed voxel size for both source and target
            # Preprocess source: skip downsampling if already small
            skip_downsample_src = len(points_src) < 30
            src_down_orig, source_fpfh_orig = self.preprocess_point_cloud(src, voxel_size=voxel_size, skip_downsample=skip_downsample_src)
            target_down, target_fpfh = self.preprocess_point_cloud(target, voxel_size=voxel_size)

            # Skip registration if source or target too small after preprocessing (avoids FGR/ICP errors)
            n_src_down = len(np.asarray(src_down_orig.points))
            n_targ_down = len(np.asarray(target_down.points))
            if n_src_down < self.registration_min_target_points or n_targ_down < self.registration_min_target_points:
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
                src_down.points = o3d.utility.Vector3dVector(
                    np.dot(np.asarray(src_down_orig.points, dtype=np.float64).copy(), rot_matrix.T)
                )
                source_fpfh = source_fpfh_orig  # FPFH is rotation invariant

                # Initial transformation (translation only)
                trans_init = np.eye(4)
                trans_init[:3, 3] = target_center
                trans_init[:3, :3] = np.eye(3)

                # Skip FGR when too few points (avoids Open3D "low must be < high" in tuple sampling)
                n_src_down = len(np.asarray(src_down.points))
                n_targ_down = len(np.asarray(target_down.points))
                maximum_tuple_count = min(500, n_src_down // 2, n_targ_down // 2)
                # FGR requires maximum_tuple_count >= 1; Open3D fails with low=0, high=0 otherwise
                if maximum_tuple_count < 1 or n_src_down < self.registration_min_target_points or n_targ_down < self.registration_min_target_points:
                    distance_threshold = max(0.005, self.registration_icp_distance_threshold_m * 0.66)
                else:
                    try:
                        result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
                            src_down, target_down, source_fpfh, target_fpfh,
                            o3d.pipelines.registration.FastGlobalRegistrationOption(
                                maximum_correspondence_distance=voxel_size * 0.5,
                                iteration_number=64,
                                tuple_scale=0.95,
                                maximum_tuple_count=maximum_tuple_count))
                        trans_init = result.transformation
                        distance_threshold = self.registration_icp_distance_threshold_m
                    except Exception as e:
                        self.get_logger().warn(f"FGR failed for cluster {cluster_id}: {str(e)}")
                        distance_threshold = max(0.005, self.registration_icp_distance_threshold_m * 0.66)

                # Perform ICP with fixed parameters
                icp_result = o3d.pipelines.registration.registration_icp(
                    src_down, target_down, distance_threshold, trans_init,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self.registration_icp_max_iteration))

                # Only log best match summary
                # Check if this is the best result so far
                if icp_result.fitness > best_fitness:
                    best_fitness = icp_result.fitness
                    rotation_icp = np.asarray(icp_result.transformation[0:3, 0:3], dtype=np.float64).copy()
                    translation = np.asarray(icp_result.transformation[0:3, 3], dtype=np.float64).copy()
                    # ICP result is T_odom_from_src_rotated; full T_odom_from_src_orig = T_icp @ diag(R_yaw,1)
                    if self.yaw_only_registration:
                        yaw_final = np.arctan2(rotation_icp[1, 0], rotation_icp[0, 0]) + yaw
                        matrix = np.array([
                            [np.cos(yaw_final), -np.sin(yaw_final), 0],
                            [np.sin(yaw_final), np.cos(yaw_final), 0],
                            [0, 0, 1]
                        ], dtype=np.float64)
                    else:
                        matrix = (rotation_icp @ rot_matrix).astype(np.float64)
                    quaternion = R.from_matrix(matrix).as_quat()

                    # Create transformation matrix (T_odom_from_src_orig) for publishing and viz
                    transformation = np.eye(4, dtype=np.float64)
                    transformation[0:3, 3] = translation
                    transformation[0:3, 0:3] = matrix

                    # Viz: transform original source (src_down_orig) so overlay matches published pose
                    src_copy = o3d.geometry.PointCloud()
                    src_copy.points = src_down_orig.points
                    src_copy.transform(transformation)
                    best_points_transformed = np.asarray(src_copy.points, dtype=np.float64).copy()
                    best_translation = translation
                    best_quaternion = quaternion
                    best_transformation = transformation
                    best_cluster_id = cluster_id

                    yaw_deg = np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))
                    self.get_logger().info(
                        f"New best match: Cluster {cluster_id}, yaw_init: {np.degrees(yaw):.1f} deg, fitness: {icp_result.fitness:.3f}, translation: {translation}, yaw: {yaw_deg:.1f} deg, yaw_only={self.yaw_only_registration}"
                    )
        
        if best_points_transformed is not None:
            # Visualize clusters again, highlighting the best cluster in red
            self.visualize_best_cluster(points_targ, cluster_labels, best_cluster_id)
            self.visualize_best_match(best_points_transformed, points_targ)
            self.get_logger().info(f"Final best fitness: {best_fitness:.3f}")
            if best_fitness < self.registration_min_fitness:
                self._last_reg_debug = {
                    "mode": "cluster_search",
                    "reason": "best_fitness_below_threshold",
                    "best_fitness": float(best_fitness),
                    "min_fitness": float(self.registration_min_fitness),
                    "best_cluster_id": int(best_cluster_id) if best_cluster_id is not None else None,
                }
                return None, None, None
            self._last_reg_debug = {
                "mode": "cluster_search",
                "reason": "ok",
                "best_fitness": float(best_fitness),
                "best_cluster_id": int(best_cluster_id) if best_cluster_id is not None else None,
                "num_clusters": int(len([c for c in unique_clusters if c != -1])),
            }
        else:
            # self.get_logger().warn("No valid registration found!")
            self._last_reg_debug = {
                "mode": "cluster_search",
                "reason": "no_valid_registration",
                "num_clusters": int(len([c for c in unique_clusters if c != -1])),
            }
        
        return best_translation, best_quaternion, best_points_transformed


def main():
    rclpy.init()
    registration_node = RegistrationNode()
    rclpy.spin(registration_node)
    registration_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
