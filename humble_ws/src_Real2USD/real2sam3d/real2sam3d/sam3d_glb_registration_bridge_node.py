"""
Bridge node: when a SAM3D object is injected with pose-at-origin, load the GLB (or PLY)
vertices as the source point cloud and a target point cloud, then publish UsdStringIdSrcTargMsg
so registration_node can run ICP and publish the real pose.

Target point cloud (same as real2usd retrieval path for good registration):
  - Prefer segment point cloud from the job dir (depth.npy + mask.png + meta.json in
    output/<job_id>/), built by unprojecting masked depth with camera/odom. This matches
    how retrieval_node produces the target in real2usd (object-region only).
  - Fallback: latest world point cloud from /point_cloud2 or /global_lidar_points.

Subscribes to:
  - /usd/Sam3dObjectForSlot: best object for slot (from sam3d_retrieval_node; job_id + track_id + data_path).
  - /point_cloud2 or /global_lidar_points: world point cloud (parameter), used if no job segment PC.

Publishes:
  - /usd/StringIdSrcTarg: src_pc = points from GLB/PLY, targ_pc = segment PC from job or world PC.
"""

import json
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

from custom_message.msg import PipelineStepTiming, Sam3dObjectForSlotMsg, UsdStringIdSrcTargMsg
from geometry_msgs.msg import Pose

# Same as retrieval_node / lidar_cam_node (Unitree Go2 front camera in odom body)
T_CAM_IN_ODOM = np.array([0.285, 0.0, 0.01], dtype=np.float64)
GROUND_PLANE_HEIGHT_THRESHOLD = 0.1


def _segment_point_cloud_from_job_dir_with_reason(
    job_dir: Path,
) -> Tuple[Optional[np.ndarray], str]:
    """
    Build segment point cloud in odom frame from job files (depth.npy, mask.png, meta.json).

    Uses meta.odometry as-is (no init_odom subtraction). So segment is in the same frame as
    the robot odom at capture time. When injector runs with use_init_odom=false, pose.json
    initial_position is also in that frame; registration output will then be in full odom.
    Returns (points, reason). reason is empty if points is not None; otherwise explains why segment is unavailable.
    """
    depth_path = job_dir / "depth.npy"
    mask_path = job_dir / "mask.png"
    meta_path = job_dir / "meta.json"
    missing = []
    if not depth_path.exists():
        missing.append("depth.npy")
    if not mask_path.exists():
        missing.append("mask.png")
    if not meta_path.exists():
        missing.append("meta.json")
    if missing:
        return None, f"missing in {job_dir.name}: {', '.join(missing)}"

    try:
        depth_crop = np.load(str(depth_path)).astype(np.float64)
        mask_crop = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_crop is None or depth_crop.size == 0:
            return None, "depth/mask empty or unreadable"
        with open(meta_path) as f:
            meta = json.load(f)
    except Exception as e:
        return None, f"read error: {e}"

    crop_bbox = meta.get("crop_bbox")
    cam = meta.get("camera_info") or {}
    odom = meta.get("odometry") or {}
    if not (crop_bbox and len(crop_bbox) >= 4):
        return None, "meta missing or invalid crop_bbox"
    # Job writer writes camera_info.K (uppercase); accept k or K
    k_flat = cam.get("k") or cam.get("K")
    if k_flat is None or len(k_flat) != 9:
        return None, "meta missing camera_info.k or camera_info.K (9 values)"
    if odom.get("position") is None:
        return None, "meta missing odometry.position"

    x_min, y_min, x_max, y_max = int(crop_bbox[0]), int(crop_bbox[1]), int(crop_bbox[2]), int(crop_bbox[3])
    K = np.array(k_flat, dtype=np.float64).reshape(3, 3)
    if mask_crop.shape != depth_crop.shape:
        mask_crop = cv2.resize(mask_crop, (depth_crop.shape[1], depth_crop.shape[0]), interpolation=cv2.INTER_NEAREST)
    vc, uc = np.where((mask_crop > 0) & (depth_crop > 1e-6))
    if len(vc) == 0:
        return None, "mask has no valid depth pixels"
    u_full = (x_min + uc).astype(np.int32)
    v_full = (y_min + vc).astype(np.int32)
    Z = depth_crop[vc, uc]

    uv1 = np.stack([u_full, v_full, np.ones_like(u_full, dtype=np.float64)], axis=0)
    xyz_cam = (np.linalg.inv(K) @ uv1) * Z
    points_cam = xyz_cam.T

    R_static_odom_to_cam = R.from_euler("xyz", np.radians([-90, 0, 90]), degrees=False).as_matrix()
    R_additional = R.from_euler("xyz", np.radians([0, 0, 180]), degrees=False).as_matrix()
    R_odom_to_cam = R_additional @ R_static_odom_to_cam
    T_odom_from_cam = np.eye(4)
    T_odom_from_cam[:3, :3] = R_odom_to_cam
    T_odom_from_cam[:3, 3] = T_CAM_IN_ODOM

    t = np.array(odom["position"], dtype=np.float64)
    q = np.array(odom["orientation"], dtype=np.float64)
    R_world_from_odom = R.from_quat(q).as_matrix()
    T_world_from_odom = np.eye(4)
    T_world_from_odom[:3, :3] = R_world_from_odom
    T_world_from_odom[:3, 3] = t
    T_world_from_cam = T_world_from_odom @ T_odom_from_cam

    ones = np.ones((points_cam.shape[0], 1), dtype=np.float64)
    h = np.hstack([points_cam, ones]).T
    points_odom = (T_world_from_cam @ h).T[:, :3]
    points_odom = points_odom[points_odom[:, 2] > 0]
    if len(points_odom) < 30:
        return None, "too few points after z>0 filter"
    points_odom = points_odom[points_odom[:, 2] > GROUND_PLANE_HEIGHT_THRESHOLD]
    if len(points_odom) < 30:
        return None, "too few points after ground filter"
    return points_odom.astype(np.float64), ""


def _segment_point_cloud_from_job_dir(job_dir: Path) -> Optional[np.ndarray]:
    """Wrapper: returns points or None (for callers that do not need reason)."""
    points, _ = _segment_point_cloud_from_job_dir_with_reason(job_dir)
    return points


def _load_points_from_glb(glb_path: str, max_points: Optional[int] = 50000) -> Optional[np.ndarray]:
    """Load vertex positions from a GLB file. Flattens scene graph so frame matches simple_scene_buffer_node."""
    try:
        import trimesh
    except ImportError:
        return None
    try:
        scene = trimesh.load(glb_path, process=False)
        if isinstance(scene, trimesh.Scene):
            verts_list = []
            for node_name in scene.graph.nodes_geometry:
                try:
                    node_tf, geom_name = scene.graph[node_name]
                    geom = scene.geometry.get(geom_name)
                    if isinstance(geom, trimesh.Trimesh) and len(geom.vertices) > 0:
                        v = np.asarray(geom.vertices, dtype=np.float64)
                        ones = np.ones((v.shape[0], 1), dtype=np.float64)
                        v_h = (np.asarray(node_tf, dtype=np.float64) @ np.hstack([v, ones]).T).T[:, :3]
                        verts_list.append(v_h)
                except Exception:
                    continue
            verts = np.vstack(verts_list) if verts_list else np.empty((0, 3), dtype=np.float64)
        elif isinstance(scene, trimesh.Trimesh):
            verts = np.asarray(scene.vertices, dtype=np.float64)
        else:
            return None
        if len(verts) == 0:
            return None
        if max_points is not None and len(verts) > max_points:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(verts), size=max_points, replace=False)
            verts = verts[idx]
        return verts
    except Exception:
        return None


def _load_points_from_ply(ply_path: str, max_points: Optional[int] = 50000) -> Optional[np.ndarray]:
    """Load x,y,z from a PLY file. Returns (N, 3) float64 or None."""
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(ply_path)
        if pcd.has_points() and len(pcd.points) > 0:
            pts = np.asarray(pcd.points, dtype=np.float64)
            if max_points is not None and len(pts) > max_points:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(pts), size=max_points, replace=False)
                pts = pts[idx]
            return pts
    except Exception:
        pass
    try:
        from plyfile import PlyData
        ply = PlyData.read(ply_path)
        for el in ply.elements:
            if not hasattr(el.data, "dtype") or el.data.dtype.names is None:
                continue
            names = set(el.data.dtype.names)
            if "x" not in names or "y" not in names or "z" not in names:
                continue
            x = np.asarray(el.data["x"], dtype=np.float64)
            y = np.asarray(el.data["y"], dtype=np.float64)
            z = np.asarray(el.data["z"], dtype=np.float64)
            pts = np.stack([x, y, z], axis=1)
            valid = np.isfinite(pts).all(axis=1)
            if not np.any(valid):
                return None
            pts = pts[valid]
            if max_points is not None and len(pts) > max_points:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(pts), size=max_points, replace=False)
                pts = pts[idx]
            return pts
    except Exception:
        pass
    return None


def load_object_points(data_path: str, max_points: Optional[int] = 50000) -> Optional[np.ndarray]:
    """
    Load point cloud from object path (GLB or PLY). Prefer GLB when both exist.
    data_path may be .../object.ply or .../object.glb. Returns (N, 3) in object-local Z-up or None.
    """
    path = Path(data_path)
    if not path.exists():
        return None
    if path.suffix.lower() == ".glb":
        return _load_points_from_glb(str(path), max_points=max_points)
    if path.suffix.lower() == ".ply":
        # Prefer same-dir object.glb if present (better geometry)
        glb_path = path.parent / "object.glb"
        if glb_path.exists():
            pts = _load_points_from_glb(str(glb_path), max_points=max_points)
            if pts is not None:
                return pts
        return _load_points_from_ply(str(path), max_points=max_points)
    return None


class Sam3dGlbRegistrationBridgeNode(Node):
    def __init__(self):
        super().__init__("sam3d_glb_registration_bridge_node")

        self.declare_parameter("queue_dir", "/data/sam3d_queue")
        self.declare_parameter("world_point_cloud_topic", "/global_lidar_points")
        self.declare_parameter("max_src_points", 50000)
        self.declare_parameter("debounce_sec", 30.0)
        # "global" = accumulated world point cloud + initial pose from SAM3D+go2 (default); "segment" = target from job dir
        self.declare_parameter("registration_target", "global")
        self.declare_parameter("global_target_radius_m", 2.5)
        self.declare_parameter("global_target_max_points", 30000)
        self.declare_parameter("global_target_min_points", 200)

        self.queue_dir = Path(self.get_parameter("queue_dir").value)
        self.world_pc_topic = self.get_parameter("world_point_cloud_topic").value
        self.max_src_points = self.get_parameter("max_src_points").value
        self.debounce_sec = self.get_parameter("debounce_sec").value
        self.registration_target = (self.get_parameter("registration_target").value or "global").strip().lower()
        self.global_target_radius_m = float(self.get_parameter("global_target_radius_m").value)
        self.global_target_max_points = int(self.get_parameter("global_target_max_points").value)
        self.global_target_min_points = int(self.get_parameter("global_target_min_points").value)
        if self.registration_target not in ("segment", "global"):
            self.registration_target = "global"

        self._latest_world_pc: Optional[PointCloud2] = None
        self._last_sent: dict = {}  # (job_id, track_id) -> timestamp

        self.declare_parameter("object_for_slot_topic", "/usd/Sam3dObjectForSlot")
        object_for_slot_topic = self.get_parameter("object_for_slot_topic").value
        self.sub_slot = self.create_subscription(
            Sam3dObjectForSlotMsg,
            object_for_slot_topic,
            self._on_object_for_slot,
            10,
        )
        self.sub_pc = self.create_subscription(
            PointCloud2,
            self.world_pc_topic,
            self._on_world_pc,
            10,
        )
        self.pub_src_targ = self.create_publisher(
            UsdStringIdSrcTargMsg,
            "/usd/StringIdSrcTarg",
            10,
        )
        self.pub_debug_src = self.create_publisher(PointCloud2, "/debug/registration/src_pc", 10)
        self.pub_debug_targ = self.create_publisher(PointCloud2, "/debug/registration/targ_pc", 10)
        self.pub_timing = self.create_publisher(PipelineStepTiming, "/pipeline/timings", 10)
        self._timing_sequence = 0

        self.get_logger().info(
            f"SAM3D GLB registration bridge: subscribe {object_for_slot_topic}; target={self.registration_target}, world PC from {self.world_pc_topic}"
        )

    def _global_target_from_latest_world(self, center_xyz: Optional[np.ndarray]) -> Optional[PointCloud2]:
        if self._latest_world_pc is None:
            return None
        if center_xyz is None:
            return self._latest_world_pc
        pts = np.asarray(
            point_cloud2.read_points_list(self._latest_world_pc, field_names=("x", "y", "z"), skip_nans=True),
            dtype=np.float64,
        )
        if pts.size == 0:
            return None
        d = np.linalg.norm(pts - center_xyz.reshape(1, 3), axis=1)
        keep = d <= self.global_target_radius_m
        pts = pts[keep]
        if len(pts) < self.global_target_min_points:
            return None
        if len(pts) > self.global_target_max_points:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(pts), size=self.global_target_max_points, replace=False)
            pts = pts[idx]
        out = point_cloud2.create_cloud_xyz32(header=Header(frame_id="odom"), points=pts.astype(np.float32))
        out.header.stamp = self.get_clock().now().to_msg()
        return out

    def _on_world_pc(self, msg: PointCloud2) -> None:
        self._latest_world_pc = msg

    def _on_object_for_slot(self, msg: Sam3dObjectForSlotMsg) -> None:
        data_path = msg.data_path
        track_id = msg.id
        job_id = msg.job_id
        if not data_path or not job_id:
            return
        path_lower = data_path.lower()
        if not (path_lower.endswith(".glb") or path_lower.endswith(".ply")):
            return
        # Debounce by (job_id, track_id)
        key = (job_id, track_id)
        now = self.get_clock().now()
        if key in self._last_sent:
            elapsed = (now - self._last_sent[key]).nanoseconds * 1e-9
            if elapsed < self.debounce_sec:
                return
        t_start = self.get_clock().now()
        points_src = load_object_points(data_path, max_points=self.max_src_points)
        if points_src is None or len(points_src) < 30:
            self.get_logger().warn(
                f"Could not load enough points from {data_path} (got {len(points_src) if points_src is not None else 0}); skip."
            )
            return

        # Target: segment from job dir or global point cloud (parameter for experiments)
        job_dir = self.queue_dir / "output" / job_id
        targ_pc_msg = None
        if self.registration_target == "global":
            if self._latest_world_pc is None:
                self.get_logger().warn(f"[registration target] job_id={job_id}: registration_target=global but no world PC; skip")
                return
            center = None
            pose_path = job_dir / "pose.json"
            if pose_path.exists():
                try:
                    with open(pose_path) as f:
                        pose_data = json.load(f)
                    ip = pose_data.get("initial_position")
                    if isinstance(ip, list) and len(ip) >= 3:
                        center = np.asarray(ip[:3], dtype=np.float64)
                except (OSError, json.JSONDecodeError, TypeError):
                    center = None
            targ_pc_msg = self._global_target_from_latest_world(center)
            if targ_pc_msg is None:
                self.get_logger().warn(
                    f"[registration target] job_id={job_id} id={track_id}: insufficient local global-PC support; fallback full world PC"
                )
                targ_pc_msg = self._latest_world_pc
            self.get_logger().info(
                f"[registration target] job_id={job_id} id={track_id}: using global PC (radius={self.global_target_radius_m:.2f}m)"
            )
        else:
            points_targ, segment_reason = _segment_point_cloud_from_job_dir_with_reason(job_dir)
            if points_targ is None:
                alt_job_dir = self.queue_dir / "input_processed" / job_id
                if alt_job_dir.exists():
                    points_targ, segment_reason = _segment_point_cloud_from_job_dir_with_reason(alt_job_dir)
            if points_targ is not None and len(points_targ) >= 30:
                targ_pc_msg = point_cloud2.create_cloud_xyz32(
                    header=Header(frame_id="odom"),
                    points=points_targ.astype(np.float32),
                )
                targ_pc_msg.header.stamp = now.to_msg()
                self.get_logger().info(
                    f"[registration target] job_id={job_id} id={track_id}: using segment PC from job dir ({len(points_targ)} pts)"
                )
            if targ_pc_msg is None:
                if self._latest_world_pc is None:
                    self.get_logger().warn(
                        f"[registration target] job_id={job_id}: no segment ({segment_reason}) and no world PC; skip"
                    )
                    return
                self.get_logger().warn(
                    f"[registration target] job_id={job_id} id={track_id}: segment unavailable ({segment_reason}); fallback world PC"
                )
                targ_pc_msg = self._latest_world_pc

        src_msg = point_cloud2.create_cloud_xyz32(
            header=Header(frame_id="odom"),
            points=points_src.astype(np.float32),
        )
        src_msg.header.stamp = now.to_msg()
        reg_msg = UsdStringIdSrcTargMsg()
        reg_msg.header = msg.header
        reg_msg.header.stamp = now.to_msg()
        reg_msg.header.frame_id = "odom"
        reg_msg.data_path = data_path
        reg_msg.id = track_id
        reg_msg.job_id = job_id
        reg_msg.src_pc = src_msg
        reg_msg.targ_pc = targ_pc_msg
        self.pub_debug_src.publish(src_msg)
        self.pub_debug_targ.publish(targ_pc_msg)
        t_elapsed = (self.get_clock().now() - t_start).nanoseconds * 1e-6
        timing_msg = PipelineStepTiming()
        timing_msg.header.stamp = now.to_msg()
        timing_msg.header.frame_id = "map"
        timing_msg.node_name = "sam3d_glb_registration_bridge_node"
        timing_msg.step_name = "build_src_targ"
        timing_msg.duration_ms = t_elapsed
        timing_msg.sequence_id = self._timing_sequence
        self._timing_sequence += 1
        self.pub_timing.publish(timing_msg)
        # Initial pose = slot's pose (where we saw the slot). Registration refines from here.
        # Retrieval may switch the object (data_path from another job); we still use the slot's pose as init.
        pose_path = job_dir / "pose.json"
        if pose_path.exists():
            try:
                with open(pose_path) as f:
                    pose_data = json.load(f)
                ip = pose_data.get("initial_position")
                io = pose_data.get("initial_orientation")
                if ip and io and len(ip) >= 3 and len(io) >= 4:
                    reg_msg.initial_pose.position.x = float(ip[0])
                    reg_msg.initial_pose.position.y = float(ip[1])
                    reg_msg.initial_pose.position.z = float(ip[2])
                    reg_msg.initial_pose.orientation.x = float(io[0])
                    reg_msg.initial_pose.orientation.y = float(io[1])
                    reg_msg.initial_pose.orientation.z = float(io[2])
                    reg_msg.initial_pose.orientation.w = float(io[3])
            except (OSError, json.JSONDecodeError, TypeError):
                pass
        self.pub_src_targ.publish(reg_msg)
        self._last_sent[key] = now
        self.get_logger().info(
            f"Published registration request: job_id={job_id} id={track_id} src_pts={len(points_src)}"
        )


def main():
    rclpy.init()
    node = Sam3dGlbRegistrationBridgeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
