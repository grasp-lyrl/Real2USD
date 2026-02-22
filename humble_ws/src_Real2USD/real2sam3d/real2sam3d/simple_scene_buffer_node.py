"""
Simple scene buffer node: subscribes to registration output (/usd/StringIdPose),
maintains a single scene buffer, and writes scene_graph.json + scene.glb.

Contract: no extra transformations. All poses are already in Z-up odom; object
meshes are Z-up at origin. We only apply the given pose T so that v_odom = R @ v + t.

  - scene_graph_sam3d_only.json / scene_sam3d_only.glb: initialization of every slot pose (one per slot,
    slot's object at slot's initial pose from injector). Baseline before retrieval/registration.
  - scene_graph.json / scene.glb: same slots, but object may be switched by retrieval and pose refined by
    registration. Single transform: object (data_path) + registration pose; no extra transformation.

Per-slot GLBs (when write_per_slot_glb=true): for each output/<job_id>/ we write
  - object.glb: vanilla from SAM3D (worker).
  - object_odom.glb: vanilla + transform_odom_from_raw (injector).
  - object_registered.glb: object + registration pose (one transform; written to slot's job dir when that slot is in buffer).
"""

import json
import os
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R

from custom_message.msg import UsdStringIdPoseMsg

# Raw object.glb is Y-up (SAM3D). Registration outputs yaw-only (R_z) in odom, so we must convert
# mesh to Z-up before applying the pose. ply_frame_utils does v_row @ R_flip_z @ R_yup_to_zup => column: v_zup = (R_flip_z @ R_yup_to_zup).T @ v_raw
try:
    from real2sam3d.ply_frame_utils import R_flip_z, R_yup_to_zup
    T_RAW_TO_ZUP = np.eye(4, dtype=np.float64)
    T_RAW_TO_ZUP[:3, :3] = (R_flip_z @ R_yup_to_zup).T
except Exception:
    T_RAW_TO_ZUP = np.eye(4, dtype=np.float64)


def _pose_to_matrix(position_xyz, quat_xyzw):
    """Build 4x4 transform from position (3,) and quaternion (x,y,z,w)."""
    T = np.eye(4)
    T[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
    T[:3, 3] = position_xyz
    return T


# Per-slot GLB filenames (in output/<job_id>/)
OBJECT_ODOM_GLB = "object_odom.glb"           # vanilla + injector transform (camera → odom), Z-up in odom
OBJECT_REGISTERED_GLB = "object_registered.glb"  # object + registration pose (single transform, no extra)


def _apply_transform_to_mesh(mesh, T: np.ndarray):
    """Apply 4x4 transform to mesh vertices; return new mesh (copy) with transformed vertices."""
    try:
        import trimesh
    except ImportError:
        return None
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        return mesh.copy()
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    ones = np.ones((verts.shape[0], 1), dtype=np.float64)
    verts_h = np.hstack([verts, ones])
    verts_new = (T @ verts_h.T).T[:, :3]
    out = mesh.copy()
    out.vertices = verts_new
    if hasattr(out, "volume") and out.volume < 0:
        out.invert()
    return out


def _load_meshes_from_path(data_path: str):
    """Load trimesh geometry from data_path (GLB or PLY). Returns list of Trimesh or None."""
    try:
        import trimesh
    except ImportError:
        return None
    path = Path(data_path)
    if not path.exists():
        return None
    suffix = path.suffix.lower()
    if suffix == ".ply":
        # Prefer object.glb in same dir if present
        glb = path.parent / "object.glb"
        if glb.exists():
            path = glb
            suffix = ".glb"
    if suffix != ".glb":
        return None
    try:
        scene = trimesh.load(str(path), process=False)
        if isinstance(scene, trimesh.Scene):
            meshes = []
            # Flatten scene graph transforms so loaded vertices match exported GLB coordinates.
            for node_name in scene.graph.nodes_geometry:
                try:
                    node_tf, geom_name = scene.graph[node_name]
                    geom = scene.geometry.get(geom_name)
                    if isinstance(geom, trimesh.Trimesh):
                        g = geom.copy()
                        g.apply_transform(np.asarray(node_tf, dtype=np.float64))
                        meshes.append(g)
                except Exception:
                    continue
            return meshes
        if isinstance(scene, trimesh.Trimesh):
            return [scene]
    except Exception:
        pass
    return None


def _scan_sam3d_only_scene(output_dir: Path):
    """
    Scan output_dir/output/ for job dirs; yield (id, data_path, position, orientation, transform_odom_from_raw).
    Mesh is vanilla object.glb (sam3d_raw). If transform_odom_from_raw is present, use it to place raw mesh in odom; else use initial_position/orientation.
    """
    out_sub = output_dir / "output"
    if not out_sub.exists():
        return
    for job_dir in out_sub.iterdir():
        if not job_dir.is_dir():
            continue
        pose_path = job_dir / "pose.json"
        glb_path = job_dir / "object.glb"
        if not pose_path.exists() or not glb_path.exists():
            continue
        try:
            with open(pose_path) as f:
                pose = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        ip = pose.get("initial_position")
        io = pose.get("initial_orientation")
        T_raw = pose.get("transform_odom_from_raw")
        if T_raw is None:
            # Fallback: compose from explicit two-step transforms if present.
            T_cam = pose.get("transform_cam_from_raw")
            T_odom_cam = pose.get("transform_odom_from_cam")
            if T_cam is not None and T_odom_cam is not None:
                try:
                    A = np.array(T_odom_cam, dtype=np.float64)
                    B = np.array(T_cam, dtype=np.float64)
                    if A.shape == (4, 4) and B.shape == (4, 4):
                        T_raw = (A @ B).tolist()
                except Exception:
                    T_raw = None
        if T_raw is None and (not ip or not io or len(ip) < 3 or len(io) < 4):
            continue
        job_id = pose.get("job_id", job_dir.name)
        label = pose.get("label")
        yield (job_id, str(glb_path.resolve()), ip or [], io or [], T_raw, label)


class SimpleSceneBufferNode(Node):
    """
    Subscribes to /usd/StringIdPose (registration output), keeps a single scene buffer,
    and periodically writes scene_graph.json + scene.glb (registration) and optionally
    scene_graph_sam3d_only.json + scene_sam3d_only.glb (SAM3D-only poses for comparison).
    """

    def __init__(self):
        super().__init__("simple_scene_buffer_node")

        self.declare_parameter("output_dir", "/data/sam3d_queue")
        self.declare_parameter("write_interval_sec", 5.0)
        self.declare_parameter("scene_json_name", "scene_graph.json")
        self.declare_parameter("scene_glb_name", "scene.glb")
        self.declare_parameter("write_sam3d_only_output", True)
        self.declare_parameter("scene_json_sam3d_only", "scene_graph_sam3d_only.json")
        self.declare_parameter("scene_glb_sam3d_only", "scene_sam3d_only.glb")
        self.declare_parameter("write_per_slot_glb", True)

        self._output_dir = Path(self.get_parameter("output_dir").value)
        self._write_interval_sec = self.get_parameter("write_interval_sec").value
        self._scene_json_name = self.get_parameter("scene_json_name").value
        self._scene_glb_name = self.get_parameter("scene_glb_name").value
        self._write_sam3d_only = self.get_parameter("write_sam3d_only_output").value
        self._scene_json_sam3d_only = self.get_parameter("scene_json_sam3d_only").value
        self._scene_glb_sam3d_only = self.get_parameter("scene_glb_sam3d_only").value
        self._write_per_slot_glb = self.get_parameter("write_per_slot_glb").value

        # id -> { "data_path", "position", "orientation" }
        self._buffer: dict[int, dict] = {}

        self._sub = self.create_subscription(
            UsdStringIdPoseMsg,
            "/usd/StringIdPose",
            self._pose_callback,
            10,
        )
        self._timer = self.create_timer(self._write_interval_sec, self._write_scene)

        self.get_logger().info(
            "simple_scene_buffer_node: output_dir=%s, write_interval=%.1fs, sam3d_only=%s"
            % (self._output_dir, self._write_interval_sec, self._write_sam3d_only)
        )

    def _pose_callback(self, msg: UsdStringIdPoseMsg):
        position = [
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ]
        orientation = [
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ]
        self._buffer[int(msg.id)] = {
            "data_path": msg.data_path,
            "position": position,
            "orientation": orientation,
            "job_id": getattr(msg, "job_id", "") or "",
        }

    def _write_scene(self):
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # 1) Registration-based scene graph JSON + joint GLB
        # One entry per slot: object (from retrieval) + refined pose (from registration). Include YOLO label and retrieved_object_label when retrieval swapped.
        scene_list = []
        out_output = self._output_dir / "output"
        for obj_id, data in self._buffer.items():
            job_id = data.get("job_id") or ""
            data_path = data["data_path"]
            label = None
            retrieved_object_label = None
            if job_id and out_output.exists():
                slot_pose_path = out_output / job_id / "pose.json"
                if slot_pose_path.exists():
                    try:
                        with open(slot_pose_path) as f:
                            slot_pose = json.load(f)
                        label = slot_pose.get("label")
                    except (OSError, json.JSONDecodeError):
                        pass
                object_job_dir = Path(data_path).parent
                if object_job_dir.name != job_id:
                    obj_pose_path = object_job_dir / "pose.json"
                    if obj_pose_path.exists():
                        try:
                            with open(obj_pose_path) as f:
                                obj_pose = json.load(f)
                            retrieved_object_label = obj_pose.get("label")
                        except (OSError, json.JSONDecodeError):
                            pass
            entry = {
                "id": obj_id,
                "data_path": data_path,
                "position": data["position"],
                "orientation": data["orientation"],
            }
            if label is not None:
                entry["label"] = label
            if retrieved_object_label is not None:
                entry["retrieved_object_label"] = retrieved_object_label
            scene_list.append(entry)
        json_path = self._output_dir / self._scene_json_name
        try:
            with open(json_path, "w") as f:
                json.dump({"objects": scene_list}, f, indent=2)
            self.get_logger().debug("Wrote %s (%d objects)" % (json_path, len(scene_list)))
        except Exception as e:
            self.get_logger().warn("Failed to write %s: %s" % (json_path, e))

        glb_path = self._output_dir / self._scene_glb_name
        try:
            self._write_joint_glb_from_list(glb_path, scene_list)
        except Exception as e:
            self.get_logger().warn("Failed to write joint GLB %s: %s" % (glb_path, e))

        # 1b) Per-slot: object_registered.glb (raw→Z-up then registration pose)
        if self._write_per_slot_glb:
            for _obj_id, data in self._buffer.items():
                T_pose = _pose_to_matrix(data["position"], data["orientation"])
                self._write_slot_glb(
                    data_path=data["data_path"],
                    pose_4x4=T_pose @ T_RAW_TO_ZUP,
                    out_filename=OBJECT_REGISTERED_GLB,
                )

        # 2) SAM3D-only scene (no registration): from output/ pose.json initial poses
        if self._write_sam3d_only:
            sam3d_list = list(_scan_sam3d_only_scene(self._output_dir))
            if sam3d_list:
                objs = []
                for oid, dp, pos, quat, T_raw, label in sam3d_list:
                    obj = {"id": oid, "data_path": dp, "position": pos, "orientation": quat, "transform_odom_from_raw": T_raw}
                    if label is not None:
                        obj["label"] = label
                    objs.append(obj)
                json_sam3d = self._output_dir / self._scene_json_sam3d_only
                try:
                    with open(json_sam3d, "w") as f:
                        json.dump({"objects": objs, "source": "sam3d_only_no_registration"}, f, indent=2)
                    self.get_logger().debug("Wrote %s (%d objects)" % (json_sam3d, len(sam3d_list)))
                except Exception as e:
                    self.get_logger().warn("Failed to write %s: %s" % (json_sam3d, e))
                glb_sam3d = self._output_dir / self._scene_glb_sam3d_only
                try:
                    self._write_joint_glb_from_list(glb_sam3d, objs)
                except Exception as e:
                    self.get_logger().warn("Failed to write SAM3D-only GLB %s: %s" % (glb_sam3d, e))
                # 2b) Per-slot: object_odom.glb (vanilla + transform_odom_from_raw) for each slot
                if self._write_per_slot_glb:
                    for oid, dp, pos, quat, T_raw, _label in sam3d_list:
                        if T_raw is None:
                            continue
                        job_dir = Path(dp).parent
                        self._write_slot_glb(
                            data_path=dp,
                            pose_4x4=np.array(T_raw, dtype=np.float64),
                            out_filename=OBJECT_ODOM_GLB,
                            job_dir=job_dir,
                        )

    def _write_slot_glb(self, data_path: str, pose_4x4: np.ndarray, out_filename: str, job_dir: Path = None):
        """Load mesh from data_path, apply pose_4x4 (4x4), write to job_dir/out_filename. If job_dir is None, use parent of data_path."""
        try:
            import trimesh
        except ImportError:
            return
        path = Path(data_path)
        if job_dir is None:
            job_dir = path.parent
        out_path = job_dir / out_filename
        meshes = _load_meshes_from_path(data_path)
        if not meshes:
            return
        if pose_4x4.shape != (4, 4):
            return
        try:
            combined = []
            for m in meshes:
                transformed = _apply_transform_to_mesh(m, pose_4x4)
                if transformed is not None:
                    combined.append(transformed)
            if not combined:
                return
            if len(combined) == 1:
                combined[0].export(str(out_path))
            else:
                scene = trimesh.Scene()
                for i, m in enumerate(combined):
                    scene.add_geometry(m, node_name="mesh_%d" % i)
                scene.export(str(out_path))
            self.get_logger().debug("Wrote per-slot %s" % out_path)
        except Exception as e:
            self.get_logger().warn("Failed to write per-slot GLB %s: %s" % (out_path, e))

    def _write_joint_glb_from_list(self, glb_path: Path, scene_list: list):
        """Build and write joint GLB. If transform_odom_from_raw present, apply 4x4 to raw mesh; else use position+orientation."""
        try:
            import trimesh
        except ImportError:
            self.get_logger().warn("trimesh not available; skipping joint GLB")
            return
        scene = trimesh.Scene()
        added = 0
        for item in scene_list:
            path = item["data_path"]
            T_raw = item.get("transform_odom_from_raw")
            if T_raw is not None:
                T = np.array(T_raw, dtype=np.float64)
                if T.shape != (4, 4):
                    T = np.eye(4)
            else:
                # Registration path: pose is yaw-only in odom; mesh is raw Y-up, so convert to Z-up then apply pose
                pos = np.array(item["position"], dtype=np.float64)
                quat = np.array(item["orientation"], dtype=np.float64)
                T = _pose_to_matrix(pos, quat) @ T_RAW_TO_ZUP
            oid = item.get("id", added)
            meshes = _load_meshes_from_path(path)
            if not meshes:
                continue
            for i, mesh in enumerate(meshes):
                name = "obj_%s" % oid if len(meshes) == 1 else "obj_%s_%d" % (oid, i)
                scene.add_geometry(mesh.copy(), transform=T, node_name=name)
                added += 1
        if added == 0:
            self.get_logger().debug("No meshes to export for joint GLB %s" % glb_path)
            return
        scene.export(str(glb_path))
        self.get_logger().info("Wrote joint GLB %s (%d meshes)" % (glb_path, added))

    def _write_joint_glb(self, glb_path: Path):
        """Legacy: write joint GLB from internal _buffer (registration poses)."""
        scene_list = [
            {"id": obj_id, "data_path": data["data_path"], "position": data["position"], "orientation": data["orientation"]}
            for obj_id, data in self._buffer.items()
        ]
        self._write_joint_glb_from_list(glb_path, scene_list)


def main(args=None):
    rclpy.init(args=args)
    node = SimpleSceneBufferNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
