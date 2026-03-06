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
import threading
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R

from custom_message.msg import ObjectInBufferMsg, UsdStringIdPoseMsg

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


def _scan_sam3d_only_scene(output_dir: Path, use_registration_when_present: bool = True):
    """
    Scan output_dir/output/ for job dirs; yield (id, data_path, position, orientation, transform_odom_from_raw, label).

    When use_registration_when_present=True: prefer registration (position, orientation, registered_data_path)
    from pose.json when present; else use initial_position/orientation and transform_odom_from_raw.
    When use_registration_when_present=False: always use initial pose and the slot's own object.glb only
    (no retrieval swap, no registration). Use False for scene_graph_sam3d_only.json / scene_sam3d_only.glb
    so they show the true SAM3D-only baseline (slot-initial mesh and pose only).
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
        # Use registration output when present (written by registration_node) only if requested
        if use_registration_when_present:
            reg_pos = pose.get("position")
            reg_ori = pose.get("orientation")
            registered_data_path = pose.get("registered_data_path")
            if reg_pos and reg_ori and len(reg_pos) >= 3 and len(reg_ori) >= 4:
                data_path = registered_data_path if registered_data_path else str(glb_path.resolve())
                job_id = pose.get("job_id", job_dir.name)
                label = pose.get("label")
                yield (job_id, data_path, reg_pos, reg_ori, None, label)
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
    Subscribes to /usd/StringIdPose (registration output), keeps a single scene buffer.
    - JSON (scene_graph.json, scene_graph_sam3d_only.json) is written first and is decoupled from GLB:
      on each timer tick we write JSON synchronously, then start the GLB thread. So JSON is always
      complete even if GLB fails or the process is killed (e.g. OOM) during GLB writing.
    - JSON is also updated on pose (write_on_pose debounce). GLB runs at most one at a time in a background thread.
    """

    def __init__(self):
        super().__init__("simple_scene_buffer_node")

        self.declare_parameter("output_dir", "/data/sam3d_queue")
        self.declare_parameter("write_interval_sec", 5.0)
        self.declare_parameter("write_on_pose", True)
        self.declare_parameter("write_on_pose_debounce_sec", 2.0)
        self.declare_parameter("scene_json_name", "scene_graph.json")
        self.declare_parameter("scene_glb_name", "scene.glb")
        self.declare_parameter("write_sam3d_only_output", True)
        self.declare_parameter("scene_json_sam3d_only", "scene_graph_sam3d_only.json")
        self.declare_parameter("scene_glb_sam3d_only", "scene_sam3d_only.glb")
        self.declare_parameter("write_per_slot_glb", True)

        self._output_dir = Path(self.get_parameter("output_dir").value)
        self._write_interval_sec = self.get_parameter("write_interval_sec").value
        p_wop = self.get_parameter("write_on_pose").value
        self._write_on_pose = p_wop is True or (isinstance(p_wop, str) and p_wop.lower() == "true")
        self._write_on_pose_debounce_sec = float(self.get_parameter("write_on_pose_debounce_sec").value)
        self._defer_write_timer = None
        self._scene_json_name = self.get_parameter("scene_json_name").value
        self._scene_glb_name = self.get_parameter("scene_glb_name").value
        self._write_sam3d_only = self.get_parameter("write_sam3d_only_output").value
        self._scene_json_sam3d_only = self.get_parameter("scene_json_sam3d_only").value
        self._scene_glb_sam3d_only = self.get_parameter("scene_glb_sam3d_only").value
        self._write_per_slot_glb = self.get_parameter("write_per_slot_glb").value

        # job_id (or str(track_id) when job_id missing) -> { "data_path", "position", "orientation", "job_id", "track_id" }
        # Keying by job_id gives one entry per job that received a pose, matching SAM3D-only count; keying by track_id would collapse multiple jobs into one.
        self._buffer: dict = {}
        self._glb_write_lock = threading.Lock()
        self._glb_write_in_progress = False

        self._sub = self.create_subscription(
            UsdStringIdPoseMsg,
            "/usd/StringIdPose",
            self._pose_callback,
            100,
        )
        self._pub_in_buffer = self.create_publisher(
            ObjectInBufferMsg,
            "/pipeline/object_in_buffer",
            10,
        )
        self._timer = self.create_timer(self._write_interval_sec, self._write_scene)

        self.get_logger().info(
            "simple_scene_buffer_node: output_dir=%s, write_interval=%.1fs, sam3d_only=%s, write_on_pose=%s"
            % (self._output_dir, self._write_interval_sec, self._write_sam3d_only, self._write_on_pose)
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
        job_id = (getattr(msg, "job_id", None) or "").strip()
        # Key by job_id when present so we get one entry per job (matches SAM3D-only); else fall back to track_id so older msgs without job_id still work
        key = job_id if job_id else str(int(msg.id))
        self._buffer[key] = {
            "data_path": msg.data_path,
            "position": position,
            "orientation": orientation,
            "job_id": job_id,
            "track_id": int(msg.id),
        }
        # "Done" for full-pipeline E2E: object is in buffer (wall clock)
        if job_id:
            ob = ObjectInBufferMsg()
            ob.header.stamp = self.get_clock().now().to_msg()
            ob.header.frame_id = "map"
            ob.job_id = job_id
            # Worker-reported inference time (from pose.json) for pipeline timing (inference per object, not latency)
            pose_path = self._output_dir / "output" / job_id / "pose.json"
            inference_ms = 0.0
            if pose_path.exists():
                try:
                    with open(pose_path) as f:
                        pose_data = json.load(f)
                    inference_ms = float(pose_data.get("inference_ms", 0) or 0)
                except (OSError, json.JSONDecodeError, TypeError):
                    pass
            ob.inference_ms = inference_ms
            self._pub_in_buffer.publish(ob)
        self.get_logger().info(
            "Buffer: pose id=%s job_id=%s → buffer_size=%d"
            % (int(msg.id), job_id or "(none)", len(self._buffer))
        )
        # Schedule JSON-only write so scene_graph.json is updated quickly (GLB is on the slow timer only)
        if self._write_on_pose and self._defer_write_timer is None:
            self._defer_write_timer = self.create_timer(
                self._write_on_pose_debounce_sec,
                self._on_defer_write,
            )

    def _on_defer_write(self):
        if self._defer_write_timer is not None:
            self._defer_write_timer.cancel()
            self._defer_write_timer = None
        self.get_logger().info("Debounced JSON write (write_on_pose): buffer=%d" % len(self._buffer))
        self._write_scene_json_only()

    def _build_scene_list(self):
        """Build registration scene_list from buffer (for JSON and for GLB snapshot)."""
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
                "job_id": job_id,
                "data_path": data_path,
                "position": data["position"],
                "orientation": data["orientation"],
            }
            if data.get("track_id") is not None:
                entry["track_id"] = data["track_id"]
            if label is not None:
                entry["label"] = label
            if retrieved_object_label is not None:
                entry["retrieved_object_label"] = retrieved_object_label
            scene_list.append(entry)
        return scene_list

    def _build_scene_list_full(self):
        """Build scene list with same object set as scan (all jobs in output/). For each job use
        registration pose from _buffer when available, else initial pose from pose.json (not registration
        from pose.json). So scene_graph.json has the same count as scene_graph_sam3d_only.json and fills
        at the same rate, but uses true registration-only when we have buffer data."""
        out_output = self._output_dir / "output"
        # Scan with registration when present to get full job set; we override with initial when buffer missing
        sam3d_list = list(_scan_sam3d_only_scene(self._output_dir, use_registration_when_present=True))
        if not sam3d_list:
            return self._build_scene_list()
        scene_list = []
        for oid, dp, pos, quat, T_raw, label in sam3d_list:
            # Use registration data from buffer when we have it; else use initial pose from pose.json
            data = self._buffer.get(oid)
            if data is not None:
                position = data["position"]
                orientation = data["orientation"]
                data_path = data["data_path"]
                track_id = data.get("track_id")
            else:
                # Buffer missing: use initial pose and slot's object.glb (read from pose.json)
                try:
                    pose_path = out_output / oid / "pose.json"
                    if pose_path.exists():
                        with open(pose_path) as f:
                            p = json.load(f)
                        ip = p.get("initial_position")
                        io = p.get("initial_orientation")
                        if ip and io and len(ip) >= 3 and len(io) >= 4:
                            position = ip
                            orientation = io
                            data_path = str((out_output / oid / "object.glb").resolve())
                        else:
                            position = pos
                            orientation = quat
                            data_path = dp
                        track_id = p.get("track_id")
                        if isinstance(track_id, float):
                            track_id = int(track_id)
                        if label is None:
                            label = p.get("label")
                    else:
                        position = pos
                        orientation = quat
                        data_path = dp
                        track_id = None
                except (OSError, json.JSONDecodeError, TypeError):
                    position = pos
                    orientation = quat
                    data_path = dp
                    track_id = None
            entry = {"id": oid, "job_id": oid, "data_path": data_path, "position": position, "orientation": orientation}
            if track_id is not None:
                entry["track_id"] = track_id
            if label is not None:
                entry["label"] = label
            scene_list.append(entry)
        return scene_list

    def _write_scene_json_only(self):
        """Fast path: update scene_graph.json (and optionally scene_graph_sam3d_only.json) only. No GLB.
        scene_graph.json uses the same object set as the scan (all jobs in output/) so it fills at the
        same rate as scene_graph_sam3d_only.json; registration pose is used when available."""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        # Build full list (all jobs in output/, registration pose when available) so scene_graph.json matches sam3d_only count
        scene_list = self._build_scene_list_full()
        json_path = self._output_dir / self._scene_json_name
        try:
            with open(json_path, "w") as f:
                json.dump({"objects": scene_list}, f, indent=2)
            self.get_logger().info("Wrote %s (%d objects)" % (json_path, len(scene_list)))
        except Exception as e:
            self.get_logger().warn("Failed to write %s: %s" % (json_path, e))

        if self._write_sam3d_only:
            # Initial poses and slot object.glb only (no registration) so sam3d_only differs from scene_graph.json
            sam3d_list = list(_scan_sam3d_only_scene(self._output_dir, use_registration_when_present=False))
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
                    self.get_logger().info("Wrote %s (%d objects)" % (json_sam3d, len(sam3d_list)))
                except Exception as e:
                    self.get_logger().warn("Failed to write %s: %s" % (json_sam3d, e))

    def _write_scene(self):
        """Timer path: (1) Always write scene_graph.json + scene_graph_sam3d_only.json first (decoupled from GLB).
        (2) Then start GLB write in background if none in progress. JSON is complete even if GLB fails or OOMs."""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        # 1) Write JSON synchronously so it is always complete and independent of GLB
        self._write_scene_json_only()
        # 2) Start GLB write in background (one at a time)
        with self._glb_write_lock:
            if self._glb_write_in_progress:
                self.get_logger().debug("GLB write still in progress; skipping this timer tick")
                return
            self._glb_write_in_progress = True
        scene_list = self._build_scene_list()
        buffer_snapshot = [(k, dict(d)) for k, d in self._buffer.items()]
        self.get_logger().info(
            "Starting GLB write: buffer=%d entries (scene.glb / scene_sam3d_only.glb in background)"
            % len(self._buffer)
        )
        t = threading.Thread(target=self._write_scene_io, args=(scene_list, buffer_snapshot))
        t.daemon = True
        t.start()

    def _write_scene_io(self, scene_list: list, buffer_snapshot: list):
        """Runs in a background thread: heavy GLB writing so the executor can process pose callbacks.
        Uses snapshot data only; does not read self._buffer. Clears _glb_write_in_progress when done.
        """
        try:
            out_output = self._output_dir / "output"
            glb_path = self._output_dir / self._scene_glb_name
            try:
                self._write_joint_glb_from_list(glb_path, scene_list)
            except Exception as e:
                self.get_logger().warn("Failed to write joint GLB %s: %s" % (glb_path, e))

            if self._write_per_slot_glb:
                for _obj_id, data in buffer_snapshot:
                    T_pose = _pose_to_matrix(data["position"], data["orientation"])
                    job_id = data.get("job_id") or ""
                    job_dir = (out_output / job_id) if job_id and out_output.exists() else None
                    self._write_slot_glb(
                        data_path=data["data_path"],
                        pose_4x4=T_pose @ T_RAW_TO_ZUP,
                        out_filename=OBJECT_REGISTERED_GLB,
                        job_dir=job_dir,
                    )

            if self._write_sam3d_only:
                sam3d_list = list(_scan_sam3d_only_scene(self._output_dir, use_registration_when_present=False))
                if sam3d_list:
                    n_buf = len(scene_list)
                    sam3d_count = len(sam3d_list)
                    if sam3d_count > n_buf:
                        self.get_logger().info(
                            "Registration scene: %d meshes (%d buffer entries). SAM3D-only has %d jobs → %d slots not yet in scene.glb (registration backlog or still processing)."
                            % (n_buf, n_buf, sam3d_count, sam3d_count - n_buf)
                        )
                    objs = []
                    for oid, dp, pos, quat, T_raw, label in sam3d_list:
                        obj = {"id": oid, "data_path": dp, "position": pos, "orientation": quat, "transform_odom_from_raw": T_raw}
                        if label is not None:
                            obj["label"] = label
                        objs.append(obj)
                    # GLB only; JSON is updated by _write_scene_json_only()
                    glb_sam3d = self._output_dir / self._scene_glb_sam3d_only
                    try:
                        self._write_joint_glb_from_list(glb_sam3d, objs)
                    except Exception as e:
                        self.get_logger().warn("Failed to write SAM3D-only GLB %s: %s" % (glb_sam3d, e))
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
        finally:
            with self._glb_write_lock:
                self._glb_write_in_progress = False

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


def write_joint_glb_from_list_standalone(glb_path: Path, scene_list: list, run_dir=None, log=None):
    """Write joint GLB from scene_list without a ROS node. Delegates to glb_export_standalone (no rclpy).
    If run_dir is set, accumulated RealSense point cloud from run_dir is included in the GLB."""
    from real2sam3d.glb_export_standalone import write_joint_glb_from_list_standalone as _write
    _write(glb_path, scene_list, run_dir=run_dir, log=log)


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
