"""
Simple scene buffer node for real2usd: subscribes to registration output
(/usd/StringIdPose), keeps a single scene buffer keyed by track_id, and
writes scene_graph.json, optionally scene.glb, and optionally scene.usda.

Like real2sam3d's simple_scene_buffer_node but without SAM3D job dirs or
sam3d_only outputs. All poses are in odom (position + orientation quat xyzw).
GLB export supports .glb/.ply data_paths via trimesh; .usd paths are written
in JSON and in USDA (references + OBB). OBB (oriented bounding box) is computed
per asset in local space and written as a cube prim under each instance.
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

try:
    from pxr import Gf, Sdf, Usd, UsdGeom
    _HAS_PXR = True
except ImportError:
    _HAS_PXR = False


def _pose_to_matrix(position_xyz, quat_xyzw):
    """Build 4x4 transform from position (3,) and quaternion (x,y,z,w)."""
    T = np.eye(4)
    T[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
    T[:3, 3] = position_xyz
    return T


def _load_meshes_from_path(data_path: str):
    """Load trimesh geometry from data_path (GLB or PLY). Returns list of Trimesh or None. USD not supported by trimesh."""
    try:
        import trimesh
    except ImportError:
        return None
    path = Path(data_path)
    if not path.exists():
        return None
    suffix = path.suffix.lower()
    if suffix not in (".glb", ".ply", ".obj"):
        return None
    try:
        scene = trimesh.load(str(path), process=False)
        if isinstance(scene, trimesh.Scene):
            meshes = []
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
            return meshes if meshes else None
        if isinstance(scene, trimesh.Trimesh):
            return [scene]
    except Exception:
        pass
    return None


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


def _compute_usd_local_obb(usd_path: str):
    """
    Compute oriented bounding box of a USD asset in its local coordinate system.
    Returns (local_center, local_half_extents) as numpy arrays (3,) or None if load fails.
    Local AABB in asset space is the OBB (center + half-extents); orientation is identity in local.
    """
    if not _HAS_PXR:
        return None
    path = Path(usd_path)
    if not path.exists() or path.suffix.lower() not in (".usd", ".usda", ".usdc", ".usdz"):
        return None
    try:
        stage = Usd.Stage.Open(str(path))
        if not stage:
            return None
        meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
        vertices = []
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
                            scaled = (transformed_p[0] * meters_per_unit, transformed_p[1] * meters_per_unit, transformed_p[2] * meters_per_unit)
                            vertices.append(scaled)
        if not vertices:
            return None
        verts = np.array(vertices, dtype=np.float64)
        min_bounds = np.min(verts, axis=0)
        max_bounds = np.max(verts, axis=0)
        local_center = (min_bounds + max_bounds) * 0.5
        local_half_extents = (max_bounds - min_bounds) * 0.5
        return local_center, local_half_extents
    except Exception:
        return None


def _write_usda_scene(usda_path: Path, scene_list: list, log=None):
    """
    Write a USDA stage with one Xform per object (reference to usd path + transform),
    matching usd_buffer_node field names (usd, position, quatWXYZ). OBB cube per object.
    """
    if not _HAS_PXR:
        if log:
            log.warn("pxr (Usd) not available; skipping USDA write")
        return
    try:
        # Only include objects with non-empty usd path (like usd_buffer_node uses obj['usd'])
        valid = []
        for item in scene_list:
            usd_path = (item.get("usd") or "").strip()
            if not usd_path:
                continue
            valid.append({**item, "usd": os.path.abspath(usd_path)})
        if not valid:
            if log:
                log.debug("No valid usd paths for USDA write")
            return

        stage = Usd.Stage.CreateNew(str(usda_path))
        stage.SetMetadata("metersPerUnit", 1.0)
        world_prim = UsdGeom.Xform.Define(stage, Sdf.Path("/World"))
        stage.SetDefaultPrim(world_prim.GetPrim())

        for idx, item in enumerate(valid):
            usd_path = item["usd"]
            pos = np.array(item["position"], dtype=np.float64)
            # Buffer stores quatWXYZ [w,x,y,z] like usd_buffer_node
            q = item["quatWXYZ"]
            quat_w = float(q[0])
            quat_xyz = Gf.Vec3f(float(q[1]), float(q[2]), float(q[3]))

            prim_path = Sdf.Path("/World/object_%d" % idx)
            xform_prim = stage.DefinePrim(prim_path, "Xform")
            xform_prim.GetReferences().AddReference(usd_path)
            xform = UsdGeom.Xformable(xform_prim)
            xform.ClearXformOpOrder()
            t_op = xform.AddTranslateOp()
            t_op.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
            r_op = xform.AddOrientOp()
            r_op.Set(Gf.Quatf(quat_w, quat_xyz))

            obb_data = _compute_usd_local_obb(usd_path)
            if obb_data is not None:
                local_center, local_half_extents = obb_data
                quat_xyzw = [q[1], q[2], q[3], q[0]]
                r_mat = R.from_quat(quat_xyzw).as_matrix()
                world_center = r_mat @ local_center + pos
                obb_path = prim_path.AppendPath("OBB")
                obb_prim = stage.DefinePrim(obb_path, "Xform")
                cube = UsdGeom.Cube.Define(stage, obb_path.AppendPath("Cube"))
                cube.CreateSizeAttr(1.0)
                cube_xform = UsdGeom.Xformable(cube.GetPrim())
                cube_xform.ClearXformOpOrder()
                s_op = cube_xform.AddScaleOp()
                s_op.Set(Gf.Vec3d(2.0 * local_half_extents[0], 2.0 * local_half_extents[1], 2.0 * local_half_extents[2]))
                o_op = cube_xform.AddOrientOp()
                o_op.Set(Gf.Quatf(quat_w, quat_xyz))
                tc_op = cube_xform.AddTranslateOp()
                tc_op.Set(Gf.Vec3d(float(world_center[0]), float(world_center[1]), float(world_center[2])))
                cube.GetPrim().SetCustomDataByKey("obb", 1)

        stage.GetRootLayer().Save()
        if log:
            log.info("Wrote USDA %s (%d objects, OBB per object)" % (usda_path, len(valid)))
    except Exception as e:
        if log:
            log.warn("Failed to write USDA %s: %s" % (usda_path, e))


class SimpleSceneBufferNode(Node):
    """
    Subscribes to /usd/StringIdPose (registration output), keeps a buffer keyed by track_id.
    Writes scene_graph.json and optionally a joint scene.glb (for .glb/.ply data_paths).
    """

    def __init__(self):
        super().__init__("simple_scene_buffer_node")

        self.declare_parameter("output_dir", "/data/real2usd_scene")
        self.declare_parameter("write_interval_sec", 5.0)
        self.declare_parameter("write_on_pose", True)
        self.declare_parameter("write_on_pose_debounce_sec", 2.0)
        self.declare_parameter("scene_json_name", "scene_graph.json")
        self.declare_parameter("scene_glb_name", "scene.glb")
        self.declare_parameter("scene_usda_name", "scene.usda")
        self.declare_parameter("write_glb", True)
        self.declare_parameter("write_usda", True)

        self._output_dir = Path(self.get_parameter("output_dir").value)
        self._write_interval_sec = float(self.get_parameter("write_interval_sec").value)
        p_wop = self.get_parameter("write_on_pose").value
        self._write_on_pose = p_wop is True or (isinstance(p_wop, str) and p_wop.lower() == "true")
        self._write_on_pose_debounce_sec = float(self.get_parameter("write_on_pose_debounce_sec").value)
        self._defer_write_timer = None
        self._scene_json_name = self.get_parameter("scene_json_name").value
        self._scene_glb_name = self.get_parameter("scene_glb_name").value
        self._scene_usda_name = self.get_parameter("scene_usda_name").value
        self._write_glb = self.get_parameter("write_glb").value
        self._write_usda = self.get_parameter("write_usda").value

        # track_id (int) -> { "usd", "position", "quatWXYZ", "track_id" } (same as usd_buffer_node)
        self._buffer = {}
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
            "simple_scene_buffer_node: output_dir=%s, write_interval=%.1fs, write_glb=%s, write_usda=%s, write_on_pose=%s"
            % (self._output_dir, self._write_interval_sec, self._write_glb, self._write_usda, self._write_on_pose)
        )

    def _pose_callback(self, msg: UsdStringIdPoseMsg):
        # Match usd_buffer_node: position [x,y,z], quatWXYZ [w,x,y,z]
        position = [
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ]
        quatWXYZ = [
            msg.pose.orientation.w,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
        ]
        track_id = int(msg.id)
        key = track_id
        self._buffer[key] = {
            "usd": msg.data_path,
            "position": position,
            "quatWXYZ": quatWXYZ,
            "track_id": track_id,
        }
        ob = ObjectInBufferMsg()
        ob.header.stamp = self.get_clock().now().to_msg()
        ob.header.frame_id = "odom"
        ob.job_id = str(track_id)
        self._pub_in_buffer.publish(ob)
        self.get_logger().info(
            "Buffer: pose track_id=%s → buffer_size=%d" % (track_id, len(self._buffer))
        )
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
        """Build scene list from buffer (same shape as usd_buffer_node: usd, position, quatWXYZ)."""
        return [
            {
                "id": obj_id,
                "track_id": data["track_id"],
                "usd": data["usd"],
                "position": data["position"],
                "quatWXYZ": data["quatWXYZ"],
            }
            for obj_id, data in self._buffer.items()
        ]

    def _write_scene_json_only(self):
        """Write scene_graph.json and optionally scene.usda (sync). USDA includes OBB per object."""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        scene_list = self._build_scene_list()
        json_path = self._output_dir / self._scene_json_name
        try:
            with open(json_path, "w") as f:
                json.dump({"objects": scene_list}, f, indent=2)
            self.get_logger().info("Wrote %s (%d objects)" % (json_path, len(scene_list)))
        except Exception as e:
            self.get_logger().warn("Failed to write %s: %s" % (json_path, e))
        if self._write_usda and scene_list:
            usda_path = self._output_dir / self._scene_usda_name
            try:
                _write_usda_scene(usda_path, scene_list, log=self.get_logger())
            except Exception as e:
                self.get_logger().warn("Failed to write USDA %s: %s" % (usda_path, e))

    def _write_scene(self):
        """Timer: write JSON first, then GLB in background if enabled."""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._write_scene_json_only()
        if not self._write_glb:
            return
        with self._glb_write_lock:
            if self._glb_write_in_progress:
                self.get_logger().debug("GLB write still in progress; skipping this timer tick")
                return
            self._glb_write_in_progress = True
        scene_list = self._build_scene_list()
        buffer_snapshot = [(k, dict(d)) for k, d in self._buffer.items()]
        self.get_logger().info("Starting GLB write: buffer=%d entries" % len(self._buffer))
        t = threading.Thread(target=self._write_scene_io, args=(scene_list, buffer_snapshot))
        t.daemon = True
        t.start()

    def _write_scene_io(self, scene_list: list, buffer_snapshot: list):
        """Background thread: write joint GLB from scene_list. USDA is written in _write_scene_json_only."""
        try:
            if self._write_glb:
                glb_path = self._output_dir / self._scene_glb_name
                try:
                    self._write_joint_glb_from_list(glb_path, scene_list)
                except Exception as e:
                    self.get_logger().warn("Failed to write joint GLB %s: %s" % (glb_path, e))
        finally:
            with self._glb_write_lock:
                self._glb_write_in_progress = False

    def _write_joint_glb_from_list(self, glb_path: Path, scene_list: list):
        """Build and write joint GLB. position + quatWXYZ (same as usd_buffer_node). Supports .glb/.ply; .usd skipped."""
        try:
            import trimesh
        except ImportError:
            self.get_logger().warn("trimesh not available; skipping joint GLB")
            return
        scene = trimesh.Scene()
        added = 0
        for item in scene_list:
            path = item["usd"]
            pos = np.array(item["position"], dtype=np.float64)
            q = item["quatWXYZ"]
            quat_xyzw = [q[1], q[2], q[3], q[0]]
            T = _pose_to_matrix(pos, quat_xyzw)
            oid = item.get("id", item.get("track_id", added))
            meshes = _load_meshes_from_path(path)
            if not meshes:
                self.get_logger().debug("Skip GLB for %s (not .glb/.ply or load failed)" % path)
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
