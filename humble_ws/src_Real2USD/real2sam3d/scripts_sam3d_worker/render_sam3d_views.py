#!/usr/bin/env python3
"""
Render multi-view images from SAM3D output (object.glb or object.ply) for better FAISS
retrieval. Renders with appearance (vertex colors / materials) when available so
retrieval matches on look, not just shape. Creates output/<job_id>/views/0.png .. N-1.png.
The indexer uses these when present instead of a single rgb.png.

Prefer object.glb when present (mesh + materials/colors); fall back to object.ply.
Requires: open3d (pip install open3d). For GLB: trimesh (pip install trimesh).
Optional: pyvista (pip install pyvista) for correct vertex-color rendering when Open3D
shows grey; with pyvista installed, colored meshes are rendered with PyVista.
To reduce libGL/nouveau errors when running headless: LIBGL_ALWAYS_SOFTWARE=1 or xvfb-run.
Object is in object-local frame (centroid at origin); we orbit the camera about the Y axis
(Y-up frame: glTF/Blender convention) so views are consistent with the mesh's vertical.

Usage:
  python render_sam3d_views.py --job-dir /data/sam3d_queue/output/<job_id>
  python render_sam3d_views.py --queue-dir /data/sam3d_queue [--once] [--watch-interval 10]
  python render_sam3d_views.py --use-current-run [--once]   # uses queue_dir from current_run.json

--job-dir: Render views for a single output job directory.
--queue-dir: Queue/run dir; scan output/ for jobs without views/. Ignored if --use-current-run.
--use-current-run: Use queue_dir from current_run.json (written by ros2 launch).
--num-views: Number of azimuth views (default 8).
--size: Image width and height (default 256).
"""

import argparse
import sys
import time
from pathlib import Path


def _mesh_colors(m) -> tuple:
    """Get per-vertex RGB for a trimesh (N,3) in [0,1]. Returns (colors_array or None, has_color)."""
    import numpy as np
    n = len(m.vertices)
    if n == 0:
        return None, False
    # Prefer vertex colors
    if hasattr(m.visual, "vertex_colors") and m.visual.vertex_colors is not None:
        cols = np.asarray(m.visual.vertex_colors, dtype=np.float64)
        if cols.ndim == 2 and cols.shape[0] == n and cols.shape[1] >= 3:
            cols = cols[:, :3]
            if cols.max() > 1.0:
                cols = cols / 255.0
            return np.clip(cols, 0, 1), True
    # Else material baseColorFactor (one color for whole mesh)
    if hasattr(m.visual, "material") and m.visual.material is not None:
        factor = getattr(m.visual.material, "baseColorFactor", None)
        if factor is not None and len(factor) >= 3:
            c = np.array(factor[:3], dtype=np.float64)
            return np.tile(np.clip(c, 0, 1), (n, 1)), True
    return None, False


def _load_glb(glb_path: Path):
    """Load GLB with trimesh; return Open3D TriangleMesh with vertex colors from each submesh/material, centered."""
    import numpy as np
    import open3d as o3d
    try:
        import trimesh
    except ImportError as e:
        print(f"[WARN] Cannot load GLB: trimesh not installed. pip install trimesh", file=sys.stderr)
        return None, False
    try:
        # Load as scene so we keep per-mesh materials (do not force='mesh' or we lose them)
        scene = trimesh.load(str(glb_path), process=False)
        if isinstance(scene, trimesh.Scene):
            meshes = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
        elif isinstance(scene, trimesh.Trimesh):
            meshes = [scene]
        else:
            print(f"[WARN] GLB load returned {type(scene).__name__}, expected Scene or Trimesh", file=sys.stderr)
            return None, False
        if not meshes:
            print(f"[WARN] GLB has no mesh geometry", file=sys.stderr)
            return None, False

        all_verts = []
        all_faces = []
        all_colors = []
        any_color = False
        vertex_offset = 0

        for m in meshes:
            verts = np.asarray(m.vertices, dtype=np.float64)
            faces = np.asarray(m.faces, dtype=np.int32)
            if len(verts) == 0 or len(faces) == 0:
                continue
            cols, has_c = _mesh_colors(m)
            if cols is None:
                cols = np.full((len(verts), 3), 0.7, dtype=np.float64)  # default grey
            else:
                any_color = True
            all_verts.append(verts)
            all_faces.append(faces + vertex_offset)
            all_colors.append(cols)
            vertex_offset += len(verts)

        if not all_verts:
            print(f"[WARN] GLB meshes have no vertices/faces", file=sys.stderr)
            return None, False

        verts = np.vstack(all_verts)
        faces = np.vstack(all_faces)
        colors = np.vstack(all_colors)

        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(verts),
            o3d.utility.Vector3iVector(faces),
        )
        mesh.compute_vertex_normals()
        # Ensure [0,1] float64 contiguous so Open3D renderer uses them with defaultUnlit
        colors_f = np.asarray(colors, dtype=np.float64, order="C")
        if colors_f.shape[0] != len(verts) or colors_f.shape[1] < 3:
            colors_f = np.full((len(verts), 3), 0.7, dtype=np.float64)
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors_f[:, :3])
        mesh.translate(-mesh.get_center())
        # We always set vertex_colors (from materials or default grey), so use unlit shader to show them
        return mesh, True
    except Exception as e:
        print(f"[WARN] _load_glb failed: {e}", file=sys.stderr)
        return None, False


def _load_ply_with_plyfile(ply_path: Path):
    """Fallback: load PLY with plyfile when Open3D fails (e.g. non-standard or gaussian-splat format). Returns (points Nx3, colors Nx3 or None)."""
    try:
        from plyfile import PlyData
    except ImportError:
        return None, None
    try:
        ply = PlyData.read(str(ply_path))
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
            # Drop rows with nan/inf so Open3D doesn't choke
            valid = np.isfinite(pts).all(axis=1)
            if not np.any(valid):
                return None, None
            pts = pts[valid]
            cols = None
            if "red" in names and "green" in names and "blue" in names:
                r = np.asarray(el.data["red"], dtype=np.float64)[valid]
                g = np.asarray(el.data["green"], dtype=np.float64)[valid]
                b = np.asarray(el.data["blue"], dtype=np.float64)[valid]
                cols = np.stack([r, g, b], axis=1)
                if cols.max() > 1.0:
                    cols = cols / 255.0
            return pts, cols
    except Exception:
        return None, None
    return None, None


def _load_ply(ply_path: Path):
    """Load PLY as mesh or point cloud; return (Open3D geometry, use_vertex_colors). Uses plyfile fallback when Open3D fails."""
    import numpy as np
    import open3d as o3d

    # Try Open3D triangle mesh first
    try:
        mesh = o3d.io.read_triangle_mesh(str(ply_path))
        if mesh.has_triangles() and len(mesh.triangles) > 0:
            mesh.compute_vertex_normals()
            use_color = mesh.has_vertex_colors()
            mesh.translate(-mesh.get_center())
            return mesh, use_color
    except Exception:
        pass

    # Try Open3D point cloud
    try:
        pcd = o3d.io.read_point_cloud(str(ply_path))
        if pcd.has_points() and len(pcd.points) > 0:
            hull, _ = pcd.compute_convex_hull()
            hull.compute_vertex_normals()
            use_color = False
            if pcd.has_colors() and len(pcd.colors) > 0:
                try:
                    from scipy.spatial import cKDTree
                    pts = np.asarray(pcd.points)
                    cols = np.asarray(pcd.colors)
                    if cols.max() > 1.0:
                        cols = cols.astype(np.float64) / 255.0
                    hull_pts = np.asarray(hull.vertices)
                    tree = cKDTree(pts)
                    _, idx = tree.query(hull_pts, k=1)
                    hull_colors = cols[np.clip(idx, 0, len(cols) - 1), :3]
                    hull.vertex_colors = o3d.utility.Vector3dVector(hull_colors.astype(np.float64))
                    use_color = True
                except Exception:
                    pass
            hull.translate(-hull.get_center())
            return hull, use_color
    except Exception:
        pass

    # Fallback: plyfile (handles SAM3D / gaussian-splat PLYs that break Open3D)
    pts, cols = _load_ply_with_plyfile(ply_path)
    if pts is None or len(pts) < 4:
        return None, False
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if cols is not None:
        pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
    try:
        hull, _ = pcd.compute_convex_hull()
        hull.compute_vertex_normals()
        use_color = False
        if cols is not None:
            try:
                from scipy.spatial import cKDTree
                hull_pts = np.asarray(hull.vertices)
                tree = cKDTree(pts)
                _, idx = tree.query(hull_pts, k=1)
                hull_colors = cols[np.clip(idx, 0, len(cols) - 1), :3]
                hull.vertex_colors = o3d.utility.Vector3dVector(hull_colors.astype(np.float64))
                use_color = True
            except Exception:
                pass
        hull.translate(-hull.get_center())
        return hull, use_color
    except Exception:
        return None, False


def _load_geometry(job_dir: Path):
    """Prefer object.glb, then object.ply. Return (geometry, use_vertex_colors) or (None, False)."""
    glb_path = job_dir / "object.glb"
    ply_path = job_dir / "object.ply"
    if glb_path.exists():
        geom, use_color = _load_glb(glb_path)
        if geom is not None:
            print(f"[load] object.glb (vertex_colors={use_color})", file=sys.stderr)
            return geom, use_color
        print(f"[WARN] Failed to load object.glb, trying object.ply", file=sys.stderr)
    if ply_path.exists():
        geom, use_color = _load_ply(ply_path)
        if geom is not None:
            print(f"[load] object.ply (vertex_colors={use_color})", file=sys.stderr)
            return geom, use_color
    return None, False


def _render_views_pyvista(verts, faces, colors, num_views: int, size: int, out_dir: Path) -> int:
    """Render with PyVista (reliable vertex colors). verts/faces/colors are numpy; colors in [0,1] RGB."""
    import os
    # Reduce libGL/nouveau "failed to load driver" noise (set before VTK/PyVista use). Override with LIBGL_ALWAYS_SOFTWARE=0 if needed.
    if os.environ.get("LIBGL_ALWAYS_SOFTWARE") is None:
        os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
    import numpy as np
    try:
        import pyvista as pv
    except ImportError:
        return 0
    # VTK face format: [n_pts, i0, i1, i2, n_pts, ...]
    n_face = np.full((len(faces), 1), 3, dtype=np.int32)
    faces_vtk = np.hstack([n_face, faces]).ravel().astype(np.int32)
    mesh = pv.PolyData(verts, faces_vtk)
    # PyVista RGB: 0-255 uint8 for rgb=True, or 0-1 float; use 0-255 for clarity
    rgb = (np.clip(colors[:, :3], 0, 1) * 255).astype(np.uint8)
    mesh["colors"] = rgb

    # Camera distance: keep full object in frame (match Open3D logic: ~1.2–1.5 * max extent)
    max_extent = float(np.max(np.ptp(verts, axis=0)))
    if not np.isfinite(max_extent) or max_extent < 1e-6:
        return 0
    r = max(0.5, min(max_extent * 1.5, 1e4))
    elevation_deg = 25.0
    el = np.radians(elevation_deg)
    center = [0.0, 0.0, 0.0]
    # Y-up frame: orbit about Y (camera moves in XZ plane; up = Y)
    up = [0.0, 1.0, 0.0]

    written = 0
    for i in range(num_views):
        az = 2.0 * np.pi * i / num_views
        # Orbit in XZ plane; elevation adds Y component
        x = r * np.cos(az) * np.cos(el)
        y = r * np.sin(el)
        z = r * np.sin(az) * np.cos(el)
        eye = [x, y, z]
        pl = pv.Plotter(off_screen=True, window_size=[size, size])
        pl.set_background([1.0, 1.0, 1.0])
        pl.add_mesh(mesh, scalars="colors", rgb=True, lighting=False, show_scalar_bar=False)
        pl.camera_position = [tuple(eye), tuple(center), tuple(up)]
        pl.camera.view_angle = 50.0
        out_path = out_dir / f"{i}.png"
        img = pl.screenshot(return_img=True)
        pl.close()
        if img is not None and img.size > 0:
            try:
                import cv2
                if img.ndim == 3 and img.shape[2] >= 3:
                    cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    written += 1
            except Exception:
                try:
                    import imageio
                    imageio.v2.imwrite(str(out_path), img)
                    written += 1
                except Exception:
                    pass
    return written


def _render_views(geometry, num_views: int, size: int, out_dir: Path, use_vertex_colors: bool) -> int:
    """Render num_views images (camera orbit). When use_vertex_colors, try PyVista first for correct colors."""
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    bbox = geometry.get_axis_aligned_bounding_box()
    extent = np.array(bbox.get_extent())
    max_extent = float(np.max(extent))
    if not np.isfinite(max_extent) or max_extent < 1e-6:
        return 0
    r = max(0.5, min(max_extent * 1.2, 1e4))

    # Prefer PyVista whenever mesh has vertex colors (Open3D 0.19 often renders them grey)
    if geometry.has_vertex_colors():
        verts = np.asarray(geometry.vertices)
        faces = np.asarray(geometry.triangles)
        colors = np.asarray(geometry.vertex_colors)
        if verts.shape[0] == colors.shape[0] and faces.size > 0:
            n = _render_views_pyvista(verts, faces, colors, num_views, size, out_dir)
            if n > 0:
                print("[OK] Rendered with PyVista (vertex colors)", file=sys.stderr)
                return n
            print("[WARN] PyVista failed or not installed; using Open3D (colors may appear grey). pip install pyvista", file=sys.stderr)

    import open3d as o3d
    import open3d.visualization.rendering as rendering

    elevation_deg = 25.0
    el = np.radians(elevation_deg)
    try:
        renderer = rendering.OffscreenRenderer(size, size)
    except Exception as e:
        print(f"[WARN] OffscreenRenderer failed: {e}", file=sys.stderr)
        return 0
    mat = rendering.MaterialRecord()
    if use_vertex_colors:
        renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
        try:
            renderer.scene.view.set_post_processing(False)
        except Exception:
            pass
        mat.shader = "defaultUnlit"
    else:
        mat.base_color = [0.85, 0.85, 0.85, 1.0]
        mat.shader = "defaultLit"
        renderer.scene.scene.set_sun_light([0.577, 0.577, 0.577], [1.0, 1.0, 1.0], 75000)
        renderer.scene.scene.enable_sun_light(True)

    try:
        renderer.scene.add_geometry("obj", geometry, mat, True)
    except Exception as e:
        print(f"[WARN] add_geometry failed (e.g. empty AABB): {e}", file=sys.stderr)
        return 0

    center = [0.0, 0.0, 0.0]
    # Y-up frame: orbit about Y (camera moves in XZ plane; up = Y)
    up = [0.0, 1.0, 0.0]
    fov_deg = 50.0

    written = 0
    for i in range(num_views):
        az = 2.0 * np.pi * i / num_views
        # Orbit in XZ plane; elevation adds Y component
        x = r * np.cos(az) * np.cos(el)
        y = r * np.sin(el)
        z = r * np.sin(az) * np.cos(el)
        eye = [float(x), float(y), float(z)]
        try:
            renderer.setup_camera(fov_deg, center, eye, up)
            img = renderer.render_to_image()
            out_path = out_dir / f"{i}.png"
            o3d.io.write_image(str(out_path), img, 9)
            written += 1
        except Exception as e:
            print(f"[WARN] render view {i} failed: {e}", file=sys.stderr)
    return written


def render_job_dir(job_dir: Path, num_views: int = 8, size: int = 256) -> bool:
    """Render multi-view images into job_dir/views/. Prefers object.glb then object.ply. Returns True if views were written."""
    job_dir = Path(job_dir).resolve()
    views_dir = job_dir / "views"

    if views_dir.exists() and next(views_dir.glob("*.png"), None) is not None:
        return False  # already has views

    if not (job_dir / "object.glb").exists() and not (job_dir / "object.ply").exists():
        return False

    geometry, use_vertex_colors = _load_geometry(job_dir)
    if geometry is None:
        print(f"[WARN] Could not load object.glb or object.ply from {job_dir}", file=sys.stderr)
        return False

    n = _render_views(geometry, num_views, size, views_dir, use_vertex_colors)
    print(f"[OK] Rendered {n} views to {views_dir} ({job_dir.name}, colors={use_vertex_colors})", file=sys.stderr)
    return n > 0


def main():
    parser = argparse.ArgumentParser(description="Render multi-view images from SAM3D object.ply")
    parser.add_argument("--job-dir", type=str, default=None, help="Single output job directory to render")
    parser.add_argument("--queue-dir", type=str, default=None, help="Queue/run dir (used only with --no-current-run)")
    parser.add_argument("--no-current-run", action="store_true", help="Do not use current_run.json; use --queue-dir explicitly")
    parser.add_argument("--num-views", type=int, default=8, help="Number of azimuth views")
    parser.add_argument("--size", type=int, default=256, help="View image size (width and height)")
    parser.add_argument("--watch-interval", type=float, default=10.0, help="Seconds between scans when watching")
    parser.add_argument("--once", action="store_true", help="Process once and exit (with --queue-dir or --use-current-run)")
    args = parser.parse_args()

    try:
        import open3d as o3d  # noqa: F401
    except ImportError:
        print("Error: open3d is required. Install with: pip install open3d", file=sys.stderr)
        sys.exit(1)

    if args.job_dir:
        job_dir = Path(args.job_dir).resolve()
        if not job_dir.is_dir():
            print(f"Error: Not a directory: {job_dir}", file=sys.stderr)
            sys.exit(1)
        render_job_dir(job_dir, num_views=args.num_views, size=args.size)
        return

    use_current_run = not args.no_current_run
    if use_current_run:
        try:
            from current_run import resolve_queue_and_index
            queue_dir, _ = resolve_queue_and_index(True, args.queue_dir or "/data/sam3d_queue", None)
        except ImportError:
            if not args.queue_dir:
                print("Error: current_run module not found. Use --no-current-run and --queue-dir.", file=sys.stderr)
                sys.exit(1)
            queue_dir = Path(args.queue_dir).resolve()
    elif args.queue_dir:
        queue_dir = Path(args.queue_dir).resolve()
    else:
        print("Error: Provide --job-dir or --queue-dir (or run without --no-current-run to use current_run.json).", file=sys.stderr)
        sys.exit(1)
    output_dir = queue_dir / "output"
    if not output_dir.exists():
        print(f"Output dir does not exist: {output_dir}", file=sys.stderr)
        sys.exit(1)

    def scan_and_render():
        count = 0
        for job_path in output_dir.iterdir():
            if not job_path.is_dir():
                continue
            if (job_path / "object.glb").exists() or (job_path / "object.ply").exists() or (job_path / "object.usd").exists():
                if render_job_dir(job_path, num_views=args.num_views, size=args.size):
                    count += 1
        return count

    if args.once:
        n = scan_and_render()
        print(f"Rendered views for {n} job(s).", file=sys.stderr)
        return

    print(f"Watching {output_dir} every {args.watch_interval}s. Ctrl+C to stop.", file=sys.stderr)
    while True:
        try:
            scan_and_render()
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[ERROR] {e}", file=sys.stderr)
        time.sleep(args.watch_interval)


if __name__ == "__main__":
    main()
