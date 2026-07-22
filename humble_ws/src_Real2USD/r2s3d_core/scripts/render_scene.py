"""Scene-level render of the placed SAM3D assets (C1 / teaser).

Assembles all objects of one scene into the world frame from a run's
scene_graph.json (each object's mesh @ T_world_mesh), preserving SAM 3D's
per-vertex COLOUR, and (a) renders a shaded PNG via Open3D's headless
offscreen renderer and (b) exports a combined colour GLB you can spin in any
glTF viewer (Blender, VS Code glTF, gltf-viewer).

Use it to compare scene reconstructions across registration modes:
  layout  = SAM 3D-native placement   (results/paper/sim/asset_layout_gt_s200)
  +ICP    = our registration          (results/paper/sim/asset_icp_gt_s200)
  GT      = ground-truth meshes        (--gt)

Usage (headless needs the GPU Xorg display):
  DISPLAY=:0 XAUTHORITY=/run/user/1003/gdm/Xauthority \
  uv run --extra procthor --extra mesh --extra registration --extra viz \
    python scripts/render_scene.py --scene 200 \
      --run results/paper/sim/asset_icp_gt_s200 --tag icp
  # ...--run .../asset_layout_gt_s200 --tag layout ; and --gt --tag gt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from r2s3d_core.data.registry import make_source

# ---- render knobs (edit here) ---------------------------------------------
W, H = 1500, 1050            # render resolution (px)
FOV = 55.0                   # vertical field of view (deg)
ELEV, AZIM = 38.0, 45.0      # camera elevation / azimuth (deg)
DIST_MULT = 1.05             # eye distance = DIST_MULT * scene bbox diagonal (lower = zoom in)
BG = [1, 1, 1, 1]            # background (white)
SUN_DIR = [-0.4, -0.5, -0.9]
SUN_INT = 60000.0
# ---------------------------------------------------------------------------


def _posed_meshes_from_run(run: Path):
    sg = json.load(open(next(run.glob("*/scene_graph.json"))))
    queue = Path(sg.get("sam3d_queue") or (run / "sam3d_queue"))
    out = []
    for o in sg["objects"]:
        if not o.get("mesh") or not (queue / o["mesh"]).exists():
            continue
        m = trimesh.load(str(queue / o["mesh"]), force="mesh")
        m.apply_transform(np.asarray(o["T_world_mesh"], float))
        out.append(m)
    return out


def _gt_meshes(scene, split):
    src = make_source("procthor", scene, split=split, gt_mesh="asset", stride=1)
    list(src)
    return [g.mesh.copy() for g in src.gt() if getattr(g, "mesh", None) is not None]


def _autocrop(path, pad=8, also_pdf=False):
    from PIL import Image, ImageChops
    im = Image.open(path).convert("RGB")
    bgcol = im.getpixel((0, 0))  # sample actual background (Open3D bg is light grey, not pure white)
    diff = ImageChops.difference(im, Image.new("RGB", im.size, bgcol))
    bb = diff.convert("L").point(lambda p: 255 if p > 12 else 0).getbbox()
    if bb:
        l, t, r, b = bb
        im = im.crop((max(0, l - pad), max(0, t - pad),
                      min(im.width, r + pad), min(im.height, b + pad)))
        im.save(path)
    if also_pdf:
        im.save(str(Path(path).with_suffix(".pdf")), "PDF", resolution=200.0)


def _to_o3d(m):
    import open3d as o3d
    om = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(m.vertices, float)),
        o3d.utility.Vector3iVector(np.asarray(m.faces, np.int32)),
    )
    vc = getattr(m.visual, "vertex_colors", None)
    if vc is not None and len(vc):
        om.vertex_colors = o3d.utility.Vector3dVector(np.asarray(vc)[:, :3] / 255.0)
    om.compute_vertex_normals()
    return om


def _cluster_clouds(scene, split, detections):
    """Re-run the tracker and return the CLUSTER-payload point cloud per mature
    track (denoised fused cloud), exactly as object_track's cluster baseline builds it."""
    from r2s3d_core.detect.cache import DetectionSet
    from r2s3d_core.tracks import run_tracker, TrackState
    from r2s3d_core.baselines import sam3d_layout as s3d

    src = make_source("procthor", scene, split=split, gt_mesh="asset", stride=1)
    frames = list(src)
    ds = DetectionSet.load(detections, scene=scene)
    tracks = run_tracker(frames, ds.by_frame(), {"reid": True, "late_merge": True})
    clouds = []
    for t in tracks:
        if t.state != TrackState.MATURE:
            continue
        cloud = s3d._denoise_cloud(np.asarray(t.fused_cloud, np.float64))
        if len(cloud) >= 4:
            clouds.append(cloud)
    return clouds


def _render_cluster(args) -> None:
    """Top-down render of the cluster baseline: each object a distinctly-coloured
    point cluster (the 'labelled point cloud' node payload), no meshes."""
    import colorsys
    import open3d as o3d
    from r2s3d_core.baselines import sam3d_layout as s3d

    clouds = _cluster_clouds(args.scene, args.split, args.detections)
    print(f"cluster: {len(clouds)} objects, {sum(len(c) for c in clouds)} points total")

    # distinct colour per cluster (evenly-spaced hues) -> 'labelled clusters' look
    pts, cols = [], []
    for i, c in enumerate(clouds):
        rgb = colorsys.hsv_to_rgb((0.61 * i) % 1.0, 0.65, 0.95)
        pts.append(c); cols.append(np.tile(rgb, (len(c), 1)))
    pts, cols = np.vstack(pts), np.vstack(cols)

    r = o3d.visualization.rendering.OffscreenRenderer(W, H)
    r.scene.set_background(BG)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd.colors = o3d.utility.Vector3dVector(cols)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = 4.0
    r.scene.add_geometry("cluster", pcd, mat)

    # per-cluster OBB wireframe (the SAME box the cluster metric uses, s3d._cloud_obb)
    # -> makes the noise-inflated / mis-oriented boxes explicit.
    lmat = o3d.visualization.rendering.MaterialRecord()
    lmat.shader = "unlitLine"
    lmat.line_width = 2.0
    for i, c in enumerate(clouds):
        try:
            T, ext = s3d._cloud_obb(c)
        except Exception:
            continue
        obb = o3d.geometry.OrientedBoundingBox(T[:3, 3], T[:3, :3], np.asarray(ext, float))
        ls = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
        ls.paint_uniform_color([0.12, 0.12, 0.12])
        r.scene.add_geometry(f"box{i}", ls, lmat)

    center = np.asarray(args.center, float) if args.center else pts.mean(0)
    bmin, bmax = pts.min(0), pts.max(0)
    if args.topdown:
        fov, up = args.topdown_fov, [0, 1, 0]
        if args.eye:                       # shared eye -> identical scale across panels
            eye = np.asarray(args.eye, float)
        else:
            ext = bmax - bmin
            cam_h = float(np.max(ext[:2])) / (2 * np.tan(np.radians(fov / 2))) * 1.15
            eye = np.array([center[0], center[1], center[2] + cam_h])
    else:
        eye = np.asarray(args.eye, float) if args.eye else center + np.array([1., 1., 1.]) * \
            float(np.linalg.norm(bmax - bmin)) * DIST_MULT
        up, fov = [0, 0, 1], FOV
    r.setup_camera(fov, center.tolist(), eye.tolist(), up)

    out = f"{Path(args.out_dir)}/scene_s{args.scene}_cluster{'_top' if args.topdown else ''}.png"
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    o3d.io.write_image(out, r.render_to_image())
    _autocrop(out, also_pdf=True) if not args.no_crop else None
    print(f"wrote {out} (+ .pdf)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="200")
    ap.add_argument("--split", default="val")
    ap.add_argument("--run", default=None, help="run dir with */scene_graph.json")
    ap.add_argument("--gt", action="store_true", help="render GT meshes instead of a run")
    ap.add_argument("--cluster", action="store_true",
                    help="render the CLUSTER-payload baseline: per-object denoised fused "
                         "point clouds (labelled clusters), not meshes")
    ap.add_argument("--detections", default="results/detections/procthor/gt",
                    help="detections dir for --cluster (parent of <scene>/)")
    ap.add_argument("--tag", required=True, help="output suffix (layout|icp|gt|...)")
    ap.add_argument("--elev", type=float, default=ELEV)
    ap.add_argument("--azim", type=float, default=AZIM)
    ap.add_argument("--center", type=float, nargs=3, default=None,
                    help="explicit look-at centre (world x y z) to SHARE a camera across renders")
    ap.add_argument("--eye", type=float, nargs=3, default=None,
                    help="explicit camera eye (world x y z) to SHARE a camera across renders")
    ap.add_argument("--topdown", action="store_true",
                    help="orthographic-ish overhead view (world +Y up in image) to match "
                         "the AI2-THOR map-view RGB")
    ap.add_argument("--topdown-fov", type=float, default=40.0)
    ap.add_argument("--no-crop", action="store_true",
                    help="skip autocrop -> full canvas, identical framing across renders "
                         "(use with a shared --center/--eye for pixel-aligned comparison panels)")
    ap.add_argument("--out-dir", default="results/paper/_figs")
    args = ap.parse_args()

    if args.cluster:
        _render_cluster(args)
        return

    meshes = _gt_meshes(args.scene, args.split) if args.gt \
        else _posed_meshes_from_run(Path(args.run))
    meshes = [m for m in meshes if len(getattr(m, "vertices", []))]
    print(f"{args.tag}: {len(meshes)} objects, "
          f"{sum(len(m.vertices) for m in meshes)} verts total")

    base = Path(args.out_dir) / f"scene_s{args.scene}_{args.tag}"
    base.parent.mkdir(parents=True, exist_ok=True)

    # (b) combined colour GLB for interactive viewing
    combined = trimesh.util.concatenate(meshes)
    combined.export(f"{base}.glb")
    print(f"wrote {base}.glb")

    # (a) shaded PNG via Open3D headless offscreen
    import open3d as o3d
    r = o3d.visualization.rendering.OffscreenRenderer(W, H)
    r.scene.set_background(BG)
    r.scene.scene.set_sun_light(SUN_DIR, [1, 1, 1], SUN_INT)
    r.scene.scene.enable_sun_light(True)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.base_color = [1, 1, 1, 1]
    for i, m in enumerate(meshes):
        r.scene.add_geometry(f"obj{i}", _to_o3d(m), mat)

    b = combined.bounds
    center = np.asarray(args.center, float) if args.center else b.mean(0)
    fov, up = FOV, [0, 0, 1]
    if args.topdown:
        fov = args.topdown_fov
        up = [0, 1, 0]  # world +Y up in image, matching AI2-THOR map view
        if args.eye:                       # shared eye -> identical scale across panels
            eye = np.asarray(args.eye, float)
        else:
            ext = b[1] - b[0]
            cam_h = float(np.max(ext[:2])) / (2 * np.tan(np.radians(fov / 2))) * 1.15
            eye = np.array([center[0], center[1], center[2] + cam_h])
    elif args.eye:
        eye = np.asarray(args.eye, float)
    else:
        diag = float(np.linalg.norm(b[1] - b[0]))
        az, el = np.radians(args.azim), np.radians(args.elev)
        d = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        eye = center + d * diag * DIST_MULT
    print(f"camera center={center.round(3).tolist()} eye={eye.round(3).tolist()} up={up}")
    r.setup_camera(fov, center.tolist(), eye.tolist(), up)

    img = r.render_to_image()
    out = f"{base}.png"
    o3d.io.write_image(out, img)
    if not args.no_crop:
        _autocrop(out, also_pdf=True)
    print(f"wrote {out}{'' if args.no_crop else ' (+ .pdf)'}")


if __name__ == "__main__":
    main()
