"""Figure-1 teaser: 3D pipeline panels via Open3D headless, wrapped as Type-42 PDFs.

Featured object = lounge-0 track 0 (the armchair, best frame 52):
  5_generated     SAM 3D generated mesh (canonical), textured
  6_registration  posed mesh (T_world_mesh) overlaid on the sensor depth cloud
                  backprojected from frame 52 -> shows metric alignment

Needs the GPU display for Open3D EGL:
  DISPLAY=:0 XAUTHORITY=/run/user/1003/gdm/Xauthority \
  uv run --extra rosbag python scripts/figs/fig1_panels_3d.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import trimesh
import open3d as o3d

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from PIL import Image, ImageChops

from r2s3d_core.data.registry import make_source

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/paper/_figs/fig1"
OUT.mkdir(parents=True, exist_ok=True)

JOB = ROOT / "results/phase0_lounge0_rs_scaleicp/sam3d_queue/output/realsense_lounge-0_t26_full"
SG = ROOT / "results/phase0_lounge0_rs_icp/lounge-0/scene_graph.json"
SCENE, HERO_FRAME, TRACK_JOB = "lounge-0", 360, "realsense_lounge-0_t26_full"
W, H = 1400, 1400
BG = [1, 1, 1, 1]
DPI = 300


def _to_o3d(m, default_col=(0.75, 0.75, 0.78)):
    om = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(m.vertices, float)),
        o3d.utility.Vector3iVector(np.asarray(m.faces, np.int32)))
    vc = getattr(m.visual, "vertex_colors", None)
    if vc is not None and len(vc):
        om.vertex_colors = o3d.utility.Vector3dVector(np.asarray(vc)[:, :3] / 255.0)
    else:
        om.paint_uniform_color(default_col)
    om.compute_vertex_normals()
    return om


def _autocrop_to_pdf(png_path, pdf_name, pad=10):
    im = Image.open(png_path).convert("RGB")
    bg = im.getpixel((0, 0))
    diff = ImageChops.difference(im, Image.new("RGB", im.size, bg))
    bb = diff.convert("L").point(lambda p: 255 if p > 12 else 0).getbbox()
    if bb:
        l, t, r, b = bb
        im = im.crop((max(0, l - pad), max(0, t - pad),
                      min(im.width, r + pad), min(im.height, b + pad)))
    w, h = im.size
    fig = plt.figure(figsize=(w / DPI, h / DPI), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.imshow(im)
    p = OUT / pdf_name
    fig.savefig(p, dpi=DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print("wrote", p)


def _renderer():
    r = o3d.visualization.rendering.OffscreenRenderer(W, H)
    r.scene.set_background(BG)
    r.scene.scene.set_sun_light([-0.4, -0.5, -0.9], [1, 1, 1], 60000.0)
    r.scene.scene.enable_sun_light(True)
    return r


def _cam(r, center, eye, up=(0, 0, 1), fov=50.0):
    r.setup_camera(fov, list(center), list(eye), list(up))


def _angled_eye(center, diag, elev=28.0, azim=50.0, mult=1.5):
    az, el = np.radians(azim), np.radians(elev)
    d = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    return np.asarray(center) + d * diag * mult


def panel_generated():
    m = trimesh.load(str(JOB / "object.glb"), force="mesh", process=False)
    # centre + upright: SAM3D raw mesh -> use bounds centre, render textured
    b = m.bounds; center = b.mean(0); diag = float(np.linalg.norm(b[1] - b[0]))
    r = _renderer()
    mat = o3d.visualization.rendering.MaterialRecord(); mat.shader = "defaultLit"
    r.scene.add_geometry("obj", _to_o3d(m), mat)
    _cam(r, center, _angled_eye(center, diag, elev=18, azim=35, mult=1.7), up=(0, 1, 0), fov=45)
    png = str(OUT / "_tmp_generated.png")
    o3d.io.write_image(png, r.render_to_image())
    _autocrop_to_pdf(png, "fig1_5_generated.pdf")
    Path(png).unlink(missing_ok=True)


def _posed_mesh_and_cloud():
    import json
    sg = json.loads(SG.read_text())
    obj = next(o for o in sg["objects"] if o.get("job_id") == TRACK_JOB)
    queue = Path(sg["sam3d_queue"])
    m = trimesh.load(str(queue / obj["mesh"]), force="mesh", process=False)
    m.apply_transform(np.asarray(obj["T_world_mesh"], float))

    # backproject frame-52 depth within the object mask -> world point cloud
    src = make_source(SCENE, "lounge-0") if False else make_source("realsense", SCENE, stride=1)
    f = next(fr for fr in src if fr.frame_id == HERO_FRAME)
    mask = np.array(Image.open(JOB / "mask.png").convert("L")) > 0
    d = f.depth.astype(np.float32)
    K = np.asarray(f.K, float).reshape(3, 3)
    ys, xs = np.where(mask & (d > 0))
    zc = d[ys, xs]
    x = (xs - K[0, 2]) * zc / K[0, 0]
    y = (ys - K[1, 2]) * zc / K[1, 1]
    pts_cam = np.stack([x, y, zc], 1)
    Twc = np.asarray(f.T_world_cam, float)
    pts_w = (Twc[:3, :3] @ pts_cam.T + Twc[:3, 3:4]).T
    # drop mask-leak / far background points so the cloud reads clean on the mesh
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_w))
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return m, np.asarray(pcd.points)


def panel_registration():
    m, pts = _posed_mesh_and_cloud()
    print(f"registration: mesh {len(m.vertices)} verts, depth cloud {len(pts)} pts")
    allpts = np.vstack([np.asarray(m.vertices), pts])
    center = allpts.mean(0)
    diag = float(np.linalg.norm(allpts.max(0) - allpts.min(0)))

    # front-facing: view from the sensor side (frame-52 camera position), near level,
    # so the depth points read face-on on the chair front.
    src = make_source("realsense", SCENE, stride=1)
    campos = np.asarray(next(fr for fr in src if fr.frame_id == HERO_FRAME).T_world_cam,
                        float)[:3, 3]
    dvec = campos - center; dvec[2] = 0.0; dvec /= np.linalg.norm(dvec)
    eye = center + dvec * diag * 1.6 + np.array([0.0, 0.0, 0.15 * diag])

    r = _renderer()
    mmat = o3d.visualization.rendering.MaterialRecord(); mmat.shader = "defaultLit"
    r.scene.add_geometry("mesh", _to_o3d(m, default_col=(0.62, 0.72, 0.85)), mmat)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd.paint_uniform_color([0.90, 0.30, 0.15])          # sensor depth = orange-red
    pmat = o3d.visualization.rendering.MaterialRecord()
    pmat.shader = "defaultUnlit"; pmat.point_size = 5.0
    r.scene.add_geometry("depth", pcd, pmat)
    _cam(r, center, eye, up=(0, 0, 1), fov=50)
    png = str(OUT / "_tmp_registration.png")
    o3d.io.write_image(png, r.render_to_image())
    _autocrop_to_pdf(png, "fig1_6_registration.pdf")
    Path(png).unlink(missing_ok=True)


def main():
    panel_generated()
    panel_registration()


if __name__ == "__main__":
    main()
