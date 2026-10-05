"""Render-and-compare pose+scale refinement -- single-object PROTOTYPE.

Motivation: masks are high quality (IoU ~0.96) while the 3D fused cloud is ~3x
contaminated, yet the back-end registers pose (ICP) and scale (fused) against the
cloud. This prototype instead fits pose + per-axis scale by matching the mesh's
RENDERED silhouette (+depth) to the observed masks across the track's views --
the signal we trust, and inherently multi-view (constrains full orientation).

Scope: proves the OBJECTIVE improves multi-view silhouette IoU over the SAM3D
layout seed on one real object, before wiring into the pipeline. 7 params
[yaw, tx, ty, tz, sx, sy, sz] about the mesh centroid, seeded at layout,
gradient-free (Powell). Not yet integrated, not yet optimized for speed.

  DISPLAY=:0 XAUTHORITY=/run/user/1003/gdm/Xauthority \
  uv run --extra rosbag --extra registration --extra mesh \
    python scripts/figs/render_compare_prototype.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import trimesh
import open3d as o3d
from scipy.optimize import minimize

from r2s3d_core.data.registry import make_source
from r2s3d_core.eval import geometry as geo
from r2s3d_core.eval.geometry import Box
from r2s3d_core.eval.metrics import symmetry_for_label
from r2s3d_core.baselines.sam3d_layout import _fast_obb

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results/phase0_lounge0_rs_layout/lounge-0/scene_graph.json"  # SAM3D-native seed
DET = ROOT / "results/detections/gt/lounge-0/detections.npz"
JOB = "realsense_lounge-0_t26_full"      # the swivel chair
N_VIEWS = 3                              # top views by observed mask area
LAMBDA_DEPTH = 0.5                       # depth term weight (m^-1-ish, Huber)
LAMBDA_SCALE = 0.30                      # L2 regularizer pulling per-axis scale -> SAM3D prior (1.0)
SCALE_LO, SCALE_HI = 0.33, 3.0           # loose safety clamp (data+reg do the real work)
OUT = ROOT / "results/paper/_figs/fig1"


def load_seed_mesh():
    sg = json.loads(RUN.read_text())
    o = next(x for x in sg["objects"] if x["job_id"] == JOB)
    queue = Path(sg["sam3d_queue"])
    m = trimesh.load(str(queue / o["mesh"]), force="mesh", process=False)
    m.apply_transform(np.asarray(o["T_world_mesh"], float))   # -> world, at layout pose+scale
    # decimate for fast silhouette rendering (via open3d; shape preserved enough for a mask)
    om = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(m.vertices, float)),
        o3d.utility.Vector3iVector(np.asarray(m.faces, np.int32)))
    if len(m.faces) > 3000:
        om = om.simplify_quadric_decimation(3000)
    return np.asarray(om.vertices, float), np.asarray(om.triangles, np.int32)


def load_views():
    z = np.load(DET, allow_pickle=True)
    sg = json.loads(RUN.read_text())
    o = next(x for x in sg["objects"] if x["job_id"] == JOB)
    tids = [t for t in o["det_track_ids"] if t >= 0]
    sel = np.where(np.isin(z["track_ids"], tids))[0]
    H, W = z["hw"]
    # observed mask per frame (union of this track's detections in that frame)
    masks = {}
    for i in sel:
        fid = int(z["frame_ids"][i])
        mk = np.unpackbits(z["masks_packed"][i])[: H * W].reshape(H, W).astype(bool)
        masks[fid] = masks.get(fid, np.zeros((H, W), bool)) | mk
    need = set(masks)
    cams = {}
    for f in make_source("realsense", "lounge-0", stride=1):
        if f.frame_id in need:
            cams[f.frame_id] = (np.asarray(f.K, float).reshape(3, 3),
                                np.asarray(f.T_world_cam, float), f.depth.astype(np.float32))
        if len(cams) == len(need):
            break
    rgbs = {}
    for f in make_source("realsense", "lounge-0", stride=1):
        if f.frame_id in need:
            rgbs[f.frame_id] = np.asarray(f.rgb)
        if len(rgbs) == len(need):
            break
    views = []
    for fid, mk in masks.items():
        if fid in cams and mk.sum() > 200:
            K, Twc, depth = cams[fid]
            views.append({"mask": mk, "K": K, "Twc": Twc, "depth": depth,
                          "rgb": rgbs.get(fid), "fid": fid,
                          "area": int(mk.sum()), "H": H, "W": W})
    views.sort(key=lambda v: -v["area"])
    return views[:N_VIEWS]


def _param_transform(p, centroid):
    """7-param delta about the mesh centroid: yaw(Z) + translation + per-axis scale."""
    yaw, tx, ty, tz, sx, sy, sz = p
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    S = np.diag([np.clip(sx, SCALE_LO, SCALE_HI), np.clip(sy, SCALE_LO, SCALE_HI),
                 np.clip(sz, SCALE_LO, SCALE_HI)])
    A = Rz @ S
    T = np.eye(4)
    T[:3, :3] = A
    T[:3, 3] = np.asarray([tx, ty, tz]) + centroid - A @ centroid
    return T


class Renderer:
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.r = o3d.visualization.rendering.OffscreenRenderer(W, H)
        self.r.scene.set_background([0, 0, 0, 0])
        self.mat = o3d.visualization.rendering.MaterialRecord()
        self.mat.shader = "defaultUnlit"

    def silhouette_depth(self, verts, faces, K, Twc):
        self.r.scene.clear_geometry()
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(faces))
        mesh.paint_uniform_color([0.6, 0.6, 0.6])
        self.r.scene.add_geometry("m", mesh, self.mat)
        intr = o3d.camera.PinholeCameraIntrinsic(
            self.W, self.H, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        extr = np.linalg.inv(Twc)          # world -> camera
        self.r.setup_camera(intr, extr)
        d = np.asarray(self.r.render_to_depth_image(z_in_view_space=True))
        sil = np.isfinite(d) & (d > 1e-6)
        return sil, d


def make_loss(views, verts0, faces, centroid, rend):
    def loss(p):
        T = _param_transform(p, centroid)
        vh = (T[:3, :3] @ verts0.T + T[:3, 3:4]).T
        tot, n = 0.0, 0
        for v in views:
            sil, dr = rend.silhouette_depth(vh, faces, v["K"], v["Twc"])
            gm = v["mask"]
            inter = float((sil & gm).sum())
            union = float((sil | gm).sum())
            iou = inter / union if union else 0.0
            term = 1.0 - iou
            # depth term over silhouette∩mask∩valid-obs-depth
            ov = sil & gm & (v["depth"] > 0)
            if ov.sum() > 20:
                dd = np.abs(dr[ov] - v["depth"][ov])
                term += LAMBDA_DEPTH * float(np.mean(np.minimum(dd, 0.3)))  # Huber-ish
            tot += term; n += 1
        # L2 regularizer: penalize per-axis scale drift from the SAM3D prior (factor 1.0).
        # Well-observed axes overcome it via the data term; unconstrained (thin) axes are
        # pulled back to the generated aspect instead of exploding.
        reg = LAMBDA_SCALE * float(np.mean((np.asarray(p[4:7]) - 1.0) ** 2))
        return tot / max(n, 1) + reg
    return loss


def mean_iou(views, verts, faces, rend):
    ious = []
    for v in views:
        sil, _ = rend.silhouette_depth(verts, faces, v["K"], v["Twc"])
        gm = v["mask"]
        u = float((sil | gm).sum())
        ious.append(float((sil & gm).sum()) / u if u else 0.0)
    return ious


def main():
    verts0, faces = load_seed_mesh()
    views = load_views()
    print(f"object {JOB}: {len(verts0)} verts, {len(faces)} faces, {len(views)} views "
          f"(areas {[v['area'] for v in views]})")
    centroid = verts0.mean(0)
    rend = Renderer(views[0]["W"], views[0]["H"])

    iou0 = mean_iou(views, verts0, faces, rend)
    print(f"  layout seed silhouette IoU per view: {[round(x,3) for x in iou0]}  "
          f"mean={np.mean(iou0):.3f}")

    loss = make_loss(views, verts0, faces, centroid, rend)
    p0 = np.array([0, 0, 0, 0, 1, 1, 1], float)
    print(f"  seed loss = {loss(p0):.4f}  -- optimizing (Powell)...")
    res = minimize(loss, p0, method="Powell",
                   options={"maxiter": 60, "xtol": 1e-3, "ftol": 1e-3})
    T = _param_transform(res.x, centroid)
    vh = (T[:3, :3] @ verts0.T + T[:3, 3:4]).T
    iou1 = mean_iou(views, vh, faces, rend)
    yaw = np.degrees(res.x[0]); sc = res.x[4:7]
    print(f"  refined loss = {res.fun:.4f}")
    print(f"  refined silhouette IoU per view: {[round(x,3) for x in iou1]}  "
          f"mean={np.mean(iou1):.3f}")
    print(f"  delta: yaw={yaw:+.1f} deg  trans={np.round(res.x[1:4],3).tolist()} m  "
          f"scale={np.round(sc,3).tolist()}")
    print(f"  >>> mean silhouette IoU {np.mean(iou0):.3f} -> {np.mean(iou1):.3f} "
          f"({'IMPROVED' if np.mean(iou1) > np.mean(iou0) else 'no gain'})")

    # ---- GT validation: does the silhouette win carry over to GT-based metrics? ----
    gts = make_source("realsense", "lounge-0", stride=2).gt() or []
    if not gts:
        print("\n  [GT validation skipped: no GT]"); return
    gt_cent = np.array([np.asarray(g.T_world_obj, float)[:3, 3] for g in gts])

    def obb_of(verts):
        m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        T, ext = _fast_obb(m)
        return T, np.asarray(ext, float)

    def gt_metrics(verts, label_tag):
        T, ext = obb_of(verts)
        c = T[:3, 3]
        gi = int(np.argmin(np.linalg.norm(gt_cent - c, axis=1)))
        g = gts[gi]
        Rg = np.asarray(g.T_world_obj, float)[:3, :3]
        eg = np.asarray(g.extents, float)
        iou = geo.obb_iou(Box.from_pose(T, ext), Box.from_pose(np.asarray(g.T_world_obj, float), eg))
        sym = symmetry_for_label(g.label)
        rot, sc = geo.box_pose_error(T[:3, :3], ext, Rg, eg, sym)
        cd = float(np.linalg.norm(c - gt_cent[gi]))
        print(f"    {label_tag:8} vs GT '{g.label}': 3D-IoU={iou:.3f}  scale_err={np.max(sc):.3f}  "
              f"rot={rot:.1f}deg  centroid={cd:.3f}m")
        return iou

    print("\n  === GT validation (does silhouette win -> GT metrics?) ===")
    iou_lay = gt_metrics(verts0, "layout")
    iou_ref = gt_metrics(vh, "refined")
    print(f"  >>> 3D-IoU vs GT: {iou_lay:.3f} -> {iou_ref:.3f} "
          f"({'IMPROVED' if iou_ref > iou_lay else 'NO GAIN / worse'})")

    # ---- visualization: per-view overlay of mask vs layout vs refined silhouettes ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    OUT.mkdir(parents=True, exist_ok=True)
    n = len(views)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.2), dpi=140)
    if n == 1:
        axes = [axes]
    for ax, v in zip(axes, views):
        sil_l, _ = rend.silhouette_depth(verts0, faces, v["K"], v["Twc"])
        sil_r, _ = rend.silhouette_depth(vh, faces, v["K"], v["Twc"])
        bg = v["rgb"] if v["rgb"] is not None else np.zeros((v["H"], v["W"], 3), np.uint8)
        ax.imshow(bg)
        ax.contourf(v["mask"], levels=[0.5, 1], colors=["#2ca02c"], alpha=0.28)  # observed mask
        ax.contour(sil_l, levels=[0.5], colors=["#d62728"], linewidths=1.6)      # layout
        ax.contour(sil_r, levels=[0.5], colors=["#1f77b4"], linewidths=1.6)      # refined
        ax.set_title(f"frame {v['fid']}", fontsize=8); ax.axis("off")
    fig.legend(handles=[Line2D([], [], color="#2ca02c", lw=6, alpha=0.4, label="observed mask"),
                        Line2D([], [], color="#d62728", lw=2, label="layout silhouette"),
                        Line2D([], [], color="#1f77b4", lw=2, label="render-compare silhouette")],
               loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out = OUT / "render_compare_overlay.png"
    fig.savefig(out); plt.close(fig)
    print(f"\n  wrote overlay -> {out}")


if __name__ == "__main__":
    main()
