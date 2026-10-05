"""Render-and-compare on SIM (ProcTHOR) -- the decisive test.

On real, render-compare improved silhouette IoU but WORSENED GT 3D-IoU, because the
camera poses are miscalibrated (a 3D-correct object reprojects to the wrong pixels,
so matching the mask pulls the mesh to a wrong 3D pose). ProcTHOR has PERFECT camera
poses and native GT masks, so this isolates the method: if silhouette improvement
-> GT-3D-IoU improvement here, render-compare is valid and the real failure is
calibration, not the algorithm.

  uv run --extra procthor --extra registration --extra mesh \
    python scripts/figs/render_compare_sim.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import trimesh
from scipy.optimize import minimize

from r2s3d_core.data.registry import make_source
from r2s3d_core.eval import geometry as geo
from r2s3d_core.eval.geometry import Box
from r2s3d_core.eval.metrics import symmetry_for_label
from r2s3d_core.baselines.sam3d_layout import _fast_obb
# reuse the prototype's renderer + optimizer pieces
import importlib.util as _u
_spec = _u.spec_from_file_location(
    "rcp", str(Path(__file__).with_name("render_compare_prototype.py")))
rcp = _u.module_from_spec(_spec); _spec.loader.exec_module(rcp)

ROOT = Path(__file__).resolve().parents[2]
SG = ROOT / "results/paper/sim/asset_layout_gt_s200/200/scene_graph.json"
SCENE, SPLIT = "200", "val"
N_VIEWS = 4
OUT = ROOT / "results/paper/_figs/fig1"


def load_scene_objs():
    d = json.loads(SG.read_text())
    queue = Path(d["sam3d_queue"])
    objs = []
    for o in d["objects"]:
        c = np.asarray(o["T_world_obj"], float)[:3, 3]
        objs.append({"job": o["job_id"], "mesh": queue / o["mesh"],
                     "Twm": np.asarray(o["T_world_mesh"], float), "c": c, "label": o["label"]})
    return objs


def main():
    src = make_source("procthor", SCENE, split=SPLIT, gt_mesh="box", stride=1)
    gts = src.gt() or []
    gid = {g.instance_id: g for g in gts}

    # pass 1: per-instance coverage + store camera/depth/rgb per frame (cache replay is cheap)
    frames = {}
    cover = {}
    for f in src:
        frames[f.frame_id] = (np.asarray(f.K, float).reshape(3, 3),
                              np.asarray(f.T_world_cam, float),
                              f.depth.astype(np.float32), np.asarray(f.rgb))
        for g in gts:
            m = src.native_mask(f.frame_id, g.instance_id)
            if m is None:
                continue
            a = int((np.asarray(m) > 0).sum())
            if a > 3000:
                cover.setdefault(g.instance_id, []).append((f.frame_id, a))

    objs = load_scene_objs()
    obj_c = np.array([o["c"] for o in objs])

    # target: well-observed GT instance that matches a scene-graph object (mesh available)
    best = None
    for iid, lst in cover.items():
        if len(lst) < N_VIEWS:
            continue
        gc = np.asarray(gid[iid].T_world_obj, float)[:3, 3]
        j = int(np.argmin(np.linalg.norm(obj_c - gc, axis=1)))
        if np.linalg.norm(obj_c[j] - gc) > 0.6:
            continue
        score = len(lst)
        if best is None or score > best[0]:
            best = (score, iid, j, lst)
    if best is None:
        print("no well-observed matched object found"); return
    _, iid, j, lst = best
    obj = objs[j]
    g = gid[iid]
    print(f"target GT instance {iid} '{g.label}' <- mesh {obj['job']} '{obj['label']}' "
          f"({len(lst)} good frames)")

    # viewpoint-diverse views: spread across the good frames
    lst.sort(key=lambda t: t[0])
    idx = np.linspace(0, len(lst) - 1, N_VIEWS).round().astype(int)
    chosen = [lst[i][0] for i in idx]
    views = []
    for fid in chosen:
        K, Twc, depth, rgb = frames[fid]
        mask = np.asarray(src.native_mask(fid, iid)) > 0
        views.append({"mask": mask, "K": K, "Twc": Twc, "depth": depth, "rgb": rgb,
                      "fid": fid, "H": mask.shape[0], "W": mask.shape[1]})
    print(f"  views (frames): {chosen}")

    # seed mesh at layout pose, decimated for fast silhouette render
    m = trimesh.load(str(obj["mesh"]), force="mesh", process=False)
    m.apply_transform(obj["Twm"])
    import open3d as o3d
    om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(m.vertices, float)),
                                   o3d.utility.Vector3iVector(np.asarray(m.faces, np.int32)))
    if len(m.faces) > 3000:
        om = om.simplify_quadric_decimation(3000)
    verts0 = np.asarray(om.vertices, float); faces = np.asarray(om.triangles, np.int32)
    centroid = verts0.mean(0)
    rend = rcp.Renderer(views[0]["W"], views[0]["H"])

    iou0 = rcp.mean_iou(views, verts0, faces, rend)
    loss = rcp.make_loss(views, verts0, faces, centroid, rend)
    p0 = np.array([0, 0, 0, 0, 1, 1, 1], float)
    print(f"  layout silhouette IoU {[round(x,3) for x in iou0]} mean={np.mean(iou0):.3f}; "
          f"seed loss={loss(p0):.4f} -- optimizing...")
    res = minimize(loss, p0, method="Powell", options={"maxiter": 60, "xtol": 1e-3, "ftol": 1e-3})
    T = rcp._param_transform(res.x, centroid)
    vh = (T[:3, :3] @ verts0.T + T[:3, 3:4]).T
    iou1 = rcp.mean_iou(views, vh, faces, rend)
    print(f"  refined silhouette IoU {[round(x,3) for x in iou1]} mean={np.mean(iou1):.3f}")

    def gt_metrics(verts, tag):
        Tm, ext = _fast_obb(trimesh.Trimesh(vertices=verts, faces=faces, process=False))
        Rg = np.asarray(g.T_world_obj, float)[:3, :3]; eg = np.asarray(g.extents, float)
        iou = geo.obb_iou(Box.from_pose(Tm, np.asarray(ext, float)),
                          Box.from_pose(np.asarray(g.T_world_obj, float), eg))
        rot, sc = geo.box_pose_error(Tm[:3, :3], np.asarray(ext, float), Rg, eg,
                                     symmetry_for_label(g.label))
        cd = float(np.linalg.norm(Tm[:3, 3] - np.asarray(g.T_world_obj, float)[:3, 3]))
        print(f"    {tag:8} 3D-IoU={iou:.3f} scale_err={np.max(sc):.3f} rot={rot:.1f}deg cent={cd:.3f}m")
        return iou

    print("  === GT validation (PERFECT sim poses) ===")
    a = gt_metrics(verts0, "layout"); b = gt_metrics(vh, "refined")
    print(f"  >>> silhouette {np.mean(iou0):.3f}->{np.mean(iou1):.3f} | "
          f"3D-IoU {a:.3f}->{b:.3f} ({'IMPROVED' if b > a else 'NO GAIN / worse'})")

    # overlay
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(views), figsize=(4 * len(views), 3.2), dpi=140)
    axes = np.atleast_1d(axes)
    for ax, v in zip(axes, views):
        sl, _ = rend.silhouette_depth(verts0, faces, v["K"], v["Twc"])
        sr, _ = rend.silhouette_depth(vh, faces, v["K"], v["Twc"])
        ax.imshow(v["rgb"])
        ax.contourf(v["mask"], levels=[0.5, 1], colors=["#2ca02c"], alpha=0.28)
        ax.contour(sl, levels=[0.5], colors=["#d62728"], linewidths=1.6)
        ax.contour(sr, levels=[0.5], colors=["#1f77b4"], linewidths=1.6)
        ax.set_title(f"frame {v['fid']}", fontsize=8); ax.axis("off")
    fig.legend(handles=[Line2D([], [], color="#2ca02c", lw=6, alpha=0.4, label="GT mask"),
                        Line2D([], [], color="#d62728", lw=2, label="layout"),
                        Line2D([], [], color="#1f77b4", lw=2, label="render-compare")],
               loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    p = OUT / "render_compare_sim_overlay.png"; fig.savefig(p); plt.close(fig)
    print(f"  wrote overlay -> {p}")


if __name__ == "__main__":
    main()
