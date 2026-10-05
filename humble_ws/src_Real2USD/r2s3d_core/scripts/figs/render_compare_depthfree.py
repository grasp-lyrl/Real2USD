"""Depth-FREE placement in sim: can we place metrically with no depth at all?

No depth for SAM3D input, no depth for ICP, no depth for the scale metric. Placement
uses only: (a) the SAM3D mesh + its predicted rotation (shape+orientation), (b) the
object centroid TRIANGULATED from mask-centroid rays across views (metric position
from the camera baselines -- odometry, not depth), and (c) per-axis scale from
multi-view silhouette matching. Metric scale therefore comes from the camera motion,
not a depth sensor; multi-view is mandatory. Validated in sim (perfect metric poses).

  uv run --extra procthor --extra registration --extra mesh \
    python scripts/figs/render_compare_depthfree.py
"""
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np
import trimesh, open3d as o3d
from scipy.optimize import minimize

from r2s3d_core.data.registry import make_source
from r2s3d_core.eval import geometry as geo
from r2s3d_core.eval.geometry import Box
from r2s3d_core.eval.metrics import symmetry_for_label
from r2s3d_core.baselines.sam3d_layout import _fast_obb
import importlib.util as _u
_spec = _u.spec_from_file_location("rcp", str(Path(__file__).with_name("render_compare_prototype.py")))
rcp = _u.module_from_spec(_spec); _spec.loader.exec_module(rcp)

ROOT = Path(__file__).resolve().parents[2]
SG = Path(os.environ.get("RC_DF_SG",
          ROOT / "results/paper/sim/asset_layout_gt_s200/200/scene_graph.json"))
N_OBJ, N_VIEWS, MIN_AREA = 8, 4, 2500


def triangulate(centroids_px, Ks, Twcs):
    """Least-squares 3D point nearest to the back-projected mask-centroid rays.
    Uses only 2D centroids + metric camera poses -- NO depth."""
    A = np.zeros((3, 3)); b = np.zeros(3)
    for (u, v), K, T in zip(centroids_px, Ks, Twcs):
        d = T[:3, :3] @ (np.linalg.inv(K) @ np.array([u, v, 1.0]))
        d /= np.linalg.norm(d)
        P = np.eye(3) - np.outer(d, d)          # projector orthogonal to the ray
        A += P; b += P @ T[:3, 3]
    return np.linalg.solve(A, b)


def main():
    d = json.loads(SG.read_text()); queue = Path(d["sam3d_queue"])
    objs = [{"job": o["job_id"], "mesh": queue / o["mesh"], "Twm": np.asarray(o["T_world_mesh"], float),
             "c": np.asarray(o["T_world_obj"], float)[:3, 3]} for o in d["objects"]]

    src = make_source("procthor", "200", split="val", gt_mesh="box", stride=1)
    gts = src.gt() or []
    gid = {g.instance_id: g for g in gts}
    gc = {g.instance_id: np.asarray(g.T_world_obj, float)[:3, 3] for g in gts}
    for o in objs:
        iid = min(gid, key=lambda i: np.linalg.norm(gc[i] - o["c"]))
        o["iid"] = iid if np.linalg.norm(gc[iid] - o["c"]) < 0.6 else None

    cam, cover = {}, {}
    matched = {o["iid"] for o in objs if o["iid"] is not None}
    for f in src:
        cam[f.frame_id] = (np.asarray(f.K, float).reshape(3, 3), np.asarray(f.T_world_cam, float))
        for iid in matched:
            m = src.native_mask(f.frame_id, iid)
            if m is not None:
                mb = np.asarray(m) > 0
                if int(mb.sum()) > MIN_AREA:
                    ys, xs = np.where(mb)
                    cover.setdefault(iid, []).append((f.frame_id, (xs.mean(), ys.mean())))

    cand = [o for o in objs if o["iid"] in cover and len(cover[o["iid"]]) >= N_VIEWS]
    cand.sort(key=lambda o: -len(cover[o["iid"]])); cand = cand[:N_OBJ]

    # capture native masks for the chosen (spread) frames
    for o in cand:
        lst = sorted(cover[o["iid"]]); idx = np.linspace(0, len(lst) - 1, N_VIEWS).round().astype(int)
        o["frames"] = [lst[i] for i in idx]
    need = {(o["iid"], fid) for o in cand for (fid, _c) in o["frames"]}
    masks = {}
    for f in src:
        for iid, fid in [k for k in need if k[1] == f.frame_id]:
            mm = src.native_mask(fid, iid)
            if mm is not None:
                masks[(iid, fid)] = np.asarray(mm) > 0

    rend = rcp.Renderer(640, 480)
    print(f"{'obj':>4} {'tri-cent err':>12} {'3D-IoU':>16} {'scale_err':>18}")
    for o in cand:
        g = gid[o["iid"]]
        views, cpxs, Ks, Ts = [], [], [], []
        for fid, cpx in o["frames"]:
            if (o["iid"], fid) not in masks:
                continue
            K, T = cam[fid]; mk = masks[(o["iid"], fid)]
            # zeros depth -> the prototype loss's depth term stays inert (this is depth-FREE)
            views.append({"mask": mk, "K": K, "Twc": T, "depth": np.zeros(mk.shape)})
            cpxs.append(cpx); Ks.append(K); Ts.append(T)
        if len(views) < 2:
            continue
        # (b) DEPTH-FREE position: triangulate centroid from mask rays + metric camera poses
        tri = triangulate(cpxs, Ks, Ts)
        cent_err = float(np.linalg.norm(tri - gc[o["iid"]]))

        mesh = trimesh.load(str(o["mesh"]), force="mesh", process=False); mesh.apply_transform(o["Twm"])
        v0 = np.asarray(mesh.vertices, float); f0 = np.asarray(mesh.faces, np.int32)
        v0 = v0 + (tri - v0.mean(0))            # reposition to triangulated centroid (no depth)
        om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v0), o3d.utility.Vector3iVector(f0))
        if len(f0) > 3000:
            om = om.simplify_quadric_decimation(3000)
        vd, fd = np.asarray(om.vertices, float), np.asarray(om.triangles, np.int32)
        cen = vd.mean(0)

        # (c) scale from multi-view silhouette (no depth); orientation from SAM3D (layout)
        loss = rcp.make_loss(views, vd, fd, cen, rend)
        res = minimize(loss, np.array([0, 0, 0, 0, 1, 1, 1.], float), method="Powell",
                       options={"maxiter": 40, "xtol": 2e-3, "ftol": 2e-3})
        T = rcp._param_transform(res.x, cen); vh = (T[:3, :3] @ vd.T + T[:3, 3:4]).T

        def m3(v):
            Tm, ext = _fast_obb(trimesh.Trimesh(vertices=v, faces=fd, process=False))
            eg = np.asarray(g.extents, float)
            iou = geo.obb_iou(Box.from_pose(Tm, np.asarray(ext, float)),
                              Box.from_pose(np.asarray(g.T_world_obj, float), eg))
            _, sc = geo.box_pose_error(Tm[:3, :3], np.asarray(ext, float),
                                       np.asarray(g.T_world_obj, float)[:3, :3], eg, symmetry_for_label(g.label))
            return iou, float(np.max(sc))
        i0, s0 = m3(vd); i1, s1 = m3(vh)
        print(f"{o['iid']:>4} {cent_err:>11.3f}m  {i0:.2f}->{i1:.2f}       {s0:.2f}->{s1:.2f}   ({g.label})")


if __name__ == "__main__":
    main()
