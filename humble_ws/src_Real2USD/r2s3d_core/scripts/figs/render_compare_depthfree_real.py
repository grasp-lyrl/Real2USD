"""Depth-free placement on REAL data (Go2/RealSense) -- expected to struggle.

Same recipe as the sim depth-free test (triangulate the object centroid from
mask-centroid rays + camera poses, then silhouette scale), but on real odometry.
The camera poses drift over the walk, so the triangulation rays are mutually
inconsistent -- this quantifies how much depth's single-view robustness buys on
real hardware. Uses the run's detection masks (no native GT masks on real).

  uv run --extra rosbag --extra registration --extra mesh \
    python scripts/figs/render_compare_depthfree_real.py [lounge-0|hallway-1]
"""
from __future__ import annotations
import sys, json
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
SCENE = sys.argv[1] if len(sys.argv) > 1 else "lounge-0"
KEY = SCENE.replace("-", "")
SG = ROOT / f"results/phase0_{KEY}_rs_icp/{SCENE}/scene_graph.json"
DET = ROOT / f"results/detections/gt/{SCENE}/detections.npz"
N_VIEWS, MIN_AREA, N_OBJ = 4, 800, 8


def triangulate(cpx, Ks, Ts):
    A = np.zeros((3, 3)); b = np.zeros(3)
    for (u, v), K, T in zip(cpx, Ks, Ts):
        d = T[:3, :3] @ (np.linalg.inv(K) @ np.array([u, v, 1.0])); d /= np.linalg.norm(d)
        P = np.eye(3) - np.outer(d, d); A += P; b += P @ T[:3, 3]
    return np.linalg.solve(A, b)


def main():
    sg = json.loads(SG.read_text()); queue = Path(sg["sam3d_queue"])
    objs = [{"job": o["job_id"], "mesh": queue / o["mesh"], "Twm": np.asarray(o["T_world_mesh"], float),
             "c": np.asarray(o["T_world_obj"], float)[:3, 3],
             "tids": [t for t in o.get("det_track_ids", []) if t >= 0]} for o in sg["objects"]]

    z = np.load(DET, allow_pickle=True); H, W = z["hw"]
    src = make_source("realsense", SCENE, stride=1)
    gts = src.gt() or []
    gc = np.array([np.asarray(g.T_world_obj, float)[:3, 3] for g in gts])

    # per-object views from its detection tracks: (frame_id, mask-centroid px, mask)
    for o in objs:
        sel = np.where(np.isin(z["track_ids"], o["tids"]))[0] if o["tids"] else []
        byf = {}
        for i in sel:
            fid = int(z["frame_ids"][i])
            m = np.unpackbits(z["masks_packed"][i])[:H * W].reshape(H, W).astype(bool)
            byf[fid] = byf.get(fid, np.zeros((H, W), bool)) | m
        o["frames"] = {fid: m for fid, m in byf.items() if m.sum() > MIN_AREA}

    cand = [o for o in objs if len(o["frames"]) >= N_VIEWS]
    cand.sort(key=lambda o: -len(o["frames"])); cand = cand[:N_OBJ]
    need = {fid for o in cand for fid in o["frames"]}
    cam = {}
    for f in src:
        if f.frame_id in need:
            cam[f.frame_id] = (np.asarray(f.K, float).reshape(3, 3), np.asarray(f.T_world_cam, float))
        if len(cam) == len(need):
            break

    rend = rcp.Renderer(W, H)
    print(f"{SCENE}: {len(cand)} objects\n{'obj':>18} {'tri-cent err':>12} {'3D-IoU':>16} {'scale_err':>16}")
    for o in cand:
        fr = sorted(o["frames"], key=lambda fid: -int(o["frames"][fid].sum()))[:N_VIEWS]
        views, cpx, Ks, Ts = [], [], [], []
        for fid in fr:
            if fid not in cam:
                continue
            K, T = cam[fid]; mk = o["frames"][fid]; ys, xs = np.where(mk)
            views.append({"mask": mk, "K": K, "Twc": T, "depth": np.zeros((H, W))})
            cpx.append((xs.mean(), ys.mean())); Ks.append(K); Ts.append(T)
        if len(views) < 2:
            continue
        tri = triangulate(cpx, Ks, Ts)
        gi = int(np.argmin(np.linalg.norm(gc - tri, axis=1)))
        g = gts[gi]; cent_err = float(np.linalg.norm(tri - gc[gi]))

        m = trimesh.load(str(o["mesh"]), force="mesh", process=False); m.apply_transform(o["Twm"])
        v0 = np.asarray(m.vertices, float); v0 = v0 + (tri - v0.mean(0))
        om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v0),
                                       o3d.utility.Vector3iVector(np.asarray(m.faces, np.int32)))
        if len(m.faces) > 3000:
            om = om.simplify_quadric_decimation(3000)
        vd, fd = np.asarray(om.vertices, float), np.asarray(om.triangles, np.int32); cen = vd.mean(0)
        res = minimize(rcp.make_loss(views, vd, fd, cen, rend),
                       np.array([0, 0, 0, 0, 1, 1, 1.], float), method="Powell",
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
        print(f"{str(o['job'])[-14:]:>18} {cent_err:>11.3f}m {i0:.2f}->{i1:.2f}     {s0:.2f}->{s1:.2f}  ({g.label})")


if __name__ == "__main__":
    main()
