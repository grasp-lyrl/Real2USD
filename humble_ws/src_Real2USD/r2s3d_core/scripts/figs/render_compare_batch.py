"""Render-and-compare batch validation on SIM (ProcTHOR, perfect poses).

Runs the render-compare pose+scale fit over N matched objects, under TWO mask
sources -- native GT masks (clean) and YOLOE detection masks (realistic) -- and
aggregates GT 3D-IoU / scale_err (layout vs render-compare). Tells us whether the
single-object fridge win generalizes, and how much detector-mask noise costs.

  uv run --extra procthor --extra registration --extra mesh \
    python scripts/figs/render_compare_batch.py
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
import importlib.util as _u
_spec = _u.spec_from_file_location("rcp", str(Path(__file__).with_name("render_compare_prototype.py")))
rcp = _u.module_from_spec(_spec); _spec.loader.exec_module(rcp)

ROOT = Path(__file__).resolve().parents[2]
SG = ROOT / "results/paper/sim/asset_layout_gt_s200/200/scene_graph.json"
DET = ROOT / "results/detections/procthor/gt/200/detections.npz"
N_OBJ, N_VIEWS, MIN_AREA = 10, 4, 2500


def decimate(mesh):
    om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(mesh.vertices, float)),
                                   o3d.utility.Vector3iVector(np.asarray(mesh.faces, np.int32)))
    if len(mesh.faces) > 3000:
        om = om.simplify_quadric_decimation(3000)
    return np.asarray(om.vertices, float), np.asarray(om.triangles, np.int32)


def fit_and_score(views, verts0, faces, g, rend):
    centroid = verts0.mean(0)
    loss = rcp.make_loss(views, verts0, faces, centroid, rend)
    res = minimize(loss, np.array([0, 0, 0, 0, 1, 1, 1], float), method="Powell",
                   options={"maxiter": 50, "xtol": 2e-3, "ftol": 2e-3})
    T = rcp._param_transform(res.x, centroid)
    vh = (T[:3, :3] @ verts0.T + T[:3, 3:4]).T

    def gm(verts):
        Tm, ext = _fast_obb(trimesh.Trimesh(vertices=verts, faces=faces, process=False))
        Rg = np.asarray(g.T_world_obj, float)[:3, :3]; eg = np.asarray(g.extents, float)
        iou = geo.obb_iou(Box.from_pose(Tm, np.asarray(ext, float)),
                          Box.from_pose(np.asarray(g.T_world_obj, float), eg))
        _, sc = geo.box_pose_error(Tm[:3, :3], np.asarray(ext, float), Rg, eg,
                                   symmetry_for_label(g.label))
        return float(iou), float(np.max(sc))
    return gm(verts0), gm(vh)


def main():
    d = json.loads(SG.read_text())
    queue = Path(d["sam3d_queue"])
    objs = [{"job": o["job_id"], "mesh": queue / o["mesh"], "Twm": np.asarray(o["T_world_mesh"], float),
             "c": np.asarray(o["T_world_obj"], float)[:3, 3], "label": o["label"],
             "dtids": [t for t in o.get("det_track_ids", []) if t >= 0]} for o in d["objects"]]

    z = np.load(DET, allow_pickle=True)
    H, W = z["hw"]
    src = make_source("procthor", "200", split="val", gt_mesh="box", stride=1)
    gts = src.gt() or []
    gid = {g.instance_id: g for g in gts}
    gc = {g.instance_id: np.asarray(g.T_world_obj, float)[:3, 3] for g in gts}

    # match each object -> nearest GT instance
    for o in objs:
        iid = min(gid, key=lambda i: np.linalg.norm(gc[i] - o["c"]))
        o["iid"] = iid if np.linalg.norm(gc[iid] - o["c"]) < 0.6 else None

    # pass 1: camera+depth per frame; native-mask coverage for matched instances
    cam, depth16, cover = {}, {}, {}
    matched_iids = {o["iid"] for o in objs if o["iid"] is not None}
    for f in src:
        fid = f.frame_id
        cam[fid] = (np.asarray(f.K, float).reshape(3, 3), np.asarray(f.T_world_cam, float))
        depth16[fid] = f.depth.astype(np.float16)
        for iid in matched_iids:
            m = src.native_mask(fid, iid)
            if m is not None:
                a = int((np.asarray(m) > 0).sum())
                if a > MIN_AREA:
                    cover.setdefault(iid, []).append(fid)

    # pick N_OBJ well-observed matched objects
    cand = [o for o in objs if o["iid"] in cover and len(cover[o["iid"]]) >= N_VIEWS]
    cand.sort(key=lambda o: -len(cover[o["iid"]]))
    cand = cand[:N_OBJ]
    print(f"selected {len(cand)} objects: {[(o['label'], len(cover[o['iid']])) for o in cand]}")

    # native chosen frames (spread) per object
    for o in cand:
        fr = sorted(cover[o["iid"]])
        o["native_frames"] = [fr[i] for i in np.linspace(0, len(fr) - 1, N_VIEWS).round().astype(int)]

    # pass 2: capture native masks for chosen frames
    need = {(o["iid"], fid) for o in cand for fid in o["native_frames"]}
    nat = {}
    for f in src:
        for (iid, fid) in [k for k in need if k[1] == f.frame_id]:
            m = src.native_mask(fid, iid)
            if m is not None:
                nat[(iid, fid)] = np.asarray(m) > 0

    # YOLOE masks per object (from det_track_ids)
    det_by_tid_frame = {}
    for i in range(len(z["track_ids"])):
        det_by_tid_frame.setdefault(int(z["track_ids"][i]), {}).setdefault(int(z["frame_ids"][i]), []).append(i)

    rend = rcp.Renderer(W, H)
    agg = {"native": {"lay": [], "ref": [], "lay_s": [], "ref_s": []},
           "yoloe": {"lay": [], "ref": [], "lay_s": [], "ref_s": []}}

    for o in cand:
        g = gid[o["iid"]]
        mesh = trimesh.load(str(o["mesh"]), force="mesh", process=False); mesh.apply_transform(o["Twm"])
        verts0, faces = decimate(mesh)

        # native views
        nv = []
        for fid in o["native_frames"]:
            if (o["iid"], fid) in nat and fid in cam:
                K, Twc = cam[fid]
                nv.append({"mask": nat[(o["iid"], fid)], "K": K, "Twc": Twc,
                           "depth": depth16[fid].astype(np.float32), "H": H, "W": W})
        # yoloe views: frames with a det of this object's track ids, area>thresh, spread
        yframes = []
        for tid in o["dtids"]:
            for fid, idxs in det_by_tid_frame.get(tid, {}).items():
                if fid not in cam:
                    continue
                mk = np.zeros((H, W), bool)
                for i in idxs:
                    mk |= np.unpackbits(z["masks_packed"][i])[:H * W].reshape(H, W).astype(bool)
                if mk.sum() > MIN_AREA:
                    yframes.append((fid, mk))
        yframes.sort(key=lambda t: t[0])
        yv = []
        if yframes:
            sel = np.linspace(0, len(yframes) - 1, min(N_VIEWS, len(yframes))).round().astype(int)
            for i in sel:
                fid, mk = yframes[i]
                K, Twc = cam[fid]
                yv.append({"mask": mk, "K": K, "Twc": Twc,
                           "depth": depth16[fid].astype(np.float32), "H": H, "W": W})

        for tag, vs in (("native", nv), ("yoloe", yv)):
            if len(vs) < 2:
                continue
            try:
                (li, ls), (ri, rs) = fit_and_score(vs, verts0, faces, g, rend)
            except Exception as e:
                print(f"  {o['label']:12} {tag}: FAIL {e}"); continue
            agg[tag]["lay"].append(li); agg[tag]["ref"].append(ri)
            agg[tag]["lay_s"].append(ls); agg[tag]["ref_s"].append(rs)
            print(f"  {o['label']:12} {tag:6} ({len(vs)}v): 3D-IoU {li:.2f}->{ri:.2f}  scale {ls:.2f}->{rs:.2f}")

    print("\n=== AGGREGATE (median over objects) ===")
    for tag in ("native", "yoloe"):
        a = agg[tag]
        if not a["lay"]:
            print(f"  {tag}: no objects"); continue
        n = len(a["lay"])
        print(f"  {tag:6} n={n}: 3D-IoU {np.median(a['lay']):.3f}->{np.median(a['ref']):.3f}  "
              f"scale_err {np.median(a['lay_s']):.3f}->{np.median(a['ref_s']):.3f}  "
              f"(improved {sum(r>l for l,r in zip(a['lay'],a['ref']))}/{n})")


if __name__ == "__main__":
    main()
