"""Experiment 1: paired per-object scale-error test — does scale-fit measurably help on real data,
independent of the centroid-scatter-floored IoU-F1?

For each track present in BOTH the `layout` (SAM3D-native) and `scale_icp` (our reprojection scale-fit
+ ICP) runs, match it to a GT cuboid by centroid (<= 1 m), and compute its axis-invariant scale error
under each. Because the track's centroid barely moves between the two, the pairing is clean. Pool over
the 4 Go2 scenes; report win-rate (scale-fit reduces the object's scale error) + Wilcoxon p — a
statement immune to matching-set noise and centroid scatter.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

from r2s3d_core.data.supervisely import load_supervisely_gt
from r2s3d_core.eval import geometry as geo
from r2s3d_core.eval.metrics import symmetry_for_label

GT_DIR = Path("/home/chris.hsu/repos/Real2USD/humble_ws/evaluations/supervisely")
SCENES = {  # (layout run, scale_icp run, gt scene id)
    "hallway-1": ("phase0_hallway1_rs_layout", "phase0_hallway1_rs_scale_icp_reproj"),
    "lounge-0": ("phase0_lounge0_rs_layout", "phase0_lounge0_rs_scaleicp"),
    "smalloffice-0": ("phase0_smalloffice0_rs_layout", "phase0_smalloffice0_rs_scaleicp"),
    "smalloffice-1": ("phase0_smalloffice1_rs_layout", "phase0_smalloffice1_rs_scaleicp"),
}
MATCH_M = 1.0


def _load_objs(run_dir: str):
    fs = glob.glob(f"results/{run_dir}/*/scene_graph.json")
    if not fs:
        return {}
    objs = json.load(open(fs[0]))["objects"]
    out = {}
    for o in objs:
        T = np.asarray(o["T_world_obj"], float)
        out[int(o["id"])] = {"R": T[:3, :3], "c": T[:3, 3], "ext": np.asarray(o["extents"], float),
                             "label": o.get("label", "")}
    return out


def _scale_err(pred, g):
    sym = symmetry_for_label(g.label)
    _, s = geo.box_pose_error(pred["R"], pred["ext"], g.T_world_obj[:3, :3], g.extents, sym)
    return float(np.max(s))


def main():
    pairs = []  # (scene, track, scale_err_layout, scale_err_scaleicp, cent_layout, cent_scaleicp)
    for scene, (lrun, srun) in SCENES.items():
        lay, sic = _load_objs(lrun), _load_objs(srun)
        gts = load_supervisely_gt(str(GT_DIR / f"{scene}_voxel_pointcloud.pcd.json"))
        gc = np.array([g.T_world_obj[:3, 3] for g in gts])
        used = set()
        for tid in sorted(set(lay) & set(sic)):
            c = lay[tid]["c"]
            d = np.linalg.norm(gc - c, axis=1)
            order = np.argsort(d)
            gi = next((int(j) for j in order if j not in used and d[j] <= MATCH_M), None)
            if gi is None:
                continue
            used.add(gi)
            g = gts[gi]
            pairs.append((scene, tid,
                          _scale_err(lay[tid], g), _scale_err(sic[tid], g),
                          float(np.linalg.norm(lay[tid]["c"] - g.T_world_obj[:3, 3])),
                          float(np.linalg.norm(sic[tid]["c"] - g.T_world_obj[:3, 3]))))

    sl = np.array([p[2] for p in pairs]); ss = np.array([p[3] for p in pairs])
    cl = np.array([p[4] for p in pairs]); cs = np.array([p[5] for p in pairs])
    n = len(pairs)
    print(f"paired objects (matched under both, <= {MATCH_M} m): {n}\n")

    # per-object CSV (LaTeX-able)
    import csv as _csv
    out = Path("results/paper/_tables/paired_scale_test.csv"); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["scene", "track_id", "scale_err_layout", "scale_err_scaleicp",
                    "centroid_err_layout", "centroid_err_scaleicp"])
        for p in pairs:
            w.writerow([p[0], p[1], round(p[2], 4), round(p[3], 4), round(p[4], 4), round(p[5], 4)])
    print(f"wrote {out}\n")

    def report(name, a, b):
        win = int(np.sum(b < a)); tie = int(np.sum(np.isclose(a, b)))
        print(f"=== {name} ===")
        print(f"  layout  median {np.median(a):.3f}  mean {np.mean(a):.3f}")
        print(f"  scale+ICP median {np.median(b):.3f}  mean {np.mean(b):.3f}")
        print(f"  scale-fit better: {win}/{n}  (ties {tie})")
        try:
            from scipy.stats import wilcoxon
            nz = ~np.isclose(a, b)
            if nz.sum() >= 5:
                w = wilcoxon(a[nz], b[nz], alternative="greater")  # H1: layout error > scaleicp error
                print(f"  Wilcoxon (layout>scaleicp) p = {w.pvalue:.4g}  (n_nonzero {int(nz.sum())})")
        except Exception as e:
            print("  wilcoxon skipped:", e)
        print()

    report("SCALE error (max per-axis, axis-invariant)", sl, ss)
    report("CENTROID error (m)", cl, cs)

    # per-scene win counts (scale)
    print("=== per-scene scale-fit win-rate (scale) ===")
    for scene in SCENES:
        idx = [i for i, p in enumerate(pairs) if p[0] == scene]
        if idx:
            w = sum(ss[i] < sl[i] for i in idx)
            print(f"  {scene:14s} {w}/{len(idx)}")


if __name__ == "__main__":
    main()
