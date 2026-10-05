"""Qualitative panel for the generation-value figure (Fig 4 inset).

Renders, for one object, three point sets in the object's OBB-local frame (centered, upright,
metric) from a common viewpoint:
  (A) observed cluster  — the fused points the robot actually saw (a fragment)
  (B) completed asset   — the oracle-placed SAM3D mesh surface (the whole object)
  (C) GT mesh           — ground truth
Shows visually why generation matters under partial views (chair-under-table). Reuses the exact
oracle-placement + cluster-cleaning from gen_shape_completion.py.

Usage:
  uv run --extra procthor --extra mesh --extra registration --extra detector --extra viz \
    python scripts/render_completion_panel.py --scene 200 --track-id 128 \
    --asset-run results/procthor_procthor_object_track_icp_s200_val
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from r2s3d_core.data.registry import make_source
from r2s3d_core.detect.cache import DetectionSet
from r2s3d_core.eval import geometry as geo
from r2s3d_core.tracks import TrackState, run_tracker
from gen_shape_completion import _ROT24, _to_obb_local, oracle_place_asset, N


def _load_asset(asset_run: Path, tid: int):
    import trimesh
    sg = json.load(open(next(asset_run.glob("*/scene_graph.json"))))
    queue = Path(sg.get("sam3d_queue") or (asset_run / "sam3d_queue"))
    for o in sg["objects"]:
        if int(o["id"]) == tid and o.get("mesh") and (queue / o["mesh"]).exists():
            m = trimesh.load(str(queue / o["mesh"]), force="mesh")
            m.apply_transform(np.asarray(o["T_world_mesh"], float))
            return m
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="200")
    ap.add_argument("--split", default="val")
    ap.add_argument("--track-id", type=int, required=True)
    ap.add_argument("--asset-run", required=True)
    ap.add_argument("--detections", default="results/detections/procthor/gt")
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--elev", type=float, default=18)
    ap.add_argument("--azim", type=float, default=None, help="default None = auto (view unobserved side)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--separate", action="store_true",
                    help="save 3 tight, title-less panels (<base>_a/_b/_c.png) for "
                         "LaTeX subfigure arrangement instead of one combined image")
    args = ap.parse_args()

    src = make_source("procthor", args.scene, split=args.split, gt_mesh="asset", stride=1)
    frames = list(src)
    gts = [g for g in src.gt() if getattr(g, "mesh", None) is not None]
    ds = DetectionSet.load(args.detections, scene=args.scene)
    tracks = run_tracker(frames, ds.by_frame(), {"reid": True, "late_merge": True})
    t = next((x for x in tracks if x.track_id == args.track_id and x.state == TrackState.MATURE), None)
    if t is None:
        raise SystemExit(f"track {args.track_id} not found / not mature")

    gt_c = np.array([g.T_world_obj[:3, 3] for g in gts])
    gi = int(np.argmin(np.linalg.norm(gt_c[:, :2] - t.centroid[:2], axis=1)))
    g = gts[gi]
    gR, gc, gext = g.T_world_obj[:3, :3], g.T_world_obj[:3, 3], np.asarray(g.extents, float)

    gpts = geo.sample_surface(g.mesh, N, seed=1)
    obs = np.asarray(t.fused_cloud, float)
    loc = _to_obb_local(obs, gR, gc, gext)
    obs_clip = obs[np.all(np.abs(loc) <= 0.5 * 1.15, axis=1)]

    a_unit = oracle_place_asset(_load_asset(Path(args.asset_run), args.track_id), gR, gc, gext)
    g_unit = _to_obb_local(gpts, gR, gc, gext)
    best_R = min(_ROT24, key=lambda R: geo.chamfer_and_fscore(a_unit @ R.T, g_unit)["chamfer_mean"])
    asset_world = (a_unit @ best_R.T) * gext @ gR.T + gc

    # center at the object but keep WORLD axes so gravity-up is preserved (the OBB up-axis sign is
    # ambiguous and was rendering objects upside-down). World is Z-up, so objects stay upright.
    def L(p):
        return p - gc
    from scipy.spatial import cKDTree
    d_cov, _ = cKDTree(obs_clip).query(gpts)
    observed = d_cov <= args.tau
    coverage = float(observed.mean())

    # density-match the cluster to the asset/GT sample count so the panels compare STRUCTURE, not
    # point density (the raw fused cloud is far denser and would read as "more complete").
    rng = np.random.RandomState(0)
    obs_viz = obs_clip if len(obs_clip) <= N else obs_clip[rng.choice(len(obs_clip), N, replace=False)]

    # view from the UNOBSERVED side so the cluster's hole (and the asset filling it) is visible;
    # otherwise the densely-seen front hides the missing back/underside.
    if args.azim is None:
        u = L(gpts[~observed])[:, :2].mean(0) if (~observed).any() else np.array([1.0, 0.0])
        azim = float(np.degrees(np.arctan2(u[1], u[0])))
    else:
        azim = args.azim

    # the part GENERATION ADDS = asset surface not explained by the observed cluster
    added = asset_world[cKDTree(obs_clip).query(asset_world)[0] > args.tau]

    # ---- render knobs (edit here) ------------------------------------------
    PT = 2.0                                  # scatter point size
    PAD = 1.05                                # bbox padding (1.0 = hug tight)
    RED, BLUE, GREY = "#d1495b", "#2e86ab", "#6b7280"
    plt.rcParams["pdf.fonttype"] = 42         # embed TrueType (correct for any PDF text)
    plt.rcParams["ps.fonttype"] = 42
    # ------------------------------------------------------------------------

    # Shared, object-hugging bounding box (from the full GT points) so all three
    # panels use identical limits AND the 3D box matches the object's real shape
    # -> no cubic whitespace around a short/wide object like an armchair.
    Pref = L(gpts)
    bmin, bmax = Pref.min(0), Pref.max(0)
    bctr, bhalf = 0.5 * (bmin + bmax), 0.5 * (bmax - bmin) * PAD

    # each panel = list of (points, colour, legend-label); NO titles (caption describes).
    panels = {
        "a": [(L(obs_viz), RED, None)],                                   # observed cluster
        "b": [(L(obs_viz), RED, "observed"), (L(added), BLUE, "generated")],  # seen + generated
        "c": [(L(gpts), GREY, None)],                                     # ground truth
    }

    def _draw(axp, key, legend=False):
        for P, c, lab in panels[key]:
            axp.scatter(P[:, 0], P[:, 1], P[:, 2], s=PT, c=c, linewidths=0, label=lab)
        axp.set_xlim(bctr[0] - bhalf[0], bctr[0] + bhalf[0])
        axp.set_ylim(bctr[1] - bhalf[1], bctr[1] + bhalf[1])
        axp.set_zlim(bctr[2] - bhalf[2], bctr[2] + bhalf[2])
        axp.set_box_aspect(tuple(bhalf))  # box matches object extents -> dense
        axp.view_init(elev=args.elev, azim=azim)
        axp.set_axis_off()  # drop panes/ticks/labels -> no chrome
        if key == "b" and legend:
            axp.legend(loc="upper right", fontsize=8, markerscale=3, frameon=False)

    def _autocrop(path, pad=6, also_pdf=False):
        """Crop the saved PNG to its non-white bounding box -> kills the
        whitespace matplotlib's 3D axes reserve that bbox_inches can't. With
        ``also_pdf``, wrap the cropped raster in a PDF (right choice for a dense
        point cloud -- a vector PDF of ~24k scatter points would bloat)."""
        from PIL import Image, ImageChops
        im = Image.open(path).convert("RGB")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        diff = ImageChops.difference(im, bg).convert("L").point(lambda p: 255 if p > 8 else 0)
        bb = diff.getbbox()
        if bb:
            l, t, r, b = bb
            im = im.crop((max(0, l - pad), max(0, t - pad),
                          min(im.width, r + pad), min(im.height, b + pad)))
            im.save(path)
        if also_pdf:
            pdf = str(Path(path).with_suffix(".pdf"))
            im.save(pdf, "PDF", resolution=200.0)
            print(f"wrote {pdf}")

    base = (Path(args.out) if args.out else
            Path(f"results/paper/_figs/completion_panel_s{args.scene}_t{args.track_id}"))
    base = base.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)

    if args.separate:
        # three tight, title-less, legend-less images -> arrange with LaTeX
        # subfigures; put the red=observed / blue=generated key in the caption.
        for key in ("a", "b", "c"):
            f = plt.figure(figsize=(3.0, 3.0))
            _draw(f.add_subplot(111, projection="3d"), key)
            f.subplots_adjust(left=0, right=1, bottom=0, top=1)
            out = f"{base}_{key}.png"
            f.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.0)
            plt.close(f); _autocrop(out, also_pdf=True); print(f"wrote {out}")
    else:
        fig = plt.figure(figsize=(9.0, 3.1))
        for k, key in enumerate(("a", "b", "c"), start=1):
            _draw(fig.add_subplot(1, 3, k, projection="3d"), key, legend=True)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.0)
        out = f"{base}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.02)
        _autocrop(out); print(f"wrote {out}")
    print(f"coverage {coverage:.2f} | obs_clip {len(obs_clip)} pts")


if __name__ == "__main__":
    main()
