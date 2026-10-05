"""Qualitative panel for C1 (Fig 2) — registration vs SAM 3D-native placement.

For one object, three point sets in a common centred world frame (Z-up), from a
common viewpoint:
  (a) SAM 3D layout   — the SAM 3D mesh at SAM 3D's OWN predicted pose (red),
                        overlaid on the GT surface (light grey) => misplaced.
  (b) +ICP (ours)     — the same mesh after our registration (blue), overlaid on
                        GT (light grey) => snapped into place.
  (c) ground truth    — GT surface alone (grey).

Reuses the layout/icp runs' per-object ``T_world_mesh`` (same front-end + mesh
cache, so track ids match across runs). Mirrors render_completion_panel.py:
dense, title-less, autocropped, PNG + PDF.

Usage:
  uv run --extra procthor --extra mesh --extra registration --extra detector --extra viz \
    python scripts/render_c1_panel.py --scene 200 --track-id 26 \
      --layout-run results/paper/sim/asset_layout_gt_s200 \
      --icp-run    results/paper/sim/asset_icp_gt_s200 --separate

  (omit --track-id to print a ranked table of tracks where ICP most improves
   centroid over layout, then pick a good example.)
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
from r2s3d_core.eval import geometry as geo
from gen_shape_completion import N

# ---- render knobs (edit here) ---------------------------------------------
PT = 2.0                                   # predicted-mesh point size
PT_GT = 1.4                                # GT reference point size
GT_ALPHA = 0.30                            # GT reference opacity in (a)/(b)
PAD = 1.05
RED, BLUE, GREY = "#d1495b", "#2e86ab", "#6b7280"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
# ---------------------------------------------------------------------------


def _posed_points(run: Path, tid: int, n: int) -> np.ndarray | None:
    """Sample the SAM 3D mesh for track ``tid`` at its posed (T_world_mesh) pose."""
    import trimesh
    sg = json.load(open(next(run.glob("*/scene_graph.json"))))
    queue = Path(sg.get("sam3d_queue") or (run / "sam3d_queue"))
    for o in sg["objects"]:
        if int(o["id"]) == tid and o.get("mesh") and (queue / o["mesh"]).exists():
            m = trimesh.load(str(queue / o["mesh"]), force="mesh")
            m.apply_transform(np.asarray(o["T_world_mesh"], float))
            return geo.sample_surface(m, n, seed=tid)
    return None


def _match_gt(gts, centroid):
    c = np.array([g.T_world_obj[:3, 3] for g in gts])
    return gts[int(np.argmin(np.linalg.norm(c[:, :2] - centroid[:2], axis=1)))]


def _autocrop(path, pad=6, also_pdf=False):
    from PIL import Image, ImageChops
    im = Image.open(path).convert("RGB")
    diff = ImageChops.difference(im, Image.new("RGB", im.size, (255, 255, 255)))
    bb = diff.convert("L").point(lambda p: 255 if p > 8 else 0).getbbox()
    if bb:
        l, t, r, b = bb
        im = im.crop((max(0, l - pad), max(0, t - pad),
                      min(im.width, r + pad), min(im.height, b + pad)))
        im.save(path)
    if also_pdf:
        pdf = str(Path(path).with_suffix(".pdf"))
        im.save(pdf, "PDF", resolution=200.0)
        print(f"wrote {pdf}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="200")
    ap.add_argument("--split", default="val")
    ap.add_argument("--track-id", type=int, default=None)
    ap.add_argument("--layout-run", required=True)
    ap.add_argument("--icp-run", required=True)
    ap.add_argument("--elev", type=float, default=18)
    ap.add_argument("--azim", type=float, default=45)
    ap.add_argument("--out", default=None)
    ap.add_argument("--separate", action="store_true")
    args = ap.parse_args()

    src = make_source("procthor", args.scene, split=args.split, gt_mesh="asset", stride=1)
    list(src)  # materialize (loads GT)
    gts = [g for g in src.gt() if getattr(g, "mesh", None) is not None]
    layout_run, icp_run = Path(args.layout_run), Path(args.icp_run)

    # --- no track-id: rank tracks by how much ICP improves centroid vs layout --
    if args.track_id is None:
        sg = json.load(open(next(layout_run.glob("*/scene_graph.json"))))
        rows = []
        for o in sg["objects"]:
            if not o.get("mesh"):
                continue
            tid = int(o["id"])
            gt = _match_gt(gts, np.asarray(o["T_world_obj"])[:3, 3])
            gc = gt.T_world_obj[:3, 3]
            lay = np.asarray(o["T_world_mesh"])[:3, 3]
            io = next((x for x in json.load(open(next(icp_run.glob("*/scene_graph.json"))))["objects"]
                       if int(x["id"]) == tid and x.get("mesh")), None)
            if io is None:
                continue
            icp = np.asarray(io["T_world_mesh"])[:3, 3]
            rows.append((tid, gt.label, float(np.linalg.norm(lay - gc)),
                         float(np.linalg.norm(icp - gc))))
        rows.sort(key=lambda r: r[3] - r[2])  # most improved (icp << layout) first
        print(f"{'tid':>4} {'label':16} {'layout_cent':>11} {'icp_cent':>9} {'delta':>7}")
        for tid, lab, lc, ic in rows[:15]:
            print(f"{tid:>4} {lab:16} {lc:>11.3f} {ic:>9.3f} {ic - lc:>+7.3f}")
        return

    tid = args.track_id
    lay = _posed_points(layout_run, tid, N)
    icp = _posed_points(icp_run, tid, N)
    if lay is None or icp is None:
        raise SystemExit(f"track {tid} missing in layout/icp run")
    gt = _match_gt(gts, lay.mean(0))
    gc = gt.T_world_obj[:3, 3]
    gpts = geo.sample_surface(gt.mesh, N, seed=1)
    L = lambda p: p - gc  # noqa: E731  (centre at GT, keep world Z-up)

    allc = np.vstack([L(lay), L(icp), L(gpts)])
    bmin, bmax = allc.min(0), allc.max(0)
    bctr, bhalf = 0.5 * (bmin + bmax), 0.5 * (bmax - bmin) * PAD

    # panel = list of (points, colour, size, alpha)
    panels = {
        "a": [(L(gpts), GREY, PT_GT, GT_ALPHA), (L(lay), RED, PT, 1.0)],
        "b": [(L(gpts), GREY, PT_GT, GT_ALPHA), (L(icp), BLUE, PT, 1.0)],
        "c": [(L(gpts), GREY, PT, 1.0)],
    }

    def _draw(axp, key):
        for P, c, s, a in panels[key]:
            axp.scatter(P[:, 0], P[:, 1], P[:, 2], s=s, c=c, alpha=a, linewidths=0)
        axp.set_xlim(bctr[0] - bhalf[0], bctr[0] + bhalf[0])
        axp.set_ylim(bctr[1] - bhalf[1], bctr[1] + bhalf[1])
        axp.set_zlim(bctr[2] - bhalf[2], bctr[2] + bhalf[2])
        axp.set_box_aspect(tuple(bhalf))
        axp.view_init(elev=args.elev, azim=args.azim)
        axp.set_axis_off()

    base = (Path(args.out) if args.out else
            Path(f"results/paper/_figs/c1_panel_s{args.scene}_t{tid}")).with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)

    print(f"track {tid} ({gt.label}) | layout_cent {np.linalg.norm(lay.mean(0)-gc):.3f} "
          f"icp_cent {np.linalg.norm(icp.mean(0)-gc):.3f}")

    if args.separate:
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
            _draw(fig.add_subplot(1, 3, k, projection="3d"), key)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.0)
        out = f"{base}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.02)
        _autocrop(out); print(f"wrote {out}")


if __name__ == "__main__":
    main()
