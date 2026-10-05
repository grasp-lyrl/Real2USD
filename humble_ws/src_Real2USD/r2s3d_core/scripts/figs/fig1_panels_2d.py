"""Figure-1 teaser: 2D pipeline panels as individual Type-42 PDFs.

Threads one hero frame (lounge-0, frame 52 = the featured chair's best view)
through the sensing + front-end stages:
  1_rgb          RealSense RGB
  2_depth        aligned depth (colorised)
  3_frontend     YOLOE detect+segment overlay (all dets on the frame)
  4_sam3d_input  the masked object crop actually fed to SAM 3D

All PDFs embed text as Type-42 (TrueType), not bitmap Type-3.
  uv run --extra rosbag python scripts/figs/fig1_panels_2d.py
"""
from __future__ import annotations
import sys, os
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from r2s3d_core.data.registry import make_source

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/paper/_figs/fig1"
OUT.mkdir(parents=True, exist_ok=True)

SCENE = "lounge-0"
HERO_FRAME = 360  # featured object 26 (swivel chair) best frame; tight registration
JOB = ROOT / "results/phase0_lounge0_rs_scaleicp/sam3d_queue/output/realsense_lounge-0_t26_full"
DETS = ROOT / "results/detections/gt/lounge-0/detections.npz"
DPI = 300

# colourblind-safe qualitative palette for instances
PAL = [(0.902, 0.624, 0.0), (0.337, 0.706, 0.914), (0.0, 0.620, 0.451),
       (0.835, 0.369, 0.0), (0.800, 0.475, 0.655), (0.941, 0.894, 0.259)]


def _save_pdf(fig, name):
    p = OUT / name
    fig.savefig(p, dpi=DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print("wrote", p)


def _fig_for(img):
    h, w = img.shape[:2]
    fig = plt.figure(figsize=(w / DPI, h / DPI), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    return fig, ax


def get_hero_frame():
    src = make_source("realsense", SCENE, stride=1)
    for i, f in enumerate(src):
        if f.frame_id == HERO_FRAME:
            return f
    raise RuntimeError("hero frame not found")


def panel_rgb(f):
    fig, ax = _fig_for(f.rgb); ax.imshow(f.rgb)
    _save_pdf(fig, "fig1_1_rgb.pdf")


def panel_depth(f):
    d = f.depth.astype(np.float32).copy()
    valid = d > 0
    dm = np.ma.masked_where(~valid, d)
    fig, ax = _fig_for(f.depth)
    cmap = plt.cm.turbo.copy(); cmap.set_bad("white")
    ax.imshow(dm, cmap=cmap, vmin=float(d[valid].min()), vmax=float(np.percentile(d[valid], 99)))
    _save_pdf(fig, "fig1_2_depth.pdf")


def _unpack_mask(packed, h, w):
    return np.unpackbits(packed)[: h * w].reshape(h, w).astype(bool)


def panel_frontend(f):
    z = np.load(DETS, allow_pickle=True)
    sel = np.where(z["frame_ids"] == HERO_FRAME)[0]
    h, w = z["hw"]
    fig, ax = _fig_for(f.rgb); ax.imshow(f.rgb)
    for k, idx in enumerate(sel):
        col = PAL[k % len(PAL)]
        m = _unpack_mask(z["masks_packed"][idx], h, w)
        overlay = np.zeros((h, w, 4)); overlay[m] = (*col, 0.45)
        ax.imshow(overlay)
        # contour for a crisp instance edge
        ax.contour(m, levels=[0.5], colors=[col], linewidths=1.2)
        x0, y0, x1, y1 = z["boxes"][idx]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                               edgecolor=col, linewidth=1.6))
        lbl = f"{z['labels'][idx]} {z['scores'][idx]:.2f}"
        ax.text(x0 + 2, max(y0 - 4, 8), lbl, fontsize=7, color="white",
                va="bottom", ha="left",
                bbox=dict(boxstyle="square,pad=0.15", fc=col, ec="none"))
    _save_pdf(fig, "fig1_3_frontend.pdf")


def panel_sam3d_input():
    """Full frame with the swivel-chair mask highlighted (SAM 3D takes the FULL
    image + the object mask, not a crop -- 'full frame beats crop')."""
    rgb = np.array(Image.open(JOB / "rgb.png").convert("RGB"))
    mask = np.array(Image.open(JOB / "mask.png").convert("L")) > 0
    # full-colour frame, object marked only by its outline (no dimming)
    fig, ax = _fig_for(rgb); ax.imshow(rgb)
    ax.contour(mask, levels=[0.5], colors=[(0.902, 0.624, 0.0)], linewidths=1.8)  # object edge
    _save_pdf(fig, "fig1_4_sam3d_input.pdf")


def main():
    f = get_hero_frame()
    print(f"hero frame {f.frame_id}: rgb {f.rgb.shape} depth {f.depth.shape}")
    panel_rgb(f)
    panel_depth(f)
    panel_frontend(f)
    panel_sam3d_input()


if __name__ == "__main__":
    main()
