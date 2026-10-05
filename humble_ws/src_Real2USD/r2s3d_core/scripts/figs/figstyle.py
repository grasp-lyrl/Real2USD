"""Shared plotting style for the workshop-paper figures (seaborn/matplotlib).

Import this from every ``scripts/figs/fig_*.py`` so the whole figure set reads
as one system. ALL the cross-figure knobs live here — edit them in ONE place:

  * figure widths (IEEE double-column: 1-col ~3.4in, 2-col ~7.0in)
  * font sizes (tuned so axis labels/ticks stay readable at column width)
  * palette (colour-blind-safe; series colours are named so figures agree)
  * output: writes BOTH .pdf (vector, use this in LaTeX) and .png (preview)

Per-figure knobs (axis labels, legend text, bins, which metrics) stay at the
top of each fig_*.py so you can tweak a single figure without touching others.

Run a figure with:  ``uv run --extra viz python scripts/figs/fig_<name>.py``
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# EDIT HERE — global knobs shared by every figure
# ---------------------------------------------------------------------------
# IEEE RAS double-column widths (inches). A single-column figure spans ~3.4in;
# a full-width figure spans ~7.0in. Height is per-figure (set in each script).
FIG_1COL_W = 3.4
FIG_2COL_W = 7.0

# Font sizes (points). At 3.4in column width these render ~ the caption size.
# Bump these if a printed figure looks cramped; drop them if labels collide.
FS_LABEL = 8      # axis labels
FS_TICK = 7       # tick labels
FS_LEGEND = 7     # legend entries
FS_ANNOT = 7      # in-plot annotations

# Font family. IEEE body text is Times; 'serif' keeps figures consistent with
# the caption. Set USE_TEX=True ONLY if a LaTeX toolchain + Times are installed
# (slower, but exact font match). Left off by default so this runs anywhere.
FONT_FAMILY = "serif"
USE_TEX = False

# Colour-blind-safe palette. Named series colours so, e.g., "the asset is blue"
# holds across every figure. Swap hexes here to re-theme the whole paper.
PALETTE = sns.color_palette("colorblind")
C_ASSET = PALETTE[0]      # generated asset / ours
C_CLUSTER = PALETTE[3]    # observed cluster / baseline
C_LAYOUT = PALETTE[7]     # SAM 3D-native (layout)
C_ICP = PALETTE[0]        # +ICP (ours)
C_ACCENT = PALETTE[2]     # highlight / third series

LINEWIDTH = 1.6
MARKERSIZE = 4
GRID_ALPHA = 0.35
SCATTER_ALPHA = 0.18      # raw-point scatter behind trend lines
DPI = 300

# Output dir = <repo>/paper/figs  (resolved from this file's location).
OUT_DIR = Path(__file__).resolve().parents[5] / "paper" / "figs"
# Aggregated CSVs live under r2s3d_core/results/paper/_tables.
TABLES_DIR = Path(__file__).resolve().parents[2] / "results" / "paper" / "_tables"


def setup() -> None:
    """Apply the shared style. Call once at the top of each figure script."""
    sns.set_theme(style="whitegrid", palette=PALETTE)
    mpl.rcParams.update({
        "text.usetex": USE_TEX,
        "font.family": FONT_FAMILY,
        "axes.titlesize": FS_LABEL,   # titles are unused (captions describe figs)
        "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEGEND,
        "legend.title_fontsize": FS_LEGEND,
        "lines.linewidth": LINEWIDTH,
        "lines.markersize": MARKERSIZE,
        "grid.alpha": GRID_ALPHA,
        "axes.grid": True,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "figure.dpi": DPI,
        "pdf.fonttype": 42,           # embed TrueType (editable text in the PDF)
        "ps.fonttype": 42,
    })


def save(fig, name: str) -> None:
    """Write <name>.pdf (for LaTeX) and <name>.png (preview) into paper/figs.

    NO title is drawn — figures are described in their captions.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = OUT_DIR / f"{name}.{ext}"
        fig.savefig(path, dpi=DPI)
        print(f"wrote {path}")
    plt.close(fig)
