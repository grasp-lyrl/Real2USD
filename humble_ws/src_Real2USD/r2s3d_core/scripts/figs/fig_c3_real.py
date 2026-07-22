"""Fig 3 (C3) — paired per-object scale error, SAM 3D layout -> our scale-fit (real).

This is the C3 lead result: on the 61 real objects matched under BOTH layout and
scale-fit, the reprojection scale-fit lowers scale error (median 0.74 -> 0.59,
38/61 objects, Wilcoxon p<1e-4). A paired plot shows each object as a line from
its layout error to its scale-fit error, with the medians overlaid.

Data: results/paper/_tables/paired_scale_test.csv
Columns used: scale_err_layout, scale_err_scaleicp.

Run:  uv run --extra viz python scripts/figs/fig_c3_real.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import figstyle as fs

# ---------------------------------------------------------------------------
# EDIT HERE — knobs for THIS figure only
# ---------------------------------------------------------------------------
NAME = "fig_c3_real"
FIG_W = fs.FIG_1COL_W
FIG_H = 2.35
Y_LABEL = "Scale error"
X0_LABEL = "SAM 3D layout"
X1_LABEL = "+ scale-fit (ours)"
LINE_ALPHA = 0.25            # per-object connector opacity
YLIM = (0.0, 2.0)            # clip extreme outliers; a few objects run 2-4x
JITTER = 0.03                # horizontal jitter so overlapping points separate


def main() -> None:
    fs.setup()
    df = pd.read_csv(fs.TABLES_DIR / "paired_scale_test.csv")
    a = pd.to_numeric(df["scale_err_layout"], errors="coerce")
    b = pd.to_numeric(df["scale_err_scaleicp"], errors="coerce")
    ok = a.notna() & b.notna()
    a, b = a[ok].to_numpy(), b[ok].to_numpy()
    improved = int((b < a).sum())
    print(f"{len(a)} paired objects | improved {improved}/{len(a)} | "
          f"median {np.median(a):.3f} -> {np.median(b):.3f}")

    rng = np.random.default_rng(0)
    x0 = 0.0 + rng.uniform(-JITTER, JITTER, len(a))
    x1 = 1.0 + rng.uniform(-JITTER, JITTER, len(a))

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    # per-object connectors: green if scale-fit helped, grey if not
    for xa, ya, xb, yb in zip(x0, a, x1, b):
        colour = fs.C_ASSET if yb < ya else "0.6"
        ax.plot([xa, xb], [ya, yb], color=colour, alpha=LINE_ALPHA, lw=0.7,
                zorder=1)
    ax.scatter(x0, a, s=9, color=fs.C_LAYOUT, zorder=2, edgecolor="none")
    ax.scatter(x1, b, s=9, color=fs.C_ICP, zorder=2, edgecolor="none")

    # median markers + connector
    ma, mb = np.median(a), np.median(b)
    ax.plot([0, 1], [ma, mb], color="black", lw=fs.LINEWIDTH, zorder=3,
            marker="o", markersize=fs.MARKERSIZE + 1)
    ax.annotate(f"{ma:.2f}", (0, ma), textcoords="offset points",
                xytext=(-4, 4), ha="right", fontsize=fs.FS_ANNOT)
    ax.annotate(f"{mb:.2f}", (1, mb), textcoords="offset points",
                xytext=(4, 4), ha="left", fontsize=fs.FS_ANNOT)

    ax.set_xticks([0, 1])
    ax.set_xticklabels([X0_LABEL, X1_LABEL])
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(*YLIM)
    ax.set_ylabel(Y_LABEL)
    ax.set_xlabel("")
    ax.grid(axis="x", visible=False)

    fig.tight_layout()
    fs.save(fig, NAME)


if __name__ == "__main__":
    main()
