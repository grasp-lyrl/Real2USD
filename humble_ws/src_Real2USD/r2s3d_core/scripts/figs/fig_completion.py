"""Fig 4 (HEADLINE, C2) — unobserved-surface reconstruction vs observed coverage.

Story: the generated asset reconstructs the surface the robot never saw about
as well regardless of how little it observed (flat, low error), while the
observed point cluster degrades sharply as coverage drops. Two lines that
diverge at low coverage.

Data: results/paper/_tables/shape_completion_s*.csv (10 scenes, 212 objects).
Columns used: coverage, asset_unobs_recon, cluster_unobs_recon.

Run:  uv run --extra viz python scripts/figs/fig_completion.py
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import figstyle as fs

# ---------------------------------------------------------------------------
# EDIT HERE — knobs for THIS figure only
# ---------------------------------------------------------------------------
NAME = "fig_completion"
FIG_W = fs.FIG_1COL_W
FIG_H = 2.35                       # inches
X_LABEL = "Observed surface coverage"
Y_LABEL = "Unobserved-surface error (m)"
ASSET_LABEL = "Generated asset (ours)"
CLUSTER_LABEL = "Observed cluster"
COV_BINS = np.arange(0.0, 1.0001, 0.1)   # coverage bin edges; coarsen if sparse
SHOW_SCATTER = True                       # raw points behind the trend lines
ERRORBAR = ("ci", 95)                     # seaborn aggregation band; e.g. ("se",1)
XLIM = (0.0, 1.0)
YLIM = None                               # e.g. (0, 0.30) to pin the axis


def load() -> pd.DataFrame:
    files = sorted(glob.glob(str(fs.TABLES_DIR / "shape_completion_s*.csv")))
    if not files:
        raise SystemExit(f"no shape_completion CSVs under {fs.TABLES_DIR}")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    for c in ("coverage", "asset_unobs_recon", "cluster_unobs_recon"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["coverage", "asset_unobs_recon", "cluster_unobs_recon"])
    print(f"loaded {len(df)} objects across {len(files)} scenes")
    return df


def to_long(df: pd.DataFrame) -> pd.DataFrame:
    centers = 0.5 * (COV_BINS[:-1] + COV_BINS[1:])
    df = df.copy()
    df["cov_bin"] = pd.cut(df["coverage"], bins=COV_BINS, labels=centers,
                           include_lowest=True).astype(float)
    long = df.melt(
        id_vars=["coverage", "cov_bin"],
        value_vars=["asset_unobs_recon", "cluster_unobs_recon"],
        var_name="series", value_name="recon",
    )
    long["series"] = long["series"].map({
        "asset_unobs_recon": ASSET_LABEL,
        "cluster_unobs_recon": CLUSTER_LABEL,
    })
    return long


def main() -> None:
    fs.setup()
    df = load()
    long = to_long(df)
    palette = {ASSET_LABEL: fs.C_ASSET, CLUSTER_LABEL: fs.C_CLUSTER}

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    if SHOW_SCATTER:
        import seaborn as sns
        sns.scatterplot(
            data=long, x="coverage", y="recon", hue="series", palette=palette,
            s=8, alpha=fs.SCATTER_ALPHA, edgecolor=None, legend=False, ax=ax,
        )

    import seaborn as sns
    sns.lineplot(
        data=long, x="cov_bin", y="recon", hue="series", palette=palette,
        errorbar=ERRORBAR, marker="o", markersize=fs.MARKERSIZE, ax=ax,
    )

    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.set_xlim(*XLIM)
    if YLIM:
        ax.set_ylim(*YLIM)
    ax.legend(title=None, loc="upper right", frameon=True, framealpha=0.9)

    fig.tight_layout()
    fs.save(fig, NAME)


if __name__ == "__main__":
    main()
