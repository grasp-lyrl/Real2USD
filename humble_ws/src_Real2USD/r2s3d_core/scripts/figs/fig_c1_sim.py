"""Fig 2 companion (C1) — registration beats SAM 3D's own placement (sim).

Grouped bar: SAM 3D-native pose (layout) vs +ICP (ours) vs +scale+ICP vs the
cluster baseline, on the strict placement metrics. Shows the C1 jump
(layout -> +ICP) on IoU-F1 and recall@0.5.

Data: results/paper/_tables/sim_per_config_mean.csv (val-10 means).

Run:  uv run --extra viz python scripts/figs/fig_c1_sim.py
"""
from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import figstyle as fs

# ---------------------------------------------------------------------------
# EDIT HERE — knobs for THIS figure only
# ---------------------------------------------------------------------------
NAME = "fig_c1_sim"
FIG_W = fs.FIG_1COL_W
FIG_H = 2.35
Y_LABEL = "Score"
# metric column -> pretty x label
METRICS = {"iou_f1": "IoU-F1", "recall@0.5": "recall@0.5"}
# config row -> (pretty legend label, colour)
CONFIGS = {
    "asset_layout_gt":  ("SAM 3D layout", fs.C_LAYOUT),
    "asset_icp_gt":     ("+ICP (ours)", fs.C_ICP),
    "asset_scaleicp_gt": ("+scale+ICP (ours)", fs.C_ACCENT),
    "cluster_gt":       ("cluster", fs.C_CLUSTER),
}
YLIM = (0.0, 0.6)


def main() -> None:
    fs.setup()
    df = pd.read_csv(fs.TABLES_DIR / "sim_per_config_mean.csv")
    df = df[df["config"].isin(CONFIGS)].copy()

    long = df.melt(id_vars="config", value_vars=list(METRICS),
                   var_name="metric", value_name="value")
    long["metric"] = long["metric"].map(METRICS)
    long["config"] = long["config"].map(lambda c: CONFIGS[c][0])

    order = [CONFIGS[c][0] for c in CONFIGS]
    palette = {CONFIGS[c][0]: CONFIGS[c][1] for c in CONFIGS}

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    sns.barplot(data=long, x="metric", y="value", hue="config",
                hue_order=order, palette=palette, ax=ax, edgecolor="0.2",
                linewidth=0.4)

    ax.set_xlabel("")
    ax.set_ylabel(Y_LABEL)
    ax.set_ylim(*YLIM)
    ax.legend(title=None, loc="upper left", frameon=True, framealpha=0.9,
              ncol=2, columnspacing=0.8, handletextpad=0.4)

    fig.tight_layout()
    fs.save(fig, NAME)


if __name__ == "__main__":
    main()
