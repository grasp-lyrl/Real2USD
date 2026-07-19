"""Aggregate paper run.json files into tidy CSVs (docs/EXPERIMENT_MATRIX.md).

Regenerates the paper's sim tables from the per-run `run.json` aggregates — never hand-edit
tables, run this. Parses `results/paper/sim/<config>_s<id>/run.json`, emits a per-run CSV and a
per-config val-10 mean, into `results/paper/_tables/`.

Usage:  uv run python scripts/agg_paper.py
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path("results/paper/sim")
OUT = Path("results/paper/_tables")
# metrics we table on (label -> run.json aggregate key)
COLS = [
    ("n_pred", "n_pred"), ("iou_f1", "f1"), ("recall@0.5", "recall@0.5"),
    ("scan2cad", "scan2cad_accuracy"), ("centroid_m", "centroid_err_median_m"),
    ("scale_err", "scale_err_median"), ("chamfer_m", "scene_chamfer_mean_m"),
    ("footprint_iou", "footprint_iou"), ("cd_f1@1m", "cd_f1"),
    ("cd_micro_f1", "cd_micro_f1"), ("class_free@1m", "class_free_recall_1m"),
]
_DIR = re.compile(r"^(?P<config>.+)_s(?P<scene>\d+)$")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for rj in sorted(ROOT.glob("*/run.json")):
        m = _DIR.match(rj.parent.name)
        if not m:
            continue
        agg = json.load(open(rj)).get("metrics", {}).get("aggregate", {})
        row = {"config": m["config"], "scene": m["scene"]}
        for label, key in COLS:
            v = agg.get(key)
            row[label] = round(v, 4) if isinstance(v, (int, float)) else ""
        rows.append(row)

    if not rows:
        print("no paper/sim runs found yet")
        return

    per_run = OUT / "sim_per_run.csv"
    fields = ["config", "scene"] + [c[0] for c in COLS]
    with open(per_run, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # per-config mean over scenes (val-10 aggregate)
    means = {}
    for r in rows:
        means.setdefault(r["config"], []).append(r)
    mean_rows = []
    for cfg, rs in sorted(means.items()):
        mr = {"config": cfg, "scene": f"mean(n={len(rs)})"}
        for label, _ in COLS:
            vals = [x[label] for x in rs if isinstance(x[label], (int, float))]
            mr[label] = round(float(np.mean(vals)), 4) if vals else ""
        mean_rows.append(mr)
    per_cfg = OUT / "sim_per_config_mean.csv"
    with open(per_cfg, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(mean_rows)

    print(f"wrote {per_run} ({len(rows)} runs) and {per_cfg} ({len(mean_rows)} configs)")
    print("\nper-config val-N mean:")
    hdr = ["config", "scene", "n_pred", "iou_f1", "footprint_iou", "scale_err", "cd_f1@1m"]
    print("  " + "".join(f"{h:>15s}" for h in hdr))
    for mr in mean_rows:
        print("  " + "".join(f"{str(mr.get(h, '')):>15s}" for h in hdr))


if __name__ == "__main__":
    main()
