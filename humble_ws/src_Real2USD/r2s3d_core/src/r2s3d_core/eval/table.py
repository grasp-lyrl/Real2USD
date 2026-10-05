"""Render a comparison table from run.json files.

Usage::

    python -m r2s3d_core.eval.table results/phase0_replica_oracle results/phase0_replica_sam3d_layout
    python -m r2s3d_core.eval.table --glob 'results/phase0_*'

Always regenerated from run.json; never hand-edited.
"""

from __future__ import annotations

import argparse
import glob as _glob
import json
from pathlib import Path
from typing import List

# (metric key, column header, format, lower_is_better)
# Coworker-comparable columns use centroid matching (cd_*, tau=1m) + class_free_recall_1m;
# f1@.25 is OUR stricter OBB-IoU protocol, shown alongside (see AI-7 / metrics.py).
_COLUMNS = [
    ("cd_micro_f1", "microF1", "{:.3f}", False),   # coworker Object Micro F1 (greedy XY, label-aware)
    ("cd_micro_f1_many_to_one", "m2oF1", "{:.3f}", False),  # over-seg-tolerant any-overlap F1
    ("cd_macro_f1", "macroF1", "{:.3f}", False),   # coworker Object Macro F1
    ("cd_f1", "cdF1@1m", "{:.3f}", False),         # centroid, label-agnostic
    ("class_free_recall_1m", "cfR@1m", "{:.3f}", False),  # coworker Class-Free Geo Recall
    ("f1", "iouF1@.25", "{:.3f}", False),          # OUR stricter OBB-IoU protocol
    ("scan2cad_accuracy", "S2C-acc", "{:.3f}", False),
    ("centroid_err_median_m", "cent(m)", "{:.3f}", True),
    ("scale_err_median", "scale", "{:.3f}", True),
    ("scene_chamfer_mean_m", "chamfer(m)", "{:.3f}", True),   # scene-level pooled (coworker Chamfer)
    ("footprint_iou", "footIoU", "{:.3f}", False),            # top-down occupancy IoU (coworker Mesh row)
    ("surf_fscore@0.05", "surfF@5cm", "{:.3f}", False),       # surface-recon coverage (NOT geo recall)
]


def _load(path: Path) -> dict:
    rj = path / "run.json" if path.is_dir() else path
    with open(rj) as f:
        return json.load(f)


def render(records: List[dict]) -> str:
    header = ["method", "scenes"] + [c[1] for c in _COLUMNS]
    rows = []
    for rec in records:
        agg = rec.get("metrics", {}).get("aggregate", {})
        method = rec.get("config", {}).get("method", "?")
        scenes = ",".join(rec.get("scenes", []))
        cells = [method, scenes]
        for key, _hdr, fmt, _low in _COLUMNS:
            v = agg.get(key)
            cells.append(fmt.format(v) if isinstance(v, (int, float)) else "-")
        rows.append(cells)

    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) if rows else len(header[i])
              for i in range(len(header))]
    def fmt_row(cells):
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    lines = [fmt_row(header), "| " + " | ".join("-" * widths[i] for i in range(len(header))) + " |"]
    lines += [fmt_row(r) for r in rows]
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description="render comparison table from run.json files")
    p.add_argument("paths", nargs="*", help="run dirs or run.json files")
    p.add_argument("--glob", default=None, help="glob pattern for run dirs")
    args = p.parse_args(argv)

    paths = [Path(x) for x in args.paths]
    if args.glob:
        paths += [Path(x) for x in sorted(_glob.glob(args.glob))]
    if not paths:
        p.error("provide run paths or --glob")

    records = []
    for path in paths:
        try:
            records.append(_load(path))
        except FileNotFoundError:
            print(f"# skip (no run.json): {path}")
    print(render(records))


if __name__ == "__main__":
    main()
