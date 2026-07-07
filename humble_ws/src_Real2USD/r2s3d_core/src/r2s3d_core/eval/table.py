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
_COLUMNS = [
    ("f1", "F1@.25", "{:.3f}", False),
    ("recall@0.5", "R@.5", "{:.3f}", False),
    ("scan2cad_accuracy", "S2C-acc", "{:.3f}", False),
    ("centroid_err_median_m", "cent(m)", "{:.3f}", True),
    ("rotation_err_median_deg", "rot(deg)", "{:.1f}", True),
    ("scale_err_median", "scale", "{:.3f}", True),
    ("duplicate_rate", "dup", "{:.2f}", True),
    ("chamfer_l1_median_m", "chamfer(m)", "{:.3f}", True),
    ("fscore@0.05_mean", "F@5cm", "{:.3f}", False),
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
