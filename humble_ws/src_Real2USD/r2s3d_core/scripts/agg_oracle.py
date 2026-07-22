"""Aggregate the val-10 ORACLE campaign (GT-mask sam3d_layout) into a mean row per
registration variant, and write results/paper/_tables/oracle_val10_mean.csv.

The oracle = perception UPPER BOUND (perfect detection). Compare against the
detector-driven Table 1 (sim_per_config_mean.csv) to size the perception gap.

  uv run python scripts/agg_oracle.py    # (pure-python; no extras needed)
"""
from __future__ import annotations
import csv, json
from pathlib import Path

IDS = [137, 200, 428, 434, 534, 569, 573, 683, 771, 912]
KEYS = ["f1", "recall@0.5", "scan2cad_accuracy", "centroid_err_median_m",
        "rotation_err_median_deg", "scale_err_median", "cd_f1", "class_free_recall_1m"]
VARIANTS = [("layout", "oracle_layout"), ("icp", "oracle_icp"), ("scale_icp", "oracle_scaleicp")]
ROOT = Path("results/paper/sim")
OUT = Path("results/paper/_tables/oracle_val10_mean.csv")


def _mean(pfx):
    acc = {k: [] for k in KEYS}; n = 0
    for i in IDS:
        p = ROOT / f"{pfx}_gt_s{i}" / "run.json"
        if not p.exists():
            continue
        a = json.load(open(p))["metrics"]["aggregate"]; n += 1
        for k in KEYS:
            v = a.get(k)
            if isinstance(v, (int, float)) and v == v:
                acc[k].append(v)
    return n, {k: (sum(v) / len(v) if v else float("nan")) for k, v in acc.items()}


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, pfx in VARIANTS:
        n, m = _mean(pfx)
        rows.append({"registration": name, "n": n, **{k: round(m[k], 4) for k in KEYS}})
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["registration", "n", *KEYS]); w.writeheader(); w.writerows(rows)
    print(f"wrote {OUT}")
    hdr = ["f1", "rec@.5", "s2c", "cent", "rot", "scale", "cd_f1", "clsfree"]
    print(f"{'oracle val-10 (mean)':22}" + "".join(f"{h:>9}" for h in hdr))
    for r in rows:
        lbl = f"{r['registration']} (n={r['n']})"
        print(f"{lbl:22}" + "".join(f"{r[k]:>9.3f}" for k in KEYS))


if __name__ == "__main__":
    main()
