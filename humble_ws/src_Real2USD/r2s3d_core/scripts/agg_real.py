"""Aggregate the real-robot (Go2, 4 scenes) registration ablation into a tidy CSV for the paper.

Reads the per-scene run.json for each registration mode and writes
`results/paper/_tables/real_registration_ablation.csv` (per-scene rows + a MEAN row per mode).
The `layout` row = SAM3D-native placement (the baseline); `scale_icp` = reprojection scale-fit + ICP;
`+gate` = with the precision gate. Clio (external baseline) is scored separately by
`scripts/score_clio_baseline.py`.

Usage:  uv run python scripts/agg_real.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

# per-scene run dir for each registration mode (naming differs by scene — pinned explicitly)
SCENES = {
    "hallway-1": {"layout": "phase0_hallway1_rs_layout", "icp": "phase0_hallway1_rs_icp",
                  "scale_icp": "phase0_hallway1_rs_scale_icp_reproj",
                  "gate": "phase0_hallway1_rs_scaleicp_gate_gentle"},
    "lounge-0": {"layout": "phase0_lounge0_rs_layout", "icp": "phase0_lounge0_rs_icp",
                 "scale_icp": "phase0_lounge0_rs_scaleicp", "gate": "phase0_lounge0_rs_scaleicp_gate"},
    "smalloffice-0": {"layout": "phase0_smalloffice0_rs_layout", "icp": "phase0_smalloffice0_rs_icp",
                      "scale_icp": "phase0_smalloffice0_rs_scaleicp",
                      "gate": "phase0_smalloffice0_rs_scaleicp_gate"},
    "smalloffice-1": {"layout": "phase0_smalloffice1_rs_layout", "icp": "phase0_smalloffice1_rs_icp",
                      "scale_icp": "phase0_smalloffice1_rs_scaleicp",
                      "gate": "phase0_smalloffice1_rs_scaleicp_gate"},
}
MODES = ["layout", "icp", "scale_icp", "gate"]
COLS = [("iou_f1", "f1"), ("recall@0.5", "recall@0.5"), ("centroid_m", "centroid_err_median_m"),
        ("rotation_deg", "rotation_err_median_deg"), ("scale_err", "scale_err_median"),
        ("cd_f1@1m", "cd_f1"), ("class_free@1m", "class_free_recall_1m"), ("n_pred", "n_pred")]


def _agg(d):
    try:
        m = json.load(open(f"results/{d}/run.json"))["metrics"]
        return m.get("aggregate", m)
    except Exception:
        return None


def main():
    rows = []
    for scene, mp in SCENES.items():
        for mode in MODES:
            a = _agg(mp[mode])
            if a is None:
                continue
            r = {"scene": scene, "mode": mode}
            for lbl, k in COLS:
                v = a.get(k)
                r[lbl] = round(v, 4) if isinstance(v, (int, float)) else ""
            rows.append(r)
    # mean per mode
    for mode in MODES:
        r = {"scene": "MEAN", "mode": mode}
        sel = [x for x in rows if x["mode"] == mode and x["scene"] != "MEAN"]
        for lbl, _ in COLS:
            vs = [x[lbl] for x in sel if isinstance(x[lbl], (int, float))]
            r[lbl] = round(float(np.mean(vs)), 4) if vs else ""
        rows.append(r)

    out = Path("results/paper/_tables/real_registration_ablation.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["scene", "mode"] + [c[0] for c in COLS]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)\n")
    print("MEAN per mode:")
    hdr = ["mode", "iou_f1", "centroid_m", "scale_err", "rotation_deg", "cd_f1@1m"]
    print("  " + "".join(f"{h:>13}" for h in hdr))
    for r in [x for x in rows if x["scene"] == "MEAN"]:
        print("  " + "".join(f"{str(r.get(h, '')):>13}" for h in hdr))


if __name__ == "__main__":
    main()
