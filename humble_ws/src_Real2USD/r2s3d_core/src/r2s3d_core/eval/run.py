"""Evaluation runner.

Usage::

    python -m r2s3d_core.eval.run --source replica --scene room0 --method sam3d_layout

Runs a method over one or more scenes, computes the Phase-0 metric bundle against
GT, and writes ``results/<phase>_<name>/run.json`` with full provenance. Plots and
tables are always regenerated from run.json, never hand-edited.
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import time
from pathlib import Path
from typing import List

import numpy as np

from ..baselines import AVAILABLE, get_method
from ..data.registry import make_source
from ..eval.metrics import SceneObject, evaluate, gt_to_scene_object

PKG_ROOT = Path(__file__).resolve().parents[3]  # r2s3d_core/
DEFAULT_RESULTS = PKG_ROOT / "results"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PKG_ROOT), stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serializable: {type(o)}")


def run(args: argparse.Namespace) -> Path:
    method_fn = get_method(args.method)
    config = {
        "method": args.method,
        "source": args.source,
        "stride": args.stride,
        "iou_threshold": args.iou_threshold,
        "compute_geometry": not args.no_geometry,
        "surface_points": args.surface_points,
        "seed": args.seed,
        "trans_noise_m": args.trans_noise_m,
        "rot_noise_deg": args.rot_noise_deg,
        "scale_noise": args.scale_noise,
    }

    per_scene = {}
    t0 = time.time()
    for scene in args.scene:
        src = make_source(args.source, scene, root=args.data_root, stride=args.stride)
        gt = src.gt()
        if not gt:
            print(f"[{scene}] WARNING: no GT available; skipping (need semantic assets).")
            per_scene[scene] = {"error": "no_gt"}
            continue
        preds: List[SceneObject] = method_fn(src, gt, config)
        gts = [gt_to_scene_object(g) for g in gt]
        m = evaluate(preds, gts, iou_threshold=args.iou_threshold,
                     compute_geometry=config["compute_geometry"],
                     surface_points=args.surface_points)
        per_scene[scene] = m
        print(f"[{scene}] pred={m['n_pred']} gt={m['n_gt']} "
              f"F1={m['f1']:.3f} S2C={m['scan2cad_accuracy']:.3f} "
              f"cent={m['centroid_err_median_m']:.3f}m rot={m['rotation_err_median_deg']:.1f}deg "
              f"dup={m['duplicate_rate']:.2f}")

    # aggregate across scenes (mean of per-scene metrics that are scalar and finite)
    agg = {}
    scene_metrics = [m for m in per_scene.values() if "error" not in m]
    if scene_metrics:
        for k in scene_metrics[0]:
            vals = [m[k] for m in scene_metrics if isinstance(m.get(k), (int, float))]
            vals = [v for v in vals if np.isfinite(v)]
            if vals:
                agg[k] = float(np.mean(vals))

    record = {
        "git_sha": _git_sha(),
        "config": config,
        "dataset": args.source,
        "scenes": list(args.scene),
        "metrics": {"aggregate": agg, "per_scene": per_scene},
        "wall_time_s": time.time() - t0,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }

    name = args.name or f"{args.source}_{args.method}"
    out_dir = Path(args.out) if args.out else DEFAULT_RESULTS / f"phase0_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "run.json"
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2, default=_json_default)
    print(f"\nwrote {out_path}")
    return out_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="R2S3D v2 evaluation runner")
    p.add_argument("--source", default="replica", help="SequenceSource backend")
    p.add_argument("--scene", nargs="+", required=True, help="scene id(s), e.g. room0")
    p.add_argument("--method", required=True, choices=AVAILABLE)
    p.add_argument("--data-root", default=None, help="override dataset root")
    p.add_argument("--stride", type=int, default=20, help="use every Nth frame")
    p.add_argument("--iou-threshold", type=float, default=0.25)
    p.add_argument("--no-geometry", action="store_true", help="skip Chamfer/F-score (faster)")
    p.add_argument("--surface-points", type=int, default=10000)
    p.add_argument("--name", default=None, help="results subdir name")
    p.add_argument("--out", default=None, help="explicit output dir")
    # oracle_noisy knobs
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--trans-noise-m", type=float, default=0.05)
    p.add_argument("--rot-noise-deg", type=float, default=5.0)
    p.add_argument("--scale-noise", type=float, default=0.05)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
