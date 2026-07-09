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
        "full_frame": args.full_frame,
        "icp_accumulate": args.icp_accumulate,
        # object_track (Phase 2)
        "detections": args.detections,
        "reid": args.reid,
        "late_merge": args.late_merge,
        "v1_dedup": args.v1_dedup,
        "icp": args.icp,
        "debug_html": args.debug_html,
        "corrupt": {"dropout": args.det_dropout, "jitter_px": args.det_jitter,
                    "track_break": args.det_track_break, "split": args.det_split_prob},
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
        # merge any method-reported scene stats (e.g. object_track SAM3D invocations,
        # fragmentation) into the scene metrics so they land in run.json + aggregate.
        m.update(config.get("_method_stats", {}).get(scene, {}))
        per_scene[scene] = m
        print(f"[{scene}] pred={m['n_pred']} gt={m['n_gt']} "
              f"F1={m['f1']:.3f} S2C={m['scan2cad_accuracy']:.3f} "
              f"cent={m['centroid_err_median_m']:.3f}m rot={m['rotation_err_median_deg']:.1f}deg "
              f"dup={m['duplicate_rate']:.2f}"
              + (f" sam3d={m['sam3d_invocations']} tracks/gt={m.get('tracks_per_gt', float('nan')):.2f}"
                 if "sam3d_invocations" in m else ""))

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
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
        "dataset": args.source,
        "scenes": list(args.scene),
        "metrics": {"aggregate": agg, "per_scene": per_scene},
        "wall_time_s": time.time() - t0,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }

    name = args.name or f"{args.source}_{args.method}"
    out_dir = Path(args.out) if args.out else DEFAULT_RESULTS / f"{args.phase}_{name}"
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
    p.add_argument("--phase", default="phase0", help="results subdir prefix (e.g. phase2)")
    p.add_argument("--out", default=None, help="explicit output dir")
    # sam3d_layout: full frame (default) vs tight crop fed to SAM3D
    p.add_argument("--crop", dest="full_frame", action="store_false",
                   help="feed SAM3D a tight bbox crop instead of the full frame "
                        "(default full frame; crop over-predicts scale — see STATUS.md)")
    p.set_defaults(full_frame=True)
    p.add_argument("--icp-accumulate", action="store_true",
                   help="sam3d_layout_icp: fuse the object's masked depth over ALL views as "
                        "the ICP target (default single best view). Proto multi-view fusion.")
    # object_track (Phase 2): detector-driven tracks
    p.add_argument("--detections", default=None,
                   help="dir of cached DetectionSets (from `python -m r2s3d_core.detect.run`)")
    p.add_argument("--reid", dest="reid", action="store_true", default=None,
                   help="force re-ID (step-2 association) on")
    p.add_argument("--no-reid", dest="reid", action="store_false",
                   help="force re-ID off (object_track_naive default)")
    p.add_argument("--late-merge", dest="late_merge", action="store_true", default=None,
                   help="force late-merge on")
    p.add_argument("--no-late-merge", dest="late_merge", action="store_false",
                   help="force late-merge off")
    p.add_argument("--v1-dedup", action="store_true",
                   help="object_track_naive: add v1's 0.5 m same-label position suppression")
    p.add_argument("--icp", action="store_true",
                   help="object_track: refine each track against its fused multi-view cloud")
    p.add_argument("--det-dropout", type=float, default=0.0, help="corruption: drop-detection prob")
    p.add_argument("--det-jitter", type=int, default=0, help="corruption: bbox/mask jitter px")
    p.add_argument("--det-track-break", type=float, default=0.0, help="corruption: id-break prob")
    p.add_argument("--det-split-prob", type=float, default=0.0,
                   help="corruption: split one detection into two (fragmentation stress)")
    p.add_argument("--debug-html", default=None,
                   help="object_track: dir to write per-scene association/merge debug HTML")
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
