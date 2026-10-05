"""Stage-timing probe for the paper's runtime sentence (real Go2 scene, warm caches).

Measures, on a real scene (default hallway-1, stride 2 — the deployed real recipe
``scale_icp`` + reproj scale source):

  1. YOLOE detection per frame (gt-vocab prompt, GPU, warmup excluded)
  2. tracker (associate + fuse) per frame, over the cached DetectionSet
  3. placement per object: full ``object_track._run`` (cached detections + cached
     SAM3D meshes) minus the separately-timed frame-decode and tracker stages —
     i.e. mesh load + layout + reproj scale-fit + ICP, as deployed
  4. SAM3D generation per object: median gap between consecutive ``object.glb``
     mtimes in a continuously-drained queue (the big val-10 asset queue), which
     measures worker compute time per job

Usage (cwd r2s3d_core):
  uv run --extra detector --extra registration --extra mesh --extra procthor \
      python scripts/figs/timing_probe.py [--scene hallway-1] [--det-frames 40]

Writes results/paper/_tables/timing_probe.json and prints a summary.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def time_sam3d_queue(queue_out: Path, max_gap_s: float = 120.0) -> dict:
    """Per-job generation time from consecutive output mtimes in a drained queue."""
    mtimes = sorted(p.stat().st_mtime for p in queue_out.glob("*/object.glb"))
    gaps = np.diff(mtimes)
    busy = gaps[(gaps > 1.0) & (gaps < max_gap_s)]  # drop idle gaps between batches
    return {"n_jobs": len(mtimes), "n_busy_gaps": int(busy.size),
            "median_s": float(np.median(busy)), "p10_s": float(np.percentile(busy, 10)),
            "p90_s": float(np.percentile(busy, 90))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="hallway-1")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--det-frames", type=int, default=40)
    ap.add_argument("--detections", default="results/detections/gt")
    ap.add_argument("--sam3d-queue", default="results/phase0_hallway1_rs_icp/sam3d_queue")
    ap.add_argument("--gen-queue-out", default="results/paper/sim/_assetq/output",
                    help="continuously-drained queue used for per-job generation time")
    args = ap.parse_args()

    from r2s3d_core.data.registry import make_source

    out = {"scene": args.scene, "stride": args.stride}

    # -- frames (decode cost, measured so it can be subtracted) --------------------
    src = make_source("realsense", args.scene, stride=args.stride)
    t0 = time.perf_counter()
    frames = list(src)
    t_frames_cold = time.perf_counter() - t0
    # _run re-decodes with a hot OS page cache; subtract the WARM decode, not the cold one
    src_w = make_source("realsense", args.scene, stride=args.stride)
    t0 = time.perf_counter()
    _ = list(src_w)
    t_frames = time.perf_counter() - t0
    out["n_frames"] = len(frames)
    out["frame_decode_s_per_frame_cold"] = t_frames_cold / max(1, len(frames))
    out["frame_decode_s_per_frame_warm"] = t_frames / max(1, len(frames))

    # -- 1. YOLOE per frame ---------------------------------------------------------
    import cv2
    from r2s3d_core.detect.run import _gt_vocab
    from r2s3d_core.detect.yoloe import _load_model
    vocab = _gt_vocab(src.gt())
    model = _load_model("gt", None, vocab, None)
    n_det = min(args.det_frames, len(frames))
    dts = []
    for i, fr in enumerate(frames[:n_det]):
        bgr = cv2.cvtColor(fr.rgb, cv2.COLOR_RGB2BGR)
        t0 = time.perf_counter()
        model.track(bgr, persist=True, conf=0.25, iou=0.5, imgsz=1024,
                    retina_masks=True, tracker="botsort.yaml", verbose=False)
        dts.append(time.perf_counter() - t0)
    dts = np.array(dts[3:])  # drop warmup
    out["yoloe_s_per_frame"] = {"median": float(np.median(dts)),
                                "mean": float(dts.mean()), "n": int(dts.size)}

    # -- 2. tracker per frame (cached detections) ------------------------------------
    from r2s3d_core.detect.cache import DetectionSet
    from r2s3d_core.tracks.tracker import run_tracker
    ds = DetectionSet.load(Path(args.detections) / args.scene)
    tcfg = {"reid": True, "late_merge": True}
    t0 = time.perf_counter()
    tracks = run_tracker(frames, ds.by_frame(), tcfg)
    t_tracker = time.perf_counter() - t0
    out["tracker_s_per_frame"] = t_tracker / max(1, len(frames))
    out["n_tracks"] = len(tracks)

    # -- 3. placement per object (full _run minus frames+tracker; cached meshes) ----
    from r2s3d_core.baselines.object_track import _run
    config = {"detections": args.detections, "sam3d_queue": args.sam3d_queue,
              "registration": "scale_icp", "scale_source": "reproj",
              "scale_icp_iters": 5, "node_payload": "asset", "full_frame": True,
              "reid": True, "late_merge": True, "source": "realsense",
              "corrupt": {}}
    src2 = make_source("realsense", args.scene, stride=args.stride)
    gt = src2.gt()
    t0 = time.perf_counter()
    preds = _run(src2, gt, config, reid=True, late_merge=True, registration="scale_icp")
    t_run = time.perf_counter() - t0
    n_obj = len(preds)
    placement_total = max(0.0, t_run - t_frames - t_tracker)
    out["placement_s_per_object"] = placement_total / max(1, n_obj)
    out["n_objects_placed"] = n_obj
    out["run_total_s"] = t_run

    # -- 4. SAM3D generation per object (queue drain mtimes) -------------------------
    out["sam3d_generation"] = time_sam3d_queue(Path(args.gen_queue_out))

    dst = Path("results/paper/_tables/timing_probe.json")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
