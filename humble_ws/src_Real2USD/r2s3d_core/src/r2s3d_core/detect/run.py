"""Detector caching step CLI.

Runs YOLOE over one or more scenes and writes a :class:`DetectionSet` per scene under
``<out>/<scene>/`` for the ROS-free tracker/eval to consume. Heavy (torch); the rest
of the pipeline never imports torch.

    uv run python -m r2s3d_core.detect.run --source replica --scene room0 \
        --prompt gt --out results/detections

Prompt modes: ``gt`` (scene GT label vocabulary), ``generic`` (fixed indoor list),
``pf`` (prompt-free). ``--diagnose`` prints a GT-object detection-recall estimate (the
"prompting study" headline: what fraction of GT objects the detector even finds).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from ..data.registry import make_source

PKG_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = PKG_ROOT / "results" / "detections"


def _gt_vocab(gt) -> list:
    return sorted({(g.label or "").strip().lower() for g in gt if g.label})


def _diagnose(source, ds, iou_thr: float = 0.25) -> dict:
    """GT-object detection recall + median mask IoU, using each GT's best (largest,
    non-border) view. Reuses the sam3d_layout mask renderer / view selector."""
    from ..baselines.sam3d_layout import render_instance_mask, select_best_view
    from ..tracks.fusion import mask_iou

    gt = source.gt()
    frames = list(source)
    by_frame = ds.by_frame()
    fidx = {int(f.frame_id): i for i, f in enumerate(frames)}
    # GT mask source must match what the ceiling pipeline uses: prefer the dataset's TRUE
    # instance masks (occlusion-aware, RGB-aligned, e.g. ProcTHOR native seg) over a
    # mesh-projected silhouette. On ProcTHOR the GT meshes are coarse/offset from the
    # rendered RGB, so silhouettes barely overlap the detector's RGB-aligned masks and
    # recall collapses to ~0 — comparing detector masks to native GT masks is the honest
    # apples-to-apples measure. Mirror sam3d_layout._run's selection.
    if getattr(source, "native_masks", False) and hasattr(source, "native_mask"):
        gt_mask_fn = lambda g, fr: source.native_mask(fr.frame_id, g.instance_id)
    else:
        gt_mask_fn = lambda g, fr: render_instance_mask(g.mesh, fr)
    hits, ious, frag = 0, [], []
    for g in gt:
        vi = select_best_view(g, frames, gt_mask_fn)
        if vi is None:
            continue
        fr = frames[vi]
        gm = gt_mask_fn(g, fr)
        if gm is None:
            continue
        gm = gm > 0
        best, n_overlap = 0.0, 0
        for det in by_frame.get(int(fr.frame_id), []):
            iou = mask_iou(gm, det.mask)
            best = max(best, iou)
            if iou > iou_thr:
                n_overlap += 1
        if best > iou_thr:
            hits += 1
            ious.append(best)
            frag.append(n_overlap)
    n = len(gt)
    return {
        "gt_objects": n,
        "detection_recall": hits / n if n else float("nan"),
        "median_best_mask_iou": float(np.median(ious)) if ious else float("nan"),
        "mean_detections_per_found_gt": float(np.mean(frag)) if frag else float("nan"),
    }


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="YOLOE detector caching step (Phase 2)")
    p.add_argument("--source", default="replica")
    p.add_argument("--scene", nargs="+", required=True)
    p.add_argument("--data-root", default=None)
    p.add_argument("--stride", type=int, default=20)
    p.add_argument("--prompt", choices=["gt", "generic", "pf"], default="gt")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--weights", default=None, help="override YOLOE weights path")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--device", default=None, help="e.g. cuda:0 or cpu (default auto)")
    p.add_argument("--diagnose", action="store_true",
                   help="print GT detection-recall / mask-IoU (the prompting study)")
    args = p.parse_args(argv)

    from .yoloe import run_detector  # lazy: needs the detector extra

    out = Path(args.out) / args.prompt
    for scene in args.scene:
        src = make_source(args.source, scene, root=args.data_root, stride=args.stride)
        vocab = _gt_vocab(src.gt()) if args.prompt == "gt" else None
        if args.prompt == "gt" and not vocab:
            print(f"[{scene}] no GT vocabulary available; skipping"); continue
        ds = run_detector(src, prompt=args.prompt, vocab=vocab, weights=args.weights,
                          conf=args.conf, iou=args.iou, device=args.device)
        d = ds.save(out)
        labels = {}
        for det in ds.detections:
            labels[det.label] = labels.get(det.label, 0) + 1
        top = sorted(labels.items(), key=lambda kv: -kv[1])[:8]
        print(f"[{scene}] {len(ds)} detections, {len(labels)} labels -> {d}")
        print(f"          top labels: {top}")
        if args.diagnose:
            diag = _diagnose(src, ds)
            print(f"          diagnose: {diag}")


if __name__ == "__main__":
    main()
