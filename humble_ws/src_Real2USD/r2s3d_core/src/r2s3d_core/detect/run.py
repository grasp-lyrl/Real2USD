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


# A GT object counts as "visible" in a frame only if its native mask covers at least this
# fraction of the image -- a slice of a few border pixels is not a view a detector could be
# expected to fire on, and counting it would deflate detector recall. ~0.1% of a 640x480 image
# is ~300 px (roughly a 17x17 blob).
_MIN_VIS_AREA_FRAC = 0.001


def _diagnose(source, ds, iou_thr: float = 0.25, min_vis_area: float = _MIN_VIS_AREA_FRAC) -> dict:
    """GT-object detection recall + median mask IoU.

    Reports TWO recalls (see docs -- perception robustness crux):
      * ``best_view_recall`` -- object found in its single best (largest, non-border) view.
        A conservative per-view floor (the old ``detection_recall``, kept under that alias).
      * ``multi_view_recall`` -- object found in ANY frame where it is visible (>= ``min_vis_area``
        of the image). This is the honest ceiling for a multi-view tracker, which fuses
        detections across the whole trajectory rather than trusting one view.
    Decomposed against trajectory coverage so detector recall is not blamed for objects the
    camera never actually sees:
      * ``trajectory_coverage`` -- fraction of GT visible in >= 1 qualifying frame.
      * ``multi_view_recall_given_visible`` -- hits / visible (pure detector recall among the
        objects the trajectory reaches; ``multi_view_recall = coverage * this``).
    """
    from ..baselines.sam3d_layout import _view_score, render_instance_mask, select_best_view
    from ..tracks.fusion import mask_iou

    gt = source.gt()
    frames = list(source)
    by_frame = ds.by_frame()
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

    # ---- best-view recall (single largest non-border view) ----
    bv_hits, bv_ious, bv_frag = 0, [], []
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
            bv_hits += 1
            bv_ious.append(best)
            bv_frag.append(n_overlap)

    # ---- multi-view recall (found in ANY qualifying frame) ----
    mv_hits, n_visible, mv_ious = 0, 0, []
    for g in gt:
        visible = False
        best_over_traj = 0.0
        for fr in frames:
            gm = gt_mask_fn(g, fr)
            if gm is None:
                continue
            area, _border = _view_score(gm)
            if area < min_vis_area:
                continue
            visible = True
            gmb = gm > 0
            for det in by_frame.get(int(fr.frame_id), []):
                best_over_traj = max(best_over_traj, mask_iou(gmb, det.mask))
        if visible:
            n_visible += 1
            if best_over_traj > iou_thr:
                mv_hits += 1
                mv_ious.append(best_over_traj)

    n = len(gt)
    return {
        "gt_objects": n,
        # multi-view (the ceiling that bounds the tracker) + its coverage decomposition
        "multi_view_recall": mv_hits / n if n else float("nan"),
        "trajectory_coverage": n_visible / n if n else float("nan"),
        "multi_view_recall_given_visible": mv_hits / n_visible if n_visible else float("nan"),
        "median_multi_view_mask_iou": float(np.median(mv_ious)) if mv_ious else float("nan"),
        # best-view (conservative per-view floor); detection_recall kept as a back-compat alias
        "best_view_recall": bv_hits / n if n else float("nan"),
        "detection_recall": bv_hits / n if n else float("nan"),
        "median_best_mask_iou": float(np.median(bv_ious)) if bv_ious else float("nan"),
        "mean_detections_per_found_gt": float(np.mean(bv_frag)) if bv_frag else float("nan"),
    }


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="YOLOE detector caching step (Phase 2)")
    p.add_argument("--source", default="replica")
    p.add_argument("--scene", nargs="+", required=True)
    p.add_argument("--data-root", default=None)
    p.add_argument("--split", default=None,
                   help="dataset split (procthor: a scene id is a DIFFERENT house per split; "
                        "default = source default = val). Pass --split train for legacy runs.")
    p.add_argument("--stride", type=int, default=20)
    p.add_argument("--prompt", choices=["gt", "generic", "pf"], default="gt")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--weights", default=None, help="override YOLOE weights path")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--device", default=None, help="e.g. cuda:0 or cpu (default auto)")
    p.add_argument("--diagnose", action="store_true",
                   help="print GT detection-recall / mask-IoU (the prompting study)")
    p.add_argument("--backend", choices=["yoloe", "sam3"], default="yoloe",
                   help="detector backend: yoloe (default) or sam3 (transformers Sam3, gt-vocab)")
    args = p.parse_args(argv)

    if args.backend == "sam3":
        from .sam3 import run_detector  # lazy: needs the sam3 extra (transformers)
    else:
        from .yoloe import run_detector  # lazy: needs the detector extra

    out = Path(args.out) / args.prompt
    src_kwargs = {"stride": args.stride}
    if args.split is not None:
        src_kwargs["split"] = args.split
    for scene in args.scene:
        src = make_source(args.source, scene, root=args.data_root, **src_kwargs)
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
