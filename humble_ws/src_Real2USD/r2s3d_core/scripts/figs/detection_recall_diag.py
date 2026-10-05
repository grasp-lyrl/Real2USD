"""Detection-vs-association recall diagnosis (ProcTHOR s200).

Is the pipeline's recall (tracks/gt ~0.78) limited by the DETECTOR (YOLOE never
sees the object) or by ASSOCIATION/maturation (it's detected but the track
fragments / is rejected)? For each GT instance, check whether ANY frame has a
YOLOE detection whose mask overlaps the object's native GT mask (IoU > TAU).

  uv run --extra procthor python scripts/figs/detection_recall_diag.py
"""
from __future__ import annotations
import numpy as np
from r2s3d_core.data.registry import make_source

SCENE, SPLIT, TAU = "200", "val", 0.3
DET = "results/detections/procthor/gt/200/detections.npz"


def main():
    z = np.load(DET, allow_pickle=True)
    H, W = z["hw"]
    fids = z["frame_ids"]
    # per-frame list of detection masks
    dets_by_frame = {}
    for i in range(len(fids)):
        m = np.unpackbits(z["masks_packed"][i])[: H * W].reshape(H, W).astype(bool)
        dets_by_frame.setdefault(int(fids[i]), []).append(m)

    src = make_source("procthor", SCENE, split=SPLIT, gt_mesh="box", stride=1)
    gts = src.gt() or []
    gt_ids = [g.instance_id for g in gts]
    detected = set()
    frames_seen = 0
    for f in src:
        fid = f.frame_id
        dms = dets_by_frame.get(fid)
        if not dms:
            continue
        frames_seen += 1
        for g in gts:
            if g.instance_id in detected:
                continue
            gm = src.native_mask(fid, g.instance_id)
            if gm is None:
                continue
            gmb = np.asarray(gm) > 0
            ga = int(gmb.sum())
            if ga < 50:
                continue
            for dm in dms:
                inter = int((dm & gmb).sum())
                if inter == 0:
                    continue
                iou = inter / float((dm | gmb).sum())
                if iou > TAU:
                    detected.add(g.instance_id)
                    break
    n = len(gts)
    dr = len(detected) / n if n else 0.0
    print(f"\n=== ProcTHOR s200 detection-recall diagnosis ===")
    print(f"  GT objects: {n}   det frames scanned: {frames_seen}")
    print(f"  detected by YOLOE in >=1 frame (IoU>{TAU}): {len(detected)}/{n} = {dr:.3f}")
    print(f"  pipeline tracks_per_gt (mature) = 0.785   IoU>=.25 recall = 0.484")
    print(f"  => association/maturation headroom = {dr:.3f} - 0.785 = {dr-0.785:+.3f}")


if __name__ == "__main__":
    main()
