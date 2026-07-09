"""Ablation: does feeding SAM3D the FULL image (more scene context + a full-frame
depth pointmap) fix its scale over-prediction, vs. the tight object CROP we use now?

For each Replica GT object we pick the same best view as the sam3d_layout baseline, then
submit TWO SAM3D jobs through the disk-queue worker:
  - CROP : tight bbox crop + crop depth   (current baseline)
  - FULL : whole frame + full-frame mask + whole-frame depth (crop_bbox = full image)
Both reuse run_sam3d()'s exact job format + input-hash caching (the worker back-projects
whatever depth it's given, so FULL gives it the full scene pointmap).

Then we compare, per object:
  - size ratio  = reprojected-silhouette area / input-mask area   (1.0 = perfect; >1 = over-scaled)
  - scale ratio = predicted OBB diagonal / GT OBB diagonal          (1.0 = perfect)
  - reproj IoU  = silhouette vs input mask

Usage:
  uv run python scripts/ablation_full_vs_crop.py            # submit jobs (+ collect if ready)
  # then run the SAM3D worker over the queue, then re-run this to collect + report.
Outputs: results/ablation_full_vs_crop/{run.json, compare.html}
"""
import sys, os, json
import numpy as np
import cv2
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from r2s3d_core import frames
from r2s3d_core.data.registry import make_source
from r2s3d_core.eval.metrics import gt_to_scene_object
from r2s3d_core.baselines.sam3d_layout import (
    select_best_view, render_instance_mask, crop_bbox_from_mask,
    run_sam3d, place_from_sam3d, _default_queue,
)

SCENE = "room0"
STRIDE = 20
OUTDIR = os.path.join(os.path.dirname(__file__), "..", "results", "ablation_full_vs_crop")


def reproject(mesh_raw, pose, Kc, H, W):
    T = frames.T_cam_raw(pose["sam3d_scale"], pose["sam3d_rotation"], pose["sam3d_translation"])
    v = np.asarray(mesh_raw.vertices, float)
    vc = (T[:3, :3] @ v.T).T + T[:3, 3]
    z = vc[:, 2]
    fx, fy, cx, cy = Kc
    with np.errstate(divide="ignore", invalid="ignore"):
        u = fx * vc[:, 0] / z + cx
        vv = fy * vc[:, 1] / z + cy
    px = np.stack([u, vv], 1)
    sil = np.zeros((H, W), np.uint8)
    tris = [px[f].astype(np.int32) for f in mesh_raw.faces if np.all(z[f] > 1e-6)]
    if tris:
        cv2.fillPoly(sil, tris, 255)
    return sil


def metrics_for(mesh_raw, pose, frame, mask, crop_bbox, gt_diag):
    H, W = mask.shape
    x0, y0 = crop_bbox[0], crop_bbox[1]
    Kc = (frame.K[0, 0], frame.K[1, 1], frame.K[0, 2] - x0, frame.K[1, 2] - y0)
    sil = reproject(mesh_raw, pose, Kc, H, W)
    mcov = float((mask > 0).mean())
    scov = float((sil > 0).mean())
    inter = ((sil > 0) & (mask > 0)).sum()
    uni = ((sil > 0) | (mask > 0)).sum()
    iou = float(inter / uni) if uni else 0.0
    _, T_world_obj, extents = place_from_sam3d(
        mesh_raw, pose["sam3d_scale"], pose["sam3d_rotation"], pose["sam3d_translation"],
        frame.T_world_cam)
    pred_diag = float(np.linalg.norm(extents))
    return dict(size_ratio=(scov / mcov if mcov else 0.0), reproj_iou=iou,
                scale_ratio=(pred_diag / gt_diag if gt_diag else 0.0),
                pred_diag=pred_diag, mcov=mcov, scov=scov, sil=sil)


def main():
    src = make_source("replica", SCENE, stride=STRIDE)
    gt = src.gt()
    frames_list = list(src)
    queue = _default_queue()
    print(f"[{SCENE}] {len(gt)} GT objects, {len(frames_list)} frames, queue={queue}")

    rows, pending = [], 0
    for g in gt:
        vi = select_best_view(g, frames_list)
        if vi is None:
            continue
        frame = frames_list[vi]
        H, W = frame.depth.shape[:2]
        full_mask = render_instance_mask(g.mesh, frame)
        x0, y0, x1, y1 = crop_bbox_from_mask(full_mask)
        gt_diag = float(np.linalg.norm(gt_to_scene_object(g).extents))
        meta = {"track_id": g.instance_id, "label": g.label, "full_width": W, "full_height": H}

        # CROP (baseline) — cached from prior runs
        crop = run_sam3d(frame.rgb[y0:y1 + 1, x0:x1 + 1], full_mask[y0:y1 + 1, x0:x1 + 1],
                         frame.depth[y0:y1 + 1, x0:x1 + 1], frame.K, (x0, y0, x1, y1),
                         meta=meta, queue=queue)
        # FULL — whole frame; crop_bbox spans the image so worker back-projects full depth
        full = run_sam3d(frame.rgb, full_mask, frame.depth, frame.K, (0, 0, W - 1, H - 1),
                         meta=meta, queue=queue)

        if crop is None or full is None:
            pending += 1
            continue
        m_crop = metrics_for(crop[0], crop[1], frame, full_mask[y0:y1 + 1, x0:x1 + 1], (x0, y0), gt_diag)
        m_full = metrics_for(full[0], full[1], frame, full_mask, (0, 0), gt_diag)
        rows.append(dict(label=g.label, tid=g.instance_id, vi=vi, gt_diag=gt_diag,
                         crop=m_crop, full=m_full,
                         frame_rgb=frame.rgb, crop_bbox=(x0, y0, x1, y1), full_mask=full_mask))
        print(f"  {g.label:>10} #{g.instance_id:<3} | size_ratio crop {m_crop['size_ratio']:.2f} -> full {m_full['size_ratio']:.2f}"
              f" | scale_ratio crop {m_crop['scale_ratio']:.2f} -> full {m_full['scale_ratio']:.2f}")

    if pending:
        print(f"\n{pending} job(s) PENDING. Run the SAM3D worker over the queue, then re-run:")
        print("  conda run -n sam3d-objects python "
              "../real2sam3d/scripts_sam3d_worker/run_sam3d_worker.py --no-current-run "
              f"--use-depth --queue-dir {queue} --sam3d-repo ../real2sam3d/sam-3d-objects")
        # still write a manifest of what to collect
        return

    # ---- summary ----
    def agg(key, cond):
        vals = [abs(r[cond][key] - 1.0) for r in rows]  # |x-1|: distance from ideal
        return float(np.mean(vals)) if vals else 0.0
    summary = {
        "scene": SCENE, "n_objects": len(rows),
        "size_ratio_mean_crop": float(np.mean([r["crop"]["size_ratio"] for r in rows])),
        "size_ratio_mean_full": float(np.mean([r["full"]["size_ratio"] for r in rows])),
        "scale_err_mean_crop": agg("scale_ratio", "crop"),   # mean |pred/gt - 1|
        "scale_err_mean_full": agg("scale_ratio", "full"),
        "reproj_iou_mean_crop": float(np.mean([r["crop"]["reproj_iou"] for r in rows])),
        "reproj_iou_mean_full": float(np.mean([r["full"]["reproj_iou"] for r in rows])),
        "n_full_better_scale": int(sum(abs(r["full"]["scale_ratio"] - 1) < abs(r["crop"]["scale_ratio"] - 1) for r in rows)),
    }
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "run.json"), "w") as f:
        json.dump({"summary": summary,
                   "per_object": [{k: r[k] for k in ("label", "tid", "vi", "gt_diag")} |
                                  {"crop": {kk: r["crop"][kk] for kk in ("size_ratio", "scale_ratio", "reproj_iou", "pred_diag")},
                                   "full": {kk: r["full"][kk] for kk in ("size_ratio", "scale_ratio", "reproj_iou", "pred_diag")}}
                                  for r in rows]}, f, indent=1)
    write_html(rows, summary)
    print("\n==== SUMMARY (mean over %d objects) ====" % len(rows))
    print(f"  size ratio (sil/mask, ideal 1.0):  crop {summary['size_ratio_mean_crop']:.2f}x  ->  full {summary['size_ratio_mean_full']:.2f}x")
    print(f"  scale err  (|pred/gt-1|, ideal 0):  crop {summary['scale_err_mean_crop']:.2f}   ->  full {summary['scale_err_mean_full']:.2f}")
    print(f"  reproj IoU (higher better):         crop {summary['reproj_iou_mean_crop']:.2f}   ->  full {summary['reproj_iou_mean_full']:.2f}")
    print(f"  full better on scale for {summary['n_full_better_scale']}/{len(rows)} objects")
    print(f"  -> {OUTDIR}/run.json + compare.html")


def _b64(img):
    ok, buf = cv2.imencode(".png", img)
    import base64
    return base64.b64encode(buf).decode() if ok else ""


def _outline(bgr, mask, color, t=2):
    out = bgr.copy()
    c, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, c, -1, color, t)
    return out


def write_html(rows, summary):
    rows_sorted = sorted(rows, key=lambda r: -abs(r["crop"]["scale_ratio"] - 1))
    cards = []
    for r in rows_sorted:
        x0, y0, x1, y1 = r["crop_bbox"]
        rgb = cv2.cvtColor(r["frame_rgb"], cv2.COLOR_RGB2BGR)
        # CROP panel: crop rgb + mask(grn) + sam3d sil(red)
        crgb = rgb[y0:y1 + 1, x0:x1 + 1]
        cmask = r["full_mask"][y0:y1 + 1, x0:x1 + 1]
        cpanel = _outline(_outline(crgb, cmask, (0, 200, 0)), r["crop"]["sil"], (0, 0, 255))
        # FULL panel: full rgb + mask(grn) + sam3d sil(red)
        fpanel = _outline(_outline(rgb, r["full_mask"], (0, 200, 0)), r["full"]["sil"], (0, 0, 255))
        cards.append(f"""
        <div class="card">
          <div class="hdr"><b>{r['label']}</b> #{r['tid']}</div>
          <div class="cols">
            <figure><img class="c" src="data:image/png;base64,{_b64(cpanel)}">
              <figcaption>CROP · size {r['crop']['size_ratio']:.2f}× · scale {r['crop']['scale_ratio']:.2f} · IoU {r['crop']['reproj_iou']:.2f}</figcaption></figure>
            <figure><img class="f" src="data:image/png;base64,{_b64(fpanel)}">
              <figcaption>FULL · size {r['full']['size_ratio']:.2f}× · scale {r['full']['scale_ratio']:.2f} · IoU {r['full']['reproj_iou']:.2f}</figcaption></figure>
          </div>
        </div>""")
    s = summary
    html = f"""<!doctype html><html><head><meta charset=utf-8><title>SAM3D crop vs full ablation</title>
<style>
body{{font:14px system-ui,sans-serif;margin:0;padding:24px;background:#0f1115;color:#e6e6e6}}
h1{{font-size:20px;margin:0 0 12px}} b{{color:#fff}}
.sum{{background:#171a21;border:1px solid #262b36;border-radius:8px;padding:14px 18px;margin-bottom:20px;max-width:760px;line-height:1.6}}
.sum td{{padding:2px 14px 2px 0}} .win{{color:#5fd48a}} .lose{{color:#f07070}}
.card{{background:#171a21;border:1px solid #262b36;border-radius:10px;padding:10px 14px;margin-bottom:12px}}
.hdr{{margin-bottom:6px;font-size:15px}}
.cols{{display:flex;gap:18px;align-items:flex-start}}
figure{{margin:0}} img.f{{max-width:420px}} img.c{{width:180px;image-rendering:pixelated}}
figcaption{{font-size:12px;color:#9aa;margin-top:4px}}
</style></head><body>
<h1>SAM3D input ablation — tight CROP vs FULL image</h1>
<div class="sum">
Mean over {s['n_objects']} Replica {s['scene']} objects (green = GT mask, red = SAM3D output reprojected):
<table>
<tr><td></td><td><b>CROP (now)</b></td><td><b>FULL image</b></td></tr>
<tr><td>size ratio (sil/mask, →1.0)</td><td>{s['size_ratio_mean_crop']:.2f}×</td><td>{s['size_ratio_mean_full']:.2f}×</td></tr>
<tr><td>scale err |pred/gt−1| (→0)</td><td>{s['scale_err_mean_crop']:.2f}</td><td>{s['scale_err_mean_full']:.2f}</td></tr>
<tr><td>reproj IoU (higher better)</td><td>{s['reproj_iou_mean_crop']:.2f}</td><td>{s['reproj_iou_mean_full']:.2f}</td></tr>
</table>
FULL beats CROP on scale for <b>{s['n_full_better_scale']}/{s['n_objects']}</b> objects.
Sorted worst-CROP-scale first.
</div>
{''.join(cards)}
</body></html>"""
    open(os.path.join(OUTDIR, "compare.html"), "w").write(html)


if __name__ == "__main__":
    main()
