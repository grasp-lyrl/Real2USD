"""Per-object HTML diagnostic for the depth-extent scale-fit (Phase 3).

For each GT object, projects the SAM3D mesh silhouette into its best view BEFORE and AFTER
the scale-fit, overlaid on the RGB + the GT (native) mask. Shows the in-plane story: the
projected silhouette matches the mask better after scaling. Reports per-object silhouette↔
mask IoU before/after (a quantitative "did the image-plane match improve").

Usage:
    uv run python scripts/viz_scale_html.py --scene 137 \
        --queue results/procthor_procthor_sam3d_layout_3scene/sam3d_queue \
        --out results/phase3_scale_viz/scene137.html [--limit 24]

Note: the scale-fit target is the MULTI-VIEW fused masked-depth cloud (recovers the unseen
thin axis); the silhouette shown is the single best-view projection just for visualization.
"""
from __future__ import annotations

import argparse
import base64
import io
from pathlib import Path

import cv2
import numpy as np

from r2s3d_core.baselines import sam3d_layout as S
from r2s3d_core.data.registry import make_source


def _png_b64(rgb: np.ndarray) -> str:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0, b > 0
    u = float((a | b).sum())
    return float((a & b).sum()) / u if u else 0.0


def _draw(rgb, contours_masks, bbox, pad=12, size=260):
    """RGB crop around bbox with each (mask, BGR color) drawn as a contour."""
    H, W = rgb.shape[:2]
    x0, y0, x1, y1 = bbox
    x0, y0 = max(x0 - pad, 0), max(y0 - pad, 0)
    x1, y1 = min(x1 + pad, W - 1), min(y1 + pad, H - 1)
    crop = np.ascontiguousarray(rgb[y0:y1 + 1, x0:x1 + 1]).copy()
    for m, color in contours_masks:
        if m is None:
            continue
        cnts, _ = cv2.findContours((m[y0:y1 + 1, x0:x1 + 1] > 0).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(crop, cnts, -1, color, 2)
    if max(crop.shape[:2]) > size:
        s = size / max(crop.shape[:2])
        crop = cv2.resize(crop, (int(crop.shape[1] * s), int(crop.shape[0] * s)))
    return crop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="137")
    ap.add_argument("--split", default="train")
    ap.add_argument("--queue", required=True, help="per-run sam3d_queue with cached meshes")
    ap.add_argument("--out", default="results/phase3_scale_viz/scene.html")
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--min-mask-px", type=int, default=64)
    ap.add_argument("--min-mask-dim", type=int, default=6)
    args = ap.parse_args()

    src = make_source("procthor", args.scene, split=args.split)
    frames = list(src)
    gt = src.gt()
    queue = Path(args.queue)

    def mask_fn(g, fr):
        m = src.native_mask(fr.frame_id, g.instance_id)
        if m is None:
            return None
        ys, xs = np.where(m > 0)
        if len(xs) < args.min_mask_px:
            return None
        if (xs.max() - xs.min() + 1) < args.min_mask_dim or (ys.max() - ys.min() + 1) < args.min_mask_dim:
            return None
        return m

    rows = []
    for g in gt:
        vi = S.select_best_view(g, frames, mask_fn)
        if vi is None:
            continue
        frame = frames[vi]
        mask = mask_fn(g, frame)
        H, W = frame.depth.shape[:2]
        job_key = f"procthor_{args.scene}_i{g.instance_id}_full_native"
        res = S.run_sam3d(frame.rgb, mask, frame.depth, frame.K, (0, 0, W - 1, H - 1),
                          meta={"track_id": g.instance_id, "label": g.label,
                                "full_width": W, "full_height": H},
                          queue=queue, job_key=job_key)
        if res is None:  # not in cache
            continue
        mesh_raw, pose = res
        posed, _, _ = S.place_from_sam3d(mesh_raw, pose["sam3d_scale"], pose["sam3d_rotation"],
                                         pose["sam3d_translation"], frame.T_world_cam)
        sil_before = S.render_instance_mask(posed, frame)
        # fused multi-view target -> extent -> scale-fit
        clouds = []
        for fr in frames:
            m = mask_fn(g, fr)
            if m is not None and int((m > 0).sum()) > 50:
                clouds.append(S._masked_depth_cloud(fr, m))
        tgt_ext = S._observed_obb_extent(np.concatenate(clouds, 0)) if clouds else None
        if tgt_ext is None:
            continue
        M, sinfo = S._fit_scale_to_extent(posed, tgt_ext)
        posed_scale = posed.copy()
        posed_scale.apply_transform(M)
        sil_scale = S.render_instance_mask(posed_scale, frame)
        # scale + ICP (the actual method): pose refine after scaling
        posed_si = posed_scale.copy()
        src_pts = (S.geo.sample_surface(posed_si, 2000, seed=g.instance_id)
                   if len(posed_si.faces) else np.asarray(posed_si.vertices))
        delta, _ = S.refine_icp(src_pts, np.concatenate(clouds, 0), np.eye(4))
        posed_si.apply_transform(delta)
        sil_si = S.render_instance_mask(posed_si, frame)
        if sil_before is None or sil_scale is None or sil_si is None:
            continue

        ys, xs = np.where(mask > 0)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        img_b = _draw(frame.rgb, [(mask, (255, 255, 255)), (sil_before, (60, 60, 255))], bbox)
        img_s = _draw(frame.rgb, [(mask, (255, 255, 255)), (sil_scale, (0, 165, 255))], bbox)
        img_si = _draw(frame.rgb, [(mask, (255, 255, 255)), (sil_si, (0, 255, 0))], bbox)
        scales = np.asarray(sinfo["scales"])
        rows.append({
            "label": g.label, "id": g.instance_id,
            "img_b": _png_b64(img_b), "img_s": _png_b64(img_s), "img_si": _png_b64(img_si),
            "iou_b": _iou(sil_before, mask), "iou_s": _iou(sil_scale, mask),
            "iou_si": _iou(sil_si, mask),
            "scale_min": float(scales.min()), "scale_max": float(scales.max()),
            "corr": float(abs(np.log(scales)).max()),
        })
    src.close()

    # representative medians over ALL processed objects (before the display truncation)
    med = lambda k: float(np.median([r[k] for r in rows])) if rows else 0.0
    med_b, med_s, med_si = med("iou_b"), med("iou_s"), med("iou_si")
    n_all = len(rows)

    rows.sort(key=lambda r: -r["corr"])  # biggest scale corrections first (most illustrative)
    rows = rows[:args.limit]

    cells = []
    for r in rows:
        cells.append(f"""
        <div class="card">
          <div class="cap"><b>{r['label']}</b> #{r['id']} &nbsp; scale×[{r['scale_min']:.2f}–{r['scale_max']:.2f}]</div>
          <div class="pair">
            <figure><img src="data:image/png;base64,{r['img_b']}"/>
              <figcaption>layout &nbsp; IoU {r['iou_b']:.2f}</figcaption></figure>
            <figure><img src="data:image/png;base64,{r['img_s']}"/>
              <figcaption>+scale &nbsp; IoU {r['iou_s']:.2f}</figcaption></figure>
            <figure><img src="data:image/png;base64,{r['img_si']}"/>
              <figcaption>+scale+ICP &nbsp; IoU {r['iou_si']:.2f}</figcaption></figure>
          </div>
        </div>""")

    html = f"""<!doctype html><meta charset="utf-8"><title>scale-fit scene {args.scene}</title>
<style>
 body{{font:14px system-ui;margin:20px;background:#111;color:#eee}}
 h1{{font-size:18px}} .sub{{color:#aaa}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:14px;margin-top:14px}}
 .card{{background:#1b1b1b;border:1px solid #333;border-radius:8px;padding:8px}}
 .cap{{margin-bottom:6px}} .pair{{display:flex;gap:8px}} figure{{margin:0;flex:1;text-align:center}}
 img{{max-width:100%;border-radius:4px}} figcaption{{color:#bbb;margin-top:4px}}
 .key b{{padding:1px 4px;border-radius:3px}}
</style>
<h1>Depth-extent scale-fit — scene {args.scene} <span class="sub">(showing {len(rows)} of {n_all}, biggest corrections first)</span></h1>
<p class="key">Best-view silhouette vs <b style="color:#fff;border:1px solid #fff">GT mask</b>:
 <b style="color:#66f;border:1px solid #66f">layout</b>
 &nbsp; <b style="color:#fa0;border:1px solid #fa0">+scale</b>
 &nbsp; <b style="color:#6f6;border:1px solid #6f6">+scale+ICP</b>.
 Median silhouette↔mask IoU over all {n_all}: layout <b>{med_b:.2f}</b> → +scale <b>{med_s:.2f}</b>
 → +scale+ICP <b>{med_si:.2f}</b>.
 Scale target = multi-view fused masked depth (thin axis needs multiple views).</p>
<div class="grid">{''.join(cells)}</div>
"""
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"wrote {out}  ({len(rows)} objects; median IoU {med_b:.2f} -> {med_a:.2f})")


if __name__ == "__main__":
    main()
