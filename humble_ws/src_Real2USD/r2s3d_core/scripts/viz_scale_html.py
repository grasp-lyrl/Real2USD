"""Before/after HTML for the depth-extent scale-fit — reads placements.json (no re-run).

For each placed object, projects its SAM3D mesh silhouette into its best view BEFORE
(layout) and AFTER (the saved scale+ICP placement), over the GT native mask. Uses
``placements.json`` (T_world_mesh) + the cached ``object.glb``/``pose.json`` — so it does
NOT re-run scale-fit/ICP or fuse depth. Rendering is only to fetch the best-view RGB + mask.

Usage:
    uv run python scripts/viz_scale_html.py \
        --placements results/procthor_procthor_sam3d_layout_scale_icp_3scene/placements.json \
        --scene 137 --out results/phase3_scale_viz/scene137.html [--limit 24]
"""
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh

from r2s3d_core.baselines.sam3d_layout import render_instance_mask
from r2s3d_core.data.registry import make_source
from r2s3d_core.frames import T_cam_raw


def _png_b64(rgb):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _iou(a, b):
    a, b = a > 0, b > 0
    u = float((a | b).sum())
    return float((a & b).sum()) / u if u else 0.0


def _load_mesh(glb: Path):
    m = trimesh.load(str(glb), process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate([g for g in m.geometry.values()])
    return m


def _draw(rgb, layers, bbox, pad=12, size=260):
    H, W = rgb.shape[:2]
    x0, y0, x1, y1 = bbox
    x0, y0 = max(x0 - pad, 0), max(y0 - pad, 0)
    x1, y1 = min(x1 + pad, W - 1), min(y1 + pad, H - 1)
    crop = np.ascontiguousarray(rgb[y0:y1 + 1, x0:x1 + 1]).copy()
    for m, color in layers:
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
    ap.add_argument("--placements", required=True, help="placements.json from a run")
    ap.add_argument("--scene", default="137")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="results/phase3_scale_viz/scene.html")
    ap.add_argument("--limit", type=int, default=24)
    args = ap.parse_args()

    P = json.load(open(args.placements))
    sc = P["scenes"][args.scene]
    queue = Path(sc["sam3d_queue"])
    objs = sc["objects"]

    # Render only up to the last needed best-view frame (native_mask needs its seg stored).
    src = make_source("procthor", args.scene, split=args.split)
    needed = {o["best_frame_id"] for o in objs}
    frames_by_id = {}
    for fr in src:
        if fr.frame_id in needed:
            frames_by_id[fr.frame_id] = fr
        if len(frames_by_id) == len(needed):
            break

    rows = []
    for o in objs:
        fr = frames_by_id.get(o["best_frame_id"])
        if fr is None:
            continue
        mask = src.native_mask(o["best_frame_id"], o["instance_id"])
        if mask is None:
            continue
        glb = queue / "output" / o["job_id"] / "object.glb"
        pose_p = queue / "output" / o["job_id"] / "pose.json"
        if not glb.is_file() or not pose_p.is_file():
            continue
        raw = _load_mesh(glb)
        pose = json.load(open(pose_p))
        # before = layout pose (from cached SAM3D params); after = saved scale+ICP transform
        T_before = fr.T_world_cam @ T_cam_raw(pose["sam3d_scale"], pose["sam3d_rotation"],
                                              pose["sam3d_translation"])
        mb = raw.copy(); mb.apply_transform(T_before)
        ma = raw.copy(); ma.apply_transform(np.array(o["T_world_mesh"]))
        sb, sa = render_instance_mask(mb, fr), render_instance_mask(ma, fr)
        if sb is None or sa is None:
            continue
        ys, xs = np.where(mask > 0)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        img_b = _draw(fr.rgb, [(mask, (255, 255, 255)), (sb, (60, 60, 255))], bbox)
        img_a = _draw(fr.rgb, [(mask, (255, 255, 255)), (sa, (0, 255, 0))], bbox)
        sf = o.get("scale_fit") or {}
        scales = np.asarray(sf.get("scales", [1, 1, 1]))
        rows.append({"label": o["label"], "id": o["instance_id"],
                     "img_b": _png_b64(img_b), "img_a": _png_b64(img_a),
                     "iou_b": _iou(sb, mask), "iou_a": _iou(sa, mask),
                     "smin": float(scales.min()), "smax": float(scales.max()),
                     "corr": float(np.abs(np.log(np.clip(scales, 1e-3, None))).max())})
    src.close()

    med = lambda k: float(np.median([r[k] for r in rows])) if rows else 0.0
    med_b, med_a, n_all = med("iou_b"), med("iou_a"), len(rows)
    rows.sort(key=lambda r: -r["corr"])
    rows = rows[:args.limit]

    cells = "".join(f"""
      <div class="card"><div class="cap"><b>{r['label']}</b> #{r['id']} &nbsp; scale×[{r['smin']:.2f}–{r['smax']:.2f}]</div>
        <div class="pair">
          <figure><img src="data:image/png;base64,{r['img_b']}"/><figcaption>layout &nbsp; IoU {r['iou_b']:.2f}</figcaption></figure>
          <figure><img src="data:image/png;base64,{r['img_a']}"/><figcaption>scale+ICP &nbsp; IoU {r['iou_a']:.2f}</figcaption></figure>
        </div></div>""" for r in rows)

    html = f"""<!doctype html><meta charset="utf-8"><title>scale-fit before/after {args.scene}</title>
<style>body{{font:14px system-ui;margin:20px;background:#111;color:#eee}}h1{{font-size:18px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:14px;margin-top:14px}}
.card{{background:#1b1b1b;border:1px solid #333;border-radius:8px;padding:8px}}.cap{{margin-bottom:6px}}
.pair{{display:flex;gap:8px}}figure{{margin:0;flex:1;text-align:center}}img{{max-width:100%;border-radius:4px}}
figcaption{{color:#bbb;margin-top:4px}}.key b{{border:1px solid;padding:1px 4px;border-radius:3px}}</style>
<h1>Depth-extent scale-fit — scene {args.scene} <span style="color:#aaa">(showing {len(rows)} of {n_all}, biggest scale corrections first)</span></h1>
<p class="key">Best-view silhouette vs <b style="color:#fff">GT mask</b>:
 <b style="color:#66f;color:#66f">layout</b> &nbsp; <b style="color:#6f6">scale+ICP</b>.
 Median silhouette↔mask IoU over all {n_all}: layout <b>{med_b:.2f}</b> → scale+ICP <b>{med_a:.2f}</b>.
 Rebuilt from placements.json (no re-run of scale/ICP).</p>
<div class="grid">{cells}</div>"""
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"wrote {out}  ({len(rows)} shown / {n_all} total; median IoU {med_b:.2f} -> {med_a:.2f})")


if __name__ == "__main__":
    main()
