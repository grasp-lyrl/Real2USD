"""Build a self-contained HTML gallery of SAM3D worker inputs vs outputs, per object,
to sanity-check whether layout error is a SAM3D issue or an implementation issue.

For every job in the disk queue it shows:
  - the RGB crop actually fed to SAM3D,
  - the input mask overlaid on the RGB (is our mask tight/correct?),
  - the input depth crop,
  - SAM3D's OWN output mesh reprojected into that crop using SAM3D's OWN predicted
    pose (frames.T_cam_raw) + the crop intrinsics -- if this silhouette doesn't match
    the input mask, the shape/pose error is SAM3D's, in its native camera frame, with
    no world placement / registration involved,
  - a green(mask) vs red(SAM3D) comparison.

Sorted worst-IoU first. Output: a single .html with all images inlined as base64.

Run:  uv run python scripts/viz_sam3d_inputs.py [QUEUE_DIR] [OUT_HTML]
"""
import sys, os, json, glob, base64
import numpy as np
import cv2
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from r2s3d_core import frames

QUEUE = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Data/datasets/sam3d_queue")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(QUEUE, "sam3d_input_viz.html")


def png_b64(img_bgr_or_gray):
    ok, buf = cv2.imencode(".png", img_bgr_or_gray)
    return base64.b64encode(buf).decode() if ok else ""


def reproject_silhouette(glb, pose, Kc, H, W):
    m = trimesh.load(glb, process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate([g for g in m.geometry.values()])
    T = frames.T_cam_raw(pose["sam3d_scale"], pose["sam3d_rotation"], pose["sam3d_translation"])
    v = np.asarray(m.vertices, float)
    vc = (T[:3, :3] @ v.T).T + T[:3, 3]
    z = vc[:, 2]
    fx, fy, cx, cy = Kc
    with np.errstate(divide="ignore", invalid="ignore"):
        u = fx * vc[:, 0] / z + cx
        vv = fy * vc[:, 1] / z + cy
    px = np.stack([u, vv], 1)
    sil = np.zeros((H, W), np.uint8)
    tris = [px[f].astype(np.int32) for f in m.faces if np.all(z[f] > 1e-6)]
    if tris:
        cv2.fillPoly(sil, tris, 255)
    return sil


def outline(bgr, mask, color, thick=2):
    out = bgr.copy()
    cnts, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, color, thick)
    return out


def tint(bgr, mask, color, alpha=0.4):
    out = bgr.copy().astype(np.float32)
    m = (mask > 0)
    for c in range(3):
        out[..., c][m] = (1 - alpha) * out[..., c][m] + alpha * color[c]
    return out.astype(np.uint8)


cards = []
ious, ratios = [], []
jobs = sorted(glob.glob(os.path.join(QUEUE, "output", "*", "")))
for d in jobs:
    try:
        meta = json.load(open(d + "meta.json"))
        pose = json.load(open(d + "pose.json"))
        rgb = cv2.imread(d + "rgb.png")           # BGR
        mask = cv2.imread(d + "mask.png", 0)
        depth = np.load(d + "depth.npy")
        if rgb is None or mask is None:
            continue
        H, W = mask.shape
        Kf = meta["camera_info"]["K"]
        x0, y0 = meta["crop_bbox"][:2]
        Kc = (Kf[0], Kf[4], Kf[2] - x0, Kf[5] - y0)
        sil = reproject_silhouette(d + "object.glb", pose, Kc, H, W)

        inter = ((sil > 0) & (mask > 0)).sum()
        uni = ((sil > 0) | (mask > 0)).sum()
        iou = float(inter / uni) if uni else 0.0
        mcov = float((mask > 0).mean())
        scov = float((sil > 0).mean())
        ratio = scov / mcov if mcov else 0.0
        ious.append(iou); ratios.append(ratio)

        # depth -> colormap (ignore zeros/nan)
        dv = depth.copy().astype(np.float32)
        valid = np.isfinite(dv) & (dv > 0)
        dn = np.zeros_like(dv)
        if valid.any():
            lo, hi = np.percentile(dv[valid], [2, 98])
            dn[valid] = np.clip((dv[valid] - lo) / max(hi - lo, 1e-6), 0, 1)
        depth_col = cv2.applyColorMap((dn * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        depth_col[~valid] = (30, 30, 30)

        panels = {
            "RGB (fed to SAM3D)": png_b64(rgb),
            "input mask on RGB": png_b64(tint(outline(rgb, mask, (0, 200, 0)), mask, (0, 200, 0), 0.25)),
            "input depth": png_b64(depth_col),
            "SAM3D output reprojected": png_b64(outline(rgb, sil, (0, 0, 255))),
            "mask (grn) vs SAM3D (red)": png_b64(outline(outline(rgb, mask, (0, 200, 0)), sil, (0, 0, 255))),
        }
        s = pose["sam3d_scale"]; t = pose["sam3d_translation"]
        cards.append(dict(
            label=pose.get("label", "?"), tid=pose.get("track_id", "?"),
            iou=iou, mcov=mcov, scov=scov, ratio=ratio,
            scale=float(np.mean(s)), tnorm=float(np.linalg.norm(t)),
            ms=pose.get("inference_ms", 0), panels=panels,
        ))
    except Exception as e:
        print("skip", d, repr(e))

cards.sort(key=lambda c: c["iou"])  # worst first

W_IMG = 150  # css display width
def img_tag(b64):
    return f'<img style="width:{W_IMG}px;image-rendering:pixelated" src="data:image/png;base64,{b64}">'

rows = []
for c in cards:
    imgs = "".join(
        f'<figure>{img_tag(b64)}<figcaption>{name}</figcaption></figure>'
        for name, b64 in c["panels"].items()
    )
    badge = "good" if c["iou"] >= 0.6 else ("mid" if c["iou"] >= 0.4 else "bad")
    rows.append(f"""
    <div class="card">
      <div class="hdr">
        <span class="lbl">{c['label']} <span class="tid">#{c['tid']}</span></span>
        <span class="iou {badge}">IoU {c['iou']:.2f}</span>
      </div>
      <div class="panels">{imgs}</div>
      <div class="stats">
        mask cov {c['mcov']:.2f} · SAM3D cov {c['scov']:.2f} ·
        <b>size ratio {c['ratio']:.2f}×</b> · pred scale {c['scale']:.2f} ·
        |t| {c['tnorm']:.2f} m · {c['ms']/1000:.1f}s
      </div>
    </div>""")

mean_iou = np.mean(ious) if ious else 0
mean_ratio = np.mean(ratios) if ratios else 0
html = f"""<!doctype html><html><head><meta charset="utf-8"><title>SAM3D input/output viz</title>
<style>
:root{{color-scheme:dark light}}
body{{font:14px/1.4 system-ui,sans-serif;margin:0;padding:24px;background:#0f1115;color:#e6e6e6}}
h1{{font-size:20px;margin:0 0 4px}}
.sub{{color:#9aa;margin:0 0 16px}}
.legend{{background:#171a21;border:1px solid #262b36;border-radius:8px;padding:12px 16px;margin-bottom:20px;max-width:900px}}
.legend b{{color:#fff}}
.grid{{display:flex;flex-direction:column;gap:14px}}
.card{{background:#171a21;border:1px solid #262b36;border-radius:10px;padding:12px 14px}}
.hdr{{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}}
.lbl{{font-size:16px;font-weight:600}} .tid{{color:#889;font-weight:400}}
.iou{{font-weight:700;padding:2px 10px;border-radius:20px}}
.iou.good{{background:#12351f;color:#5fd48a}} .iou.mid{{background:#33300f;color:#e6c34a}} .iou.bad{{background:#3a1414;color:#f07070}}
.panels{{display:flex;gap:10px;flex-wrap:wrap}}
figure{{margin:0;text-align:center}} figcaption{{font-size:11px;color:#889;margin-top:3px;width:{W_IMG}px}}
.stats{{margin-top:8px;color:#9aa;font-size:12.5px}} .stats b{{color:#e6c34a}}
</style></head><body>
<h1>SAM3D worker — inputs vs. reprojected output</h1>
<p class="sub">{len(cards)} objects · sorted worst-IoU first · mean reproj IoU <b>{mean_iou:.2f}</b> · mean size ratio <b>{mean_ratio:.2f}×</b></p>
<div class="legend">
  <b>What to look for.</b> Columns 1–3 are the exact SAM3D inputs — check the
  <b style="color:#5fd48a">green mask</b> is tight to the RGB object (if not, it's an input/masking bug on our side).
  Columns 4–5 are SAM3D's own output mesh reprojected with SAM3D's own predicted pose into the same crop, using the real crop intrinsics.
  If the <b style="color:#f07070">red silhouette</b> is bigger than the green mask (<b>size ratio &gt; 1</b>), SAM3D over-predicted scale —
  a SAM3D error visible in its native camera frame, with zero world-registration involved. Low IoU here = SAM3D shape/pose issue, not our layout code.
</div>
<div class="grid">{''.join(rows)}</div>
</body></html>"""

os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w").write(html)
print(f"wrote {OUT}  ({len(cards)} objects, {len(html)//1024} KB)")
print(f"mean reproj IoU {mean_iou:.3f} | mean size ratio {mean_ratio:.2f}x")
