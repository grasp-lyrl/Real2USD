"""Side experiment: what does SAM3D do with a FULL-IMAGE mask (whole scene = one object)?

SAM 3D Objects is a *single-object* reconstructor: image + mask + (pointmap) -> one
canonical mesh + predicted (scale, R, t). The mask is the FOREGROUND selector — it tells
SAM3D *which* object in the frame to reconstruct. Our pipeline sends one job per detected
object with that object's mask. This experiment asks the opposite question:

  If we hand SAM3D the whole frame and a full-white mask (every pixel = foreground),
  does it need individual object masks, or can it reconstruct the whole scene at once?

Prior (worth confirming, not assuming): a full mask is out-of-distribution — SAM3D was
trained on single segmented objects — so it likely either (a) latches onto the single
most salient object, or (b) fuses the visible surfaces into one blob mesh. Either way the
asset-CENTRIC decomposition (the point of the map) is gone. This produces the picture.

Procedure, per selected frame (NOT per object):
  rgb   = full frame
  mask  = np.full((H,W), 255)         <-- the whole image is "the object"
  depth = full frame depth            <-- worker back-projects the full-scene pointmap
  crop_bbox = (0,0,W-1,H-1)
submitted through the SAME disk-queue worker + job_key cache as the real pipeline.

On collect, each frame's single mesh is placed via place_from_sam3d() @ T_world_cam and
reprojected. The report shows, per frame: input RGB, the reconstructed-mesh silhouette
overlaid, its coverage of the frame, and mesh/OBB stats — plus how one whole-scene blob
compares to the N GT objects actually in view.

Usage:
  uv run python scripts/exp_full_scene_fullmask.py [--source replica] [--scene room0]
                                                   [--stride 20] [--max-frames 12]
  # then run the SAM3D worker over the queue (command printed if jobs are pending),
  # then re-run this to collect + report.
Outputs: results/exp_full_scene_fullmask_<source>_<scene>/{run.json, report.html}
"""
import argparse
import base64
import json
import os
import sys

import cv2
import numpy as np
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from r2s3d_core import frames  # noqa: E402
from r2s3d_core.data.registry import make_source  # noqa: E402
from r2s3d_core.baselines.sam3d_layout import (  # noqa: E402
    place_from_sam3d, render_instance_mask, run_sam3d, _default_queue,
)

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def reproject(mesh_raw, pose, K, H, W):
    """Silhouette of the raw SAM3D mesh under its predicted camera-frame pose (full image)."""
    T = frames.T_cam_raw(pose["sam3d_scale"], pose["sam3d_rotation"], pose["sam3d_translation"])
    v = np.asarray(mesh_raw.vertices, float)
    vc = (T[:3, :3] @ v.T).T + T[:3, 3]
    z = vc[:, 2]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = fx * vc[:, 0] / z + cx
        vv = fy * vc[:, 1] / z + cy
    px = np.stack([u, vv], 1)
    sil = np.zeros((H, W), np.uint8)
    tris = [px[f].astype(np.int32) for f in mesh_raw.faces if np.all(z[f] > 1e-6)]
    if tris:
        cv2.fillPoly(sil, tris, 255)
    return sil


def pick_frames(frames_list, max_frames):
    """Evenly spread up to `max_frames` frames across the trajectory."""
    n = len(frames_list)
    if n <= max_frames:
        return list(range(n))
    idx = np.linspace(0, n - 1, max_frames).round().astype(int)
    return sorted(set(int(i) for i in idx))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="replica")
    ap.add_argument("--scene", default="room0")
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--max-frames", type=int, default=12)
    ap.add_argument("--split", default="val", help="ProcTHOR split (ignored for replica)")
    args = ap.parse_args()

    src_kwargs = {"stride": args.stride}
    if args.source in ("procthor", "molmospaces"):
        src_kwargs["split"] = args.split
    src = make_source(args.source, args.scene, **src_kwargs)
    frames_list = list(src)
    gt = src.gt()
    queue = _default_queue()
    outdir = os.path.join(RESULTS, f"exp_full_scene_fullmask_{args.source}_{args.scene}")
    sel = pick_frames(frames_list, args.max_frames)
    print(f"[{args.source}/{args.scene}] {len(frames_list)} frames "
          f"({len(sel)} selected), {len(gt)} GT objects, queue={queue}")

    rows, pending = [], 0
    for vi in sel:
        frame = frames_list[vi]
        H, W = frame.depth.shape[:2]
        full_mask = np.full((H, W), 255, np.uint8)  # whole image = foreground
        job_key = f"{args.source}_{args.scene}_frame{int(frame.frame_id)}_fullscene"
        result = run_sam3d(
            frame.rgb, full_mask, frame.depth, frame.K, (0, 0, W - 1, H - 1),
            meta={"track_id": 0, "label": "full_scene", "full_width": W, "full_height": H},
            queue=queue, job_key=job_key,
        )
        if result is None:
            pending += 1
            continue
        mesh_raw, pose = result
        sil = reproject(mesh_raw, pose, frame.K, H, W)
        _, _, extents = place_from_sam3d(
            mesh_raw, pose["sam3d_scale"], pose["sam3d_rotation"], pose["sam3d_translation"],
            frame.T_world_cam)
        # how many GT objects are actually visible in this frame (context for "1 blob vs N")
        n_gt_visible = sum(
            1 for g in gt if render_instance_mask(g.mesh, frame) is not None)
        rows.append(dict(
            frame_id=int(frame.frame_id), vi=vi,
            coverage=float((sil > 0).mean()),          # fraction of frame the blob covers
            reproj_iou=float(((sil > 0) & (full_mask > 0)).sum() / max((sil > 0).sum(), 1)),
            n_vertices=int(len(mesh_raw.vertices)), n_faces=int(len(mesh_raw.faces)),
            obb_diag=float(np.linalg.norm(extents)), obb_extents=extents.tolist(),
            sam3d_scale=pose["sam3d_scale"], n_gt_visible=int(n_gt_visible),
            rgb=frame.rgb, sil=sil,
        ))
        print(f"  frame {frame.frame_id:>4} | coverage {rows[-1]['coverage']:.2f} "
              f"| OBB diag {rows[-1]['obb_diag']:.2f}m | {rows[-1]['n_faces']} faces "
              f"| {n_gt_visible} GT objs in view")

    if pending:
        print(f"\n{pending}/{len(sel)} job(s) PENDING. Run the SAM3D worker over the queue, "
              "then re-run this script to collect:")
        print("  conda run -n sam3d-objects python "
              "../real2sam3d/scripts_sam3d_worker/run_sam3d_worker.py --no-current-run "
              f"--use-depth --queue-dir {queue} --sam3d-repo ../real2sam3d/sam-3d-objects")
        return

    summary = {
        "source": args.source, "scene": args.scene, "n_frames": len(rows),
        "coverage_mean": float(np.mean([r["coverage"] for r in rows])) if rows else 0.0,
        "obb_diag_mean": float(np.mean([r["obb_diag"] for r in rows])) if rows else 0.0,
        "n_faces_mean": float(np.mean([r["n_faces"] for r in rows])) if rows else 0.0,
        "n_gt_visible_mean": float(np.mean([r["n_gt_visible"] for r in rows])) if rows else 0.0,
    }
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "run.json"), "w") as f:
        json.dump({"summary": summary,
                   "per_frame": [{k: r[k] for k in (
                       "frame_id", "vi", "coverage", "reproj_iou", "n_vertices", "n_faces",
                       "obb_diag", "obb_extents", "sam3d_scale", "n_gt_visible")} for r in rows]},
                  f, indent=1)
    write_html(rows, summary, outdir)
    print("\n==== SUMMARY (full-image mask, whole scene as one object) ====")
    print(f"  frames reconstructed:            {summary['n_frames']}")
    print(f"  mean blob coverage of frame:     {summary['coverage_mean']:.2f}")
    print(f"  mean reconstructed OBB diagonal: {summary['obb_diag_mean']:.2f} m")
    print(f"  mean GT objects visible/frame:   {summary['n_gt_visible_mean']:.1f} "
          f"(each collapsed into ONE blob mesh)")
    print(f"  -> {outdir}/run.json + report.html")


def _b64(bgr):
    ok, buf = cv2.imencode(".png", bgr)
    return base64.b64encode(buf).decode() if ok else ""


def _overlay(rgb, sil):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    c, _ = cv2.findContours((sil > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tint = bgr.copy()
    tint[sil > 0] = (0.5 * tint[sil > 0] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
    cv2.drawContours(tint, c, -1, (0, 0, 255), 2)
    return tint


def write_html(rows, summary, outdir):
    cards = []
    for r in rows:
        cards.append(f"""
        <div class="card">
          <div class="hdr">frame <b>{r['frame_id']}</b> · {r['n_gt_visible']} GT objects in view
            → 1 mesh ({r['n_faces']} faces)</div>
          <div class="cols">
            <figure><img src="data:image/png;base64,{_b64(cv2.cvtColor(r['rgb'], cv2.COLOR_RGB2BGR))}">
              <figcaption>input RGB (full frame)</figcaption></figure>
            <figure><img src="data:image/png;base64,{_b64(_overlay(r['rgb'], r['sil']))}">
              <figcaption>full-mask SAM3D mesh reprojected · coverage {r['coverage']:.2f}
                · OBB diag {r['obb_diag']:.2f} m</figcaption></figure>
          </div>
        </div>""")
    s = summary
    html = f"""<!doctype html><html><head><meta charset=utf-8>
<title>SAM3D full-image-mask experiment — {s['source']}/{s['scene']}</title>
<style>
body{{font:14px system-ui,sans-serif;margin:0;padding:24px;background:#0f1115;color:#e6e6e6}}
h1{{font-size:20px;margin:0 0 6px}} b{{color:#fff}} p{{color:#9aa;max-width:820px;line-height:1.6}}
.sum{{background:#171a21;border:1px solid #262b36;border-radius:8px;padding:14px 18px;margin:16px 0 20px;max-width:820px;line-height:1.7}}
.sum td{{padding:2px 16px 2px 0}}
.card{{background:#171a21;border:1px solid #262b36;border-radius:10px;padding:10px 14px;margin-bottom:12px}}
.hdr{{margin-bottom:6px;font-size:15px}}
.cols{{display:flex;gap:18px;align-items:flex-start;flex-wrap:wrap}}
figure{{margin:0}} img{{max-width:440px;border-radius:6px}}
figcaption{{font-size:12px;color:#9aa;margin-top:4px}}
</style></head><body>
<h1>SAM3D with a FULL-IMAGE mask — {s['source']}/{s['scene']}</h1>
<p>Each frame was sent to SAM 3D Objects with a full-white mask (whole image = foreground),
so SAM3D reconstructs the entire scene as ONE object instead of receiving per-object masks.
Red = reconstructed mesh silhouette reprojected into the input view.</p>
<div class="sum"><table>
<tr><td>frames reconstructed</td><td><b>{s['n_frames']}</b></td></tr>
<tr><td>mean GT objects visible / frame</td><td><b>{s['n_gt_visible_mean']:.1f}</b> — each collapsed into a single mesh</td></tr>
<tr><td>mean blob coverage of frame</td><td><b>{s['coverage_mean']:.2f}</b></td></tr>
<tr><td>mean reconstructed OBB diagonal</td><td><b>{s['obb_diag_mean']:.2f} m</b></td></tr>
<tr><td>mean faces / mesh</td><td><b>{s['n_faces_mean']:.0f}</b></td></tr>
</table></div>
{''.join(cards)}
</body></html>"""
    open(os.path.join(outdir, "report.html"), "w").write(html)


if __name__ == "__main__":
    main()
