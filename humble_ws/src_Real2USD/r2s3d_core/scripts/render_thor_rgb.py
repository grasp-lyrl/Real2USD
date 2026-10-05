"""RGB view of a ProcTHOR scene from the SAME camera as scripts/render_scene.py.

Adds an AI2-THOR third-party camera at a given world-frame eye/target (Z-up) and
saves its RGB frame. By default it computes the camera from the GT-scene bbox with
the same elev/azim/dist/FOV formula as render_scene, and prints the resulting
``--center/--eye`` so you can pass them to render_scene.py for a pixel-matched
mesh-vs-RGB pair.

Usage (needs the GPU display; starts AI2-THOR live):
  DISPLAY=:0 XAUTHORITY=/run/user/1003/gdm/Xauthority \
  uv run --extra procthor --extra mesh --extra viz \
    python scripts/render_thor_rgb.py --scene 200
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from r2s3d_core.data.procthor import ProcThorSource, _M_WU
from render_scene import ELEV, AZIM, FOV, DIST_MULT, _autocrop


def _world_to_unity_pos(p_w):
    return (_M_WU @ np.asarray(p_w, float)).tolist()  # M_WU is self-inverse: W<->U


def _look_angles_unity(eye_w, center_w):
    """AI2-THOR third-party-camera rotation (x=pitch, y=yaw) to look eye->center.

    Inverse of procthor.thor_camera_to_world's forward vector
    fwd_unity = (sin y cos p, -sin p, cos y cos p)."""
    d = _M_WU @ (np.asarray(center_w, float) - np.asarray(eye_w, float))
    d = d / (np.linalg.norm(d) + 1e-9)
    pitch = np.degrees(np.arcsin(-d[1]))          # +pitch tilts down
    yaw = np.degrees(np.arctan2(d[0], d[2]))      # yaw 0 faces +Z
    return float(pitch), float(yaw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="200")
    ap.add_argument("--split", default="val")
    ap.add_argument("--elev", type=float, default=ELEV)
    ap.add_argument("--azim", type=float, default=AZIM)
    ap.add_argument("--center", type=float, nargs=3, default=None)
    ap.add_argument("--eye", type=float, nargs=3, default=None)
    ap.add_argument("--width", type=int, default=1500)
    ap.add_argument("--height", type=int, default=1050)
    ap.add_argument("--topdown", action="store_true",
                    help="orthographic top-down floor-plan view (no wall occlusion) "
                         "via GetMapViewCameraProperties, white background")
    ap.add_argument("--out-dir", default="results/paper/_figs")
    args = ap.parse_args()

    src = ProcThorSource(scene=args.scene, split=args.split, gt_mesh="asset",
                         width=args.width, height=args.height)
    list(src)  # materialize GT (and cache); leaves controller available via _start
    gts = [g for g in src.gt() if getattr(g, "mesh", None) is not None]

    from PIL import Image
    out = Path(args.out_dir) / f"scene_s{args.scene}_rgb{'_topdown' if args.topdown else ''}.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    # --- top-down floor-plan view (no wall occlusion, white bg) --------------
    if args.topdown:
        c = src._start()
        props = c.step(action="GetMapViewCameraProperties",
                       raise_for_failure=True).metadata["actionReturn"]
        ev = c.step(action="AddThirdPartyCamera", skyboxColor="white", **props)
        frame = np.asarray(ev.third_party_camera_frames[-1])[:, :, :3]
        src.close()
        Image.fromarray(frame).save(out)
        _autocrop(str(out), also_pdf=True)
        print(f"wrote {out} (+ .pdf)  [top-down, map-view camera]")
        return

    # camera from GT-scene bbox (same formula as render_scene) unless overridden
    pts = np.vstack([np.asarray(g.mesh.vertices) for g in gts])
    bmin, bmax = pts.min(0), pts.max(0)
    center = np.asarray(args.center, float) if args.center else 0.5 * (bmin + bmax)
    if args.eye:
        eye = np.asarray(args.eye, float)
    else:
        diag = float(np.linalg.norm(bmax - bmin))
        az, el = np.radians(args.azim), np.radians(args.elev)
        d = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        eye = center + d * diag * DIST_MULT

    pitch, yaw = _look_angles_unity(eye, center)
    print(f"SHARED CAMERA  --center {center.round(3).tolist()}  --eye {eye.round(3).tolist()}")
    print(f"thor 3pc: pos(unity)={np.round(_world_to_unity_pos(eye),3).tolist()} "
          f"pitch={pitch:.1f} yaw={yaw:.1f} fov={FOV}")

    c = src._start()
    ev = c.step(action="AddThirdPartyCamera", skyboxColor="white",
                position=dict(zip("xyz", _world_to_unity_pos(eye))),
                rotation={"x": pitch, "y": yaw, "z": 0.0},
                fieldOfView=FOV)
    frame = np.asarray(ev.third_party_camera_frames[0])[:, :, :3]
    src.close()

    Image.fromarray(frame).save(out)
    _autocrop(str(out), also_pdf=True)
    print(f"wrote {out} (+ .pdf)")


if __name__ == "__main__":
    main()
