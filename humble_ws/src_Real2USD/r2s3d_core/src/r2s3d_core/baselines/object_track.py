"""ObjectTrack eval methods (Phase 2, detector-in-sim).

Where ``sam3d_layout`` (Phase 0) fed SAM3D one *perfect GT mask* per GT instance,
these methods feed SAM3D the best view of each **mature ObjectTrack** built from a real
detector's output (cached :class:`DetectionSet`). This measures how much placement
degrades with a real detector, and how much multi-view association + late-merge recover
from detector fragmentation (duplicate rate ↓, SAM3D invocations ↓).

The SAM3D placement math is reused wholesale from ``sam3d_layout`` — Phase 2 changes
*what feeds* SAM3D (a deduped multi-view track vs a GT instance), not the placement.
The ICP variant registers against the track's **fused multi-view cloud** (the natural
multi-view target, prototyped as Phase-0 ``--icp-accumulate``).

Methods:
  * ``object_track``       — full pipeline (re-ID + late-merge on).
  * ``object_track_naive`` — foil: re-ID + late-merge off, so detector fragmentation
    survives as duplicate objects. ``config['v1_dedup']`` adds v1's same-label / 0.5 m
    position suppression as the "vs v1 replayed" row.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from scipy.spatial.transform import Rotation as _Rot

from ..data.base import GTObject
from ..detect import corrupt
from ..detect.cache import DetectionSet
from ..eval import geometry as geo
from ..eval.metrics import SceneObject
from ..frames import validate
from ..tracks import TrackState, run_tracker
from ..tracks.view import full_view_score
from . import sam3d_layout as s3d

log = logging.getLogger(__name__)

V1_DEDUP_POSITION_M = 0.5  # v1 sam3d_job_writer_node dedup_position_m


def _load_detections(source, config: dict) -> DetectionSet:
    det_dir = config.get("detections")
    if not det_dir:
        raise ValueError(
            "object_track needs cached detections: pass --detections <dir>. Produce them "
            "with:  uv run python -m r2s3d_core.detect.run --source <src> --scene <scene> "
            "--prompt gt --out <dir>")
    scene = getattr(source, "scene", None)
    ds = DetectionSet.load(det_dir, scene=scene)
    c = config.get("corrupt") or {}
    if any(c.get(k) for k in ("dropout", "jitter_px", "track_break", "split")):
        ds = corrupt.apply(ds, dropout=c.get("dropout", 0.0), jitter_px=c.get("jitter_px", 0),
                           track_break=c.get("track_break", 0.0), split=c.get("split", 0.0),
                           seed=config.get("seed", 0))
        log.info("applied detection corruptions: %s", c)
    return ds


def _best_view(track):
    """Best-scoring kept view that still has a retained mask (novelty-aware)."""
    cands = [o for o in track.kept_views if o.mask is not None]
    if not cands:
        return None
    return max(cands, key=lambda o: full_view_score(o, [x for x in cands if x is not o]))


def _v1_position_suppress(tracks):
    """Replicate v1's 0.5 m same-label position dedup: keep the first mature track per
    (label, 0.5 m) neighborhood; suppress the rest (marks them REJECTED)."""
    kept = []
    for t in sorted((t for t in tracks if t.state == TrackState.MATURE),
                    key=lambda t: t.track_id):
        dup = any(t.label() == k.label() and
                  float(np.linalg.norm(t.centroid - k.centroid)) < V1_DEDUP_POSITION_M
                  for k in kept)
        if dup:
            t.state = TrackState.REJECTED
        else:
            kept.append(t)


def _run(source, gt: Optional[List[GTObject]], config: dict, *, reid: bool,
         late_merge: bool, registration: str) -> List[SceneObject]:
    """registration: "none" | "icp" (rigid pose) | "scale" (depth-extent scale-fit) |
    "scale_icp" (scale-fit + rigid ICP for pose). The registration target is the track's
    fused multi-view cloud — the detector-driven analogue of sam3d_layout's accumulated
    masked-depth cloud. The scale-fit is the lever rigid ICP structurally lacks."""
    frames = list(source)
    if not frames:
        log.warning("source yielded no frames")
        return []
    ds = _load_detections(source, config)

    tcfg = dict(config)
    tcfg["reid"] = config.get("reid", reid) if config.get("reid") is not None else reid
    tcfg["late_merge"] = (config.get("late_merge", late_merge)
                          if config.get("late_merge") is not None else late_merge)
    tracks = run_tracker(frames, ds.by_frame(), tcfg)

    if config.get("v1_dedup"):
        _v1_position_suppress(tracks)

    mature = [t for t in tracks if t.state == TrackState.MATURE]
    preds: List[SceneObject] = []
    placements = []    # per object: enough to rebuild the posed mesh -> scene_graph.json
    pending = 0
    invocations = 0
    for t in mature:
        obs = _best_view(t)
        if obs is None:
            log.info("track %d matured with no usable view; skipping", t.track_id)
            continue
        frame = frames[obs.frame_index]
        mask = (np.asarray(obs.mask, bool).astype(np.uint8) * 255)  # 0/255 like render_instance_mask
        H, W = frame.depth.shape[:2]
        invocations += 1

        if config.get("full_frame", True):
            rgb_in, mask_in, depth_in, bbox_in = frame.rgb, mask, frame.depth, (0, 0, W - 1, H - 1)
        else:
            x0, y0, x1, y1 = s3d.crop_bbox_from_mask(mask)
            rgb_in = frame.rgb[y0:y1 + 1, x0:x1 + 1]
            mask_in = mask[y0:y1 + 1, x0:x1 + 1]
            depth_in = frame.depth[y0:y1 + 1, x0:x1 + 1]
            bbox_in = (x0, y0, x1, y1)

        queue = Path(config["sam3d_queue"]) if config.get("sam3d_queue") else None
        # Stable logical job id (survives renderer pixel jitter): identity by
        # (source, scene, track, framing). Detections are cached from disk so track ids
        # are reproducible across runs.
        scene = getattr(source, "scene", "scene")
        framing = "full" if config.get("full_frame", True) else "crop"
        job_key = f"{config.get('source', 'src')}_{scene}_t{t.track_id}_{framing}"
        result = s3d.run_sam3d(
            rgb_in, mask_in, depth_in, frame.K, bbox_in,
            meta={"track_id": t.track_id, "label": t.label(), "full_width": W, "full_height": H},
            queue=queue, job_key=job_key,
        )
        if result is None:
            pending += 1
            continue
        mesh_raw, pose = result
        val = validate.check_sam3d_scale(pose["sam3d_scale"], raise_on_fail=False)
        val = val or validate.check_translation(pose["sam3d_translation"], raise_on_fail=False)
        posed, T_world_obj, extents = s3d.place_from_sam3d(
            mesh_raw, pose["sam3d_scale"], pose["sam3d_rotation"], pose["sam3d_translation"],
            frame.T_world_cam)
        # Track raw-mesh -> world so a placement can be rebuilt later (load the cached
        # object.glb, apply T_world_mesh) without re-running placement. `post` accumulates
        # the registration deltas applied to `posed` after the layout.
        T_world_raw = frame.T_world_cam @ s3d._T_cam_raw(
            pose["sam3d_scale"], pose["sam3d_rotation"], pose["sam3d_translation"])
        post = np.eye(4)

        prov = {
            "source": "sam3d", "registration": "layout", "track_id": t.track_id,
            "views_used": len(t.kept_views), "n_obs": t.n_obs,
            "label_votes": dict(t.label_votes), "det_track_ids": sorted(t.det_track_ids),
            "merged_from": t.merged_from, "best_view_frame": obs.frame_id,
            "validation": val,
        }

        if registration != "none":
            do_scale = registration in ("scale", "scale_icp")
            do_icp = registration in ("icp", "scale_icp")
            # register against the track's FUSED multi-view cloud (Phase-2 upgrade over
            # sam3d_layout's single-view ICP target; the multi-view analogue of that
            # method's accumulated masked-depth cloud).
            target = t.fused_cloud
            prov["reg_target"] = f"fused_{len(target)}pts"
            if len(target) < 30:
                prov["registration"] = "layout(too_few_fused_points)"
            else:
                if do_scale:
                    # Set metric scale from the fused-cloud OBB extent (the observed metric
                    # size); rigid ICP (if any) then fixes pose only. This is the scale lever
                    # ICP lacks — see sam3d_layout._fit_scale_to_extent.
                    tgt_ext = s3d._observed_obb_extent(target)
                    if tgt_ext is not None:
                        M, sinfo = s3d._fit_scale_to_extent(posed, tgt_ext)
                        posed.apply_transform(M)
                        post = M @ post
                        prov["scale_fit"] = sinfo
                    else:
                        prov["scale_fit"] = {"skipped": "degenerate_target"}
                if do_icp:
                    # rigid: recovers pose only (cannot rescale).
                    src_pts = (geo.sample_surface(posed, 2000, seed=t.track_id)
                               if len(posed.faces) else np.asarray(posed.vertices))
                    delta, info = s3d.refine_icp(src_pts, target, np.eye(4))
                    posed.apply_transform(delta)
                    post = delta @ post
                    prov["icp"] = info
                prov["registration"] = "layout+" + registration
                # One OBB from sampled points after all transforms -> pose + extents.
                # scale/ICP deltas may carry scale, so re-derive rather than compose.
                T_world_obj, extents = s3d._fast_obb(posed, seed=t.track_id)

        preds.append(SceneObject(label=t.label(), T_world_obj=T_world_obj, extents=extents,
                                 mesh=posed, provenance=prov))
        # Per-object placement -> scene_graph.json (parity with sam3d_layout). Load
        # <queue>/output/<job_id>/object.glb and apply T_world_mesh to rebuild the posed
        # prediction with no re-run. Keyed by track_id (the detector-path object identity).
        _jid = s3d._stable_job_id(job_key)
        placements.append({
            "id": int(t.track_id), "label": t.label(),
            "center": np.asarray(T_world_obj, float)[:3, 3].tolist(),   # world OBB center
            "extents": np.asarray(extents, float).tolist(),             # metric OBB size
            "T_world_obj": np.asarray(T_world_obj, float).tolist(),     # OBB pose
            "T_world_mesh": (post @ T_world_raw).tolist(),              # raw object.glb verts -> world
            "mesh": f"output/{_jid}/object.glb",                        # relative to sam3d_queue
            "job_id": _jid, "registration": prov["registration"],
            # best-view camera that generated this mesh (also a nav viewpoint to observe it).
            # cam->world, OpenCV-optical (x-right, y-down, z-forward) in Z-up world.
            "best_frame_id": int(frame.frame_id), "view_index": int(obs.frame_index),
            "cam_position": np.asarray(frame.T_world_cam, float)[:3, 3].tolist(),
            "cam_quat_xyzw": _Rot.from_matrix(
                np.asarray(frame.T_world_cam, float)[:3, :3]).as_quat().tolist(),
            "T_world_cam": np.asarray(frame.T_world_cam, float).tolist(),
            "camera_K": np.asarray(frame.K, float).tolist(),
            # detector-path provenance the GT path lacks (track lineage / label vote)
            "n_obs": int(t.n_obs), "views_used": len(t.kept_views),
            "det_track_ids": sorted(t.det_track_ids), "merged_from": t.merged_from,
            "scale_fit": prov.get("scale_fit"), "icp": prov.get("icp"),
        })

    # hand per-object placements to the runner to persist as scene_graph.json
    if placements:
        config.setdefault("_placements", {})[str(getattr(source, "scene", "scene"))] = {
            "sam3d_queue": str(queue) if queue else None,
            "method": config.get("method"),
            "objects": placements,
        }

    # stash scene-level Phase-2 stats for the run record (merged by eval.run)
    scene = getattr(source, "scene", "?")
    stats = config.setdefault("_method_stats", {})
    n_gt = len(gt) if gt else 0
    stats[scene] = {
        "sam3d_invocations": invocations,
        "n_tracks_total": sum(1 for t in tracks if t.state != TrackState.MERGED),
        "n_mature": len(mature),
        "n_merged": sum(1 for t in tracks if t.state == TrackState.MERGED),
        "n_rejected": sum(1 for t in tracks if t.state == TrackState.REJECTED),
        "tracks_per_gt": (len(mature) / n_gt) if n_gt else float("nan"),
        "detector_prompt": ds.meta.get("prompt"),
    }

    if config.get("debug_html"):
        from ..tracks.debug_html import write_debug_html
        p = write_debug_html(tracks, Path(config["debug_html"]) / f"{scene}_tracks.html",
                             scene=scene, config=tcfg)
        log.info("wrote track debug HTML -> %s", p)

    if pending:
        q = queue or s3d._default_queue()
        log.warning(
            "%d SAM3D job(s) pending in %s/input. Run the worker (conda sam3d-objects), then "
            "re-run to collect (outputs are input-hash cached).", pending, q)
    return preds


def _registration(config: dict) -> str:
    """Resolve the registration mode from config. Explicit ``registration`` wins; else the
    legacy ``--icp`` flag maps to "icp"; else "none"."""
    reg = config.get("registration")
    if reg:
        if reg not in ("none", "icp", "scale", "scale_icp"):
            raise ValueError(f"unknown registration {reg!r} "
                             "(have: none, icp, scale, scale_icp)")
        return reg
    return "icp" if config.get("icp") else "none"


def object_track(source, gt, config) -> List[SceneObject]:
    return _run(source, gt, config, reid=True, late_merge=True,
                registration=_registration(config))


def object_track_naive(source, gt, config) -> List[SceneObject]:
    return _run(source, gt, config, reid=False, late_merge=False,
                registration=_registration(config))


# Named registration variants of the full pipeline — symmetric with sam3d_layout's
# {_icp, _scale, _scale_icp} so both paths appear one-to-one in AVAILABLE and read the
# same way at the CLI (e.g. object_track_scale_icp vs sam3d_layout_scale_icp). These pin
# the registration mode directly; the equivalent is object_track --registration <mode>.

def object_track_icp(source, gt, config) -> List[SceneObject]:
    """ObjectTrack + rigid ICP against the track's fused multi-view cloud (pose only)."""
    return _run(source, gt, config, reid=True, late_merge=True, registration="icp")


def object_track_scale(source, gt, config) -> List[SceneObject]:
    """ObjectTrack + depth-extent scale-fit (metric scale from the fused-cloud OBB)."""
    return _run(source, gt, config, reid=True, late_merge=True, registration="scale")


def object_track_scale_icp(source, gt, config) -> List[SceneObject]:
    """ObjectTrack + scale-fit + rigid ICP for pose (the full scale+pose fix; the
    detector-driven counterpart of sam3d_layout_scale_icp)."""
    return _run(source, gt, config, reid=True, late_merge=True, registration="scale_icp")
