"""SAM3D-layout baseline (Phase 0 motivating experiment).

Procedure (docs/PHASE_SPECS.md Phase 0), per scene:
  1. sample every Nth frame (via the source stride);
  2. for each GT instance, render its mask into every sampled frame and pick the
     single best view (largest visible area, not touching the image border);
  3. run SAM3D on that crop+mask -> canonical mesh + predicted (scale, R, t);
  4. place the mesh via SAM3D's predicted layout composed with that frame's
     T_world_cam  (variant A: ``sam3d_layout``);
  5. variant B (``sam3d_layout_icp``) additionally refines with v1-style ICP
     against the masked sensor-depth cloud.

This isolates *layout error* from detection error (perfect GT masks) — it is the
paper's motivating experiment ("even with perfect masks, SAM3D placement is off by
X") and the number every later phase must beat.

SAM3D itself runs in Meta's external ``sam-3d-objects`` repo + gated checkpoint in
its own conda env (disk-queue handoff). That is a human-gated dependency: the
worker call raises a loud, actionable error if the queue/output is missing. The
placement math and best-view selection here are pure and unit-tested so the whole
baseline is correct-by-construction once SAM3D outputs are available; results are
cached by input hash so reruns are free.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import trimesh

from .. import frames
from ..data.base import Frame, GTObject
from ..eval import geometry as geo
from ..eval.metrics import SceneObject
from ..frames import T_cam_raw as _T_cam_raw  # direct ref: `frames` is shadowed in _run by list(source)
from ..frames import validate

log = logging.getLogger(__name__)

# --------------------------------------------------------------- placement math

def place_from_sam3d(mesh_raw: trimesh.Trimesh, scale, quat_wxyz, translation,
                     T_world_cam: np.ndarray) -> Tuple[trimesh.Trimesh, np.ndarray, np.ndarray]:
    """Place a raw SAM3D mesh into the world using its predicted layout.

    Uses the consolidated frame chain (``frames.T_cam_raw``) composed with the
    camera pose. Returns ``(posed_mesh_world, T_world_obj, extents)`` where the box
    pose/extents come from the min-volume OBB of the posed mesh (scale may be
    anisotropic, so the placement is a general affine, not a rigid transform).
    """
    T_world_raw = np.asarray(T_world_cam, dtype=np.float64) @ frames.T_cam_raw(scale, quat_wxyz, translation)
    Aw = T_world_raw[:3, :3]
    bw = T_world_raw[:3, 3]

    v = np.asarray(mesh_raw.vertices, dtype=np.float64)
    v_world = (Aw @ v.T).T + bw
    posed = trimesh.Trimesh(vertices=v_world, faces=mesh_raw.faces, process=False)

    try:
        obb = posed.bounding_box_oriented
        T_world_obj = np.asarray(obb.primitive.transform, dtype=np.float64)
        extents = np.asarray(obb.primitive.extents, dtype=np.float64)
    except Exception:  # degenerate mesh -> axis-aligned fallback
        lo, hi = v_world.min(0), v_world.max(0)
        T_world_obj = np.eye(4)
        T_world_obj[:3, 3] = 0.5 * (lo + hi)
        extents = np.maximum(hi - lo, 1e-6)
    return posed, T_world_obj, extents


# ---------------------------------------------------------- mask / best view

def render_instance_mask(mesh_world: trimesh.Trimesh, frame: Frame) -> Optional[np.ndarray]:
    """Silhouette mask (uint8 0/255) of a world-frame mesh in ``frame``.

    Projects all faces in front of the camera and fills them (union of triangles).
    Occlusion by other instances is not modeled here (GT masks isolate layout, not
    detection); depth-consistency filtering can be layered on if needed.
    """
    H, W = frame.depth.shape[:2] if frame.depth.ndim == 2 else frame.rgb.shape[:2]
    T_cam_world = np.linalg.inv(frame.T_world_cam)
    v = np.asarray(mesh_world.vertices, dtype=np.float64)
    v_cam = (T_cam_world[:3, :3] @ v.T).T + T_cam_world[:3, 3]
    z = v_cam[:, 2]
    fx, fy, cx, cy = frame.K[0, 0], frame.K[1, 1], frame.K[0, 2], frame.K[1, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = fx * v_cam[:, 0] / z + cx
        vv = fy * v_cam[:, 1] / z + cy
    px = np.stack([u, vv], axis=1)

    mask = np.zeros((H, W), np.uint8)
    tris = []
    for f in mesh_world.faces:
        if np.all(z[f] > 1e-6):
            tris.append(px[f].astype(np.int32))
    if not tris:
        return None
    cv2.fillPoly(mask, tris, 255)
    return mask if mask.any() else None


def _view_score(mask: np.ndarray) -> Tuple[float, bool]:
    """Return (visible_area_fraction, touches_border)."""
    H, W = mask.shape
    area = float((mask > 0).sum()) / (H * W)
    border = bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())
    return area, border


def select_best_view(gt_obj: GTObject, frames: List[Frame], mask_fn=None) -> Optional[int]:
    """Index (into ``frames``) of the best view: largest visible area, not touching
    the image border. Falls back to largest area if all views touch the border.

    ``mask_fn(gt_obj, frame) -> uint8 mask | None`` supplies the per-view mask; defaults
    to rendering the object's mesh silhouette (``render_instance_mask``). Datasets with
    true GT masks (e.g. ProcTHOR native instance seg) pass their own occlusion-aware fn.
    """
    if mask_fn is None:
        mask_fn = lambda g, fr: render_instance_mask(g.mesh, fr)
    best_i, best_area, best_i_any, best_area_any = None, -1.0, None, -1.0
    for i, fr in enumerate(frames):
        mask = mask_fn(gt_obj, fr)
        if mask is None:
            continue
        area, border = _view_score(mask)
        if area > best_area_any:
            best_area_any, best_i_any = area, i
        if not border and area > best_area:
            best_area, best_i = area, i
    return best_i if best_i is not None else best_i_any


def crop_bbox_from_mask(mask: np.ndarray, pad: int = 8) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    H, W = mask.shape
    x0 = max(int(xs.min()) - pad, 0)
    y0 = max(int(ys.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad, W - 1)
    y1 = min(int(ys.max()) + pad, H - 1)
    return x0, y0, x1, y1


# ----------------------------------------------------- SAM3D worker (gated)

def _default_queue() -> Path:
    return Path(os.environ.get("SAM3D_QUEUE", str(Path.home() / "Data" / "datasets" / "sam3d_queue")))


def _job_hash(rgb: np.ndarray, mask: np.ndarray, depth: np.ndarray, K: np.ndarray) -> str:
    """Content hash identifying a SAM3D job by its input pixels (disk datasets).

    Used as the FALLBACK job id when the caller can't provide a stable logical key
    (``job_key``). Fine for datasets whose frames are byte-reproducible (Replica reads
    RGB/depth from disk). NOT safe for live-rendered datasets: AI2-THOR RGB *and* a few
    depth edge pixels jitter across renders, so a content hash changes between the queue
    and collect passes and misses every cached output — pass ``job_key`` for those.
    """
    h = hashlib.sha256()
    for a in (rgb, mask, depth.astype(np.float32), K.astype(np.float64)):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def _stable_job_id(job_key: str) -> str:
    """Readable, filesystem-safe job id from a deterministic logical key.

    Keeps the queue browsable (e.g. ``procthor_137_i41_full``) and — crucially —
    reproducible across re-renders, since it depends on scene/instance identity, not on
    jittery rendered pixels.
    """
    import re
    return re.sub(r"[^A-Za-z0-9_.-]", "_", job_key)


def run_sam3d(rgb_crop, mask_crop, depth_crop, full_K, crop_bbox, meta: dict,
              queue: Optional[Path] = None, job_key: Optional[str] = None
              ) -> Optional[Tuple[trimesh.Trimesh, dict]]:
    """Get one SAM3D result via the disk-queue worker; write the job if not cached.

    Job identity: a stable logical ``job_key`` when provided (required for live-rendered
    datasets — see ``_stable_job_id``), else the pixel content hash (disk datasets).

    Returns ``(mesh_raw, pose)`` if the worker output exists (cached by job id), else
    writes the job under ``<queue>/input/<job_id>`` (idempotently) and returns ``None``
    — the caller runs the worker over the queue, then re-runs to collect.

    Job format matches the worker's ``load_job`` / ``depth_to_pointmap``: rgb/mask/
    depth are the CROP; ``camera_info.K`` is the **full-image** intrinsics and
    ``crop_bbox`` = [x0,y0,x1,y1] in full-image pixels, so the worker back-projects
    the crop depth in full-image coordinates. ``odometry`` is required by the worker
    (it stamps go2_odom_* into pose.json, which we ignore — we place via T_world_cam);
    for datasets with no robot we pass identity.
    """
    import json as _json

    queue = queue or _default_queue()
    job_id = _stable_job_id(job_key) if job_key else _job_hash(rgb_crop, mask_crop, depth_crop, full_K)
    out_dir = queue / "output" / job_id
    glb = out_dir / "object.glb"
    pose_json = out_dir / "pose.json"

    if glb.is_file() and pose_json.is_file():
        with open(pose_json) as f:
            pose = _json.load(f)
        mesh = trimesh.load(str(glb), process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])
        return mesh, pose

    # not cached: write the job (idempotent) for the worker to pick up
    in_dir = queue / "input" / job_id
    in_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(in_dir / "rgb.png"), cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(in_dir / "mask.png"), mask_crop)
    np.save(in_dir / "depth.npy", depth_crop.astype(np.float32))
    meta_out = {
        "job_id": job_id,
        "track_id": int(meta.get("track_id", 0)),
        "label": str(meta.get("label", "object")),
        "camera_info": {
            "K": [float(x) for x in np.asarray(full_K).reshape(-1)],
            "width": int(meta.get("full_width", rgb_crop.shape[1])),
            "height": int(meta.get("full_height", rgb_crop.shape[0])),
        },
        "crop_bbox": [int(v) for v in crop_bbox],
        # no robot on datasets: identity odom (worker requires the key)
        "odometry": {"position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
    }
    with open(in_dir / "meta.json", "w") as f:
        _json.dump(meta_out, f)
    return None


# ------------------------------------------------------------------ ICP (B)

def refine_icp(source_pts: np.ndarray, target_pts: np.ndarray, init_T: np.ndarray,
               voxel: float = 0.01, dist: float = 0.03, max_iter: int = 50,
               min_fitness: float = 0.10) -> Tuple[np.ndarray, dict]:
    """Point-to-point ICP (v1 parity) refining ``init_T`` (4x4). Returns
    ``(T_refined, info)``; falls back to ``init_T`` on low fitness (marked in info).
    """
    try:
        import open3d as o3d
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("variant B needs open3d: `uv sync --extra registration`") from e

    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(source_pts)))
    tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(target_pts)))
    src = src.voxel_down_sample(voxel) if len(source_pts) > 30 else src
    tgt = tgt.voxel_down_sample(voxel)
    if len(tgt.points) < 30:
        return init_T, {"fitness": 0.0, "fallback": "too_few_target_points"}
    reg = o3d.pipelines.registration.registration_icp(
        src, tgt, dist, init_T,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter),
    )
    if reg.fitness < min_fitness:
        return init_T, {"fitness": float(reg.fitness), "fallback": "low_fitness"}
    return np.asarray(reg.transformation), {"fitness": float(reg.fitness), "fallback": None}


def _masked_depth_cloud(frame: Frame, mask: np.ndarray) -> np.ndarray:
    """Back-project masked, valid depth pixels into the world frame."""
    H, W = frame.depth.shape
    fx, fy, cx, cy = frame.K[0, 0], frame.K[1, 1], frame.K[0, 2], frame.K[1, 2]
    ys, xs = np.where((mask > 0) & (frame.depth > 0) & np.isfinite(frame.depth))
    z = frame.depth[ys, xs]
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    pc = np.stack([x, y, z], axis=1)
    return (frame.T_world_cam[:3, :3] @ pc.T).T + frame.T_world_cam[:3, 3]


# ----------------------------------------------------------------- entry points

def _observed_obb_extent(target_pts: np.ndarray):
    """Sorted (desc) OBB extents of a point cloud, or None if degenerate."""
    if len(target_pts) < 30:
        return None
    try:
        e = np.asarray(trimesh.points.PointCloud(target_pts).bounding_box_oriented.primitive.extents)
    except Exception:
        e = np.asarray(target_pts).max(0) - np.asarray(target_pts).min(0)
    return np.sort(e)[::-1]


def _fast_obb(mesh: trimesh.Trimesh, n: int = 3000, seed: int = 0):
    """(transform, extents) of the mesh OBB from sampled surface points — much faster than
    ``mesh.bounding_box_oriented`` on SAM3D meshes (up to ~1M faces; the OBB is called
    several times per object in the scale/registration path)."""
    pts = geo.sample_surface(mesh, n, seed=seed) if len(mesh.faces) else np.asarray(mesh.vertices)
    p = trimesh.points.PointCloud(pts).bounding_box_oriented.primitive
    return np.asarray(p.transform, dtype=np.float64), np.asarray(p.extents, dtype=np.float64)


def _fit_scale_to_extent(posed: trimesh.Trimesh, target_extent_sorted: np.ndarray,
                         clamp=(0.25, 4.0)):
    """World transform that rescales ``posed`` so its OBB extents match the (metric,
    depth-derived) ``target_extent_sorted``, per axis, about the mesh's OBB centre.

    Depth gives the object's true metric size (verified: masked-depth extent == GT to
    ~1-2% on observed axes, thin axis recovered by multi-view fusion), so matching the
    SPAN sets scale correctly — unlike ICP (can't rescale) or TEASER point-NN (shrinks
    onto the partial cloud). Anisotropic + axis-sorted; scale factors clamped for safety.
    """
    T, ext_m = _fast_obb(posed)
    R, c = T[:3, :3], T[:3, 3]
    tgt = np.sort(np.asarray(target_extent_sorted, float))[::-1]
    order = np.argsort(ext_m)[::-1]          # mesh axes, largest -> smallest
    scales = np.ones(3)
    for rank, ax in enumerate(order):
        scales[ax] = np.clip(tgt[rank] / max(ext_m[ax], 1e-6), *clamp)
    A = R @ np.diag(scales) @ R.T            # scale along mesh OBB axes, about centre c
    M = np.eye(4)
    M[:3, :3] = A
    M[:3, 3] = c - A @ c
    return M, {"scales": scales.tolist(), "mesh_ext": ext_m.tolist(), "target_ext": tgt.tolist()}


def _run(source, gt: Optional[List[GTObject]], config: dict, registration: str) -> List[SceneObject]:
    """registration: "none" | "icp" (rigid) | "teaser" (Sim3) | "scale" (depth-extent
    scale-fit) | "scale_icp" (scale-fit + rigid ICP for pose)."""
    if not gt:
        return []
    frames = list(source)
    if not frames:
        log.warning("source yielded no frames")
        return []

    queue = Path(config["sam3d_queue"]) if config.get("sam3d_queue") else None
    # Mask source: prefer the dataset's TRUE instance masks (occlusion-aware, e.g.
    # ProcTHOR native seg) when available; else render the GT mesh silhouette.
    if getattr(source, "native_masks", False) and hasattr(source, "native_mask"):
        raw_mask_fn = lambda g, fr: source.native_mask(fr.frame_id, g.instance_id)
        mask_src = "native"
    else:
        raw_mask_fn = lambda g, fr: render_instance_mask(g.mesh, fr)
        mask_src = "rendered_mesh"

    # Reject degenerate masks (tiny/thin objects): SAM3D needs a >=2x2 crop and a sliver
    # mask yields garbage anyway. Objects whose every view is too small get no view and
    # are skipped (honest: too small to reconstruct). Tunable via config.
    min_px = int(config.get("min_mask_px", 64))
    min_dim = int(config.get("min_mask_dim", 6))

    def mask_fn(g, fr):
        m = raw_mask_fn(g, fr)
        if m is None:
            return None
        ys, xs = np.where(m > 0)
        if len(xs) < min_px:
            return None
        if (xs.max() - xs.min() + 1) < min_dim or (ys.max() - ys.min() + 1) < min_dim:
            return None
        return m
    preds: List[SceneObject] = []
    pending = 0
    teaser_stats = []  # per attempt: (accepted, scale, resid_before_m, resid_after_m, n_corr)
    placements = []    # per object: reconstruct the posed mesh from cached object.glb
    for g in gt:
        vi = select_best_view(g, frames, mask_fn)
        if vi is None:
            log.info("no visible view for instance %d (%s); skipping", g.instance_id, g.label)
            continue
        frame = frames[vi]
        mask = mask_fn(g, frame)
        H, W = frame.depth.shape[:2]

        # Framing fed to SAM3D. Default is the FULL frame: an ablation (room0, 43 obj)
        # showed feeding the whole image + whole-scene depth pointmap cuts GT scale error
        # 0.32 -> 0.10 (better on 38/43) vs a tight bbox crop, which starves SAM3D's
        # pointmap normalization of scene context. `config["full_frame"]=False` restores
        # the crop for comparison. See results/ablation_full_vs_crop/ + STATUS.md.
        if config.get("full_frame", True):
            rgb_in, mask_in, depth_in = frame.rgb, mask, frame.depth
            bbox_in = (0, 0, W - 1, H - 1)
        else:
            x0, y0, x1, y1 = crop_bbox_from_mask(mask)
            rgb_in = frame.rgb[y0:y1 + 1, x0:x1 + 1]
            mask_in = mask[y0:y1 + 1, x0:x1 + 1]
            depth_in = frame.depth[y0:y1 + 1, x0:x1 + 1]
            bbox_in = (x0, y0, x1, y1)

        # Stable logical job id: identity by (source, scene, instance, framing), so the
        # queue and collect passes agree even when the renderer's pixels jitter. Falls
        # back to the pixel content hash inside run_sam3d if job_key is None.
        scene = getattr(source, "scene", "scene")
        framing = "full" if config.get("full_frame", True) else "crop"
        # mask_src is part of the identity: native vs box mask feed SAM3D different inputs
        # -> different meshes, so they must NOT collide in the cache.
        job_key = f"{config.get('source', 'src')}_{scene}_i{g.instance_id}_{framing}_{mask_src}"
        result = run_sam3d(
            rgb_in, mask_in, depth_in, frame.K, bbox_in,
            meta={"track_id": g.instance_id, "label": g.label,
                  "full_width": W, "full_height": H},
            queue=queue, job_key=job_key,
        )
        if result is None:
            pending += 1
            continue
        mesh_raw, pose = result
        # validate SAM3D outputs at the boundary; stamp provenance, don't drop silently
        val = validate.check_sam3d_scale(pose["sam3d_scale"], raise_on_fail=False)
        val = val or validate.check_translation(pose["sam3d_translation"], raise_on_fail=False)
        if val:
            log.warning("instance %d (%s): suspect SAM3D output — %s", g.instance_id, g.label, val)
        posed, T_world_obj, extents = place_from_sam3d(
            mesh_raw, pose["sam3d_scale"], pose["sam3d_rotation"], pose["sam3d_translation"],
            frame.T_world_cam,
        )
        # Track the raw-mesh -> world transform so a placement can be reconstructed later
        # (load the cached object.glb, apply T_world_mesh) without re-running placement.
        # `post` accumulates the registration deltas applied to `posed` after the layout.
        T_world_raw = frame.T_world_cam @ _T_cam_raw(
            pose["sam3d_scale"], pose["sam3d_rotation"], pose["sam3d_translation"])
        post = np.eye(4)
        prov = {"source": "sam3d", "registration": "layout", "view_index": vi,
                "validation": val, "mask_src": mask_src,
                "instance_id": g.instance_id}

        if registration != "none":
            # Registration target = the object's masked sensor depth, back-projected to
            # world. Default is the best-view sliver (~20k pts); config["icp_accumulate"]
            # fuses masked depth over ALL views (~28x more pts, a fuller surface — proto
            # multi-view ObjectTrack). TEASER wants the fuller cloud for scale + FPFH, so
            # it accumulates by default.
            # scale-fit + teaser want the fuller multi-view cloud (thin axis recovery).
            accumulate = (config.get("icp_accumulate")
                          or registration in ("teaser", "scale", "scale_icp"))
            target = _masked_depth_cloud(frame, mask)
            if accumulate:
                extra = [target]
                for fr in frames:
                    if fr is frame:
                        continue
                    m = mask_fn(g, fr)
                    if m is not None and int((m > 0).sum()) > 50:
                        extra.append(_masked_depth_cloud(fr, m))
                target = np.concatenate(extra, axis=0)
                prov["reg_target"] = f"accumulated_{len(extra)}views"
            else:
                prov["reg_target"] = "singleview"

            do_scale = registration in ("scale", "scale_icp")
            do_icp = registration in ("icp", "scale_icp")

            if do_scale:
                # Set metric scale from the depth-cloud extent (the observed metric size),
                # then rigid ICP (if any) fixes pose only. This is the scale lever ICP lacks.
                tgt_ext = _observed_obb_extent(target)
                if tgt_ext is not None:
                    M, sinfo = _fit_scale_to_extent(posed, tgt_ext)
                    posed.apply_transform(M)
                    post = M @ post
                    prov["scale_fit"] = sinfo
                else:
                    prov["scale_fit"] = {"skipped": "degenerate_target"}

            if do_icp:
                # rigid: recovers pose only (cannot rescale). Delta is a world-frame SE(3).
                src_pts = (geo.sample_surface(posed, 2000, seed=g.instance_id)
                           if len(posed.faces) else np.asarray(posed.vertices))
                delta, info = refine_icp(src_pts, target, np.eye(4))
                posed.apply_transform(delta)
                post = delta @ post
                prov["icp"] = info

            if do_scale or do_icp:
                prov["registration"] = "layout+" + registration

            if registration == "teaser":  # Sim(3): also estimates SCALE (ICP-can't-fix residual)
                src_pts = (geo.sample_surface(posed, 2000, seed=g.instance_id)
                           if len(posed.faces) else np.asarray(posed.vertices))
                from ..registration.teaser import register_teaser
                from scipy.spatial import cKDTree
                voxel = float(config.get("teaser_voxel", 0.03))
                delta, info = register_teaser(src_pts, target, voxel=voxel)
                accepted = False
                if info.get("ok"):
                    # SAFETY GATE: accept the Sim(3) only if it actually improves the
                    # mesh->depth alignment vs the layout pose. FPFH corr on a hallucinated
                    # mesh are often outliers -> garbage transforms; the gate makes bad
                    # registrations a no-op (keep layout) so registration can't regress.
                    tree = cKDTree(target)
                    r_before = float(np.median(tree.query(src_pts)[0]))
                    src_reg = (delta[:3, :3] @ src_pts.T).T + delta[:3, 3]
                    r_after = float(np.median(tree.query(src_reg)[0]))
                    info["resid_before_m"], info["resid_after_m"] = r_before, r_after
                    if r_after < r_before:
                        posed.apply_transform(delta)
                        post = delta @ post
                        accepted = True
                    teaser_stats.append((accepted, info.get("scale"), r_before, r_after,
                                         info.get("n_corr")))
                else:
                    log.info("instance %d (%s): TEASER no solution (%s) — keeping layout",
                             g.instance_id, g.label, info.get("reason"))
                    teaser_stats.append((False, None, None, None, info.get("n_corr", 0)))
                info["accepted"] = accepted
                prov["registration"] = "layout+teaser" if accepted else "layout(teaser_rejected)"
                prov["teaser"] = info

            # One OBB (from sampled points) after all transforms -> pose + extents. delta
            # may carry scale, so re-derive the OBB pose rather than composing with delta.
            T_world_obj, extents = _fast_obb(posed, seed=g.instance_id)

        preds.append(SceneObject(label=g.label, T_world_obj=T_world_obj, extents=extents,
                                 mesh=posed, provenance=prov))
        # Persist enough to rebuild the posed mesh later without re-running placement:
        # load <queue>/output/<job_id>/object.glb and apply T_world_mesh (raw verts->world).
        placements.append({
            "instance_id": int(g.instance_id), "label": g.label,
            "job_id": _stable_job_id(job_key), "registration": prov["registration"],
            "best_frame_id": int(frame.frame_id), "view_index": int(vi),
            "T_world_mesh": (post @ T_world_raw).tolist(),   # raw object.glb verts -> world
            "T_world_obj": np.asarray(T_world_obj, float).tolist(),
            "extents": np.asarray(extents, float).tolist(),
            "scale_fit": prov.get("scale_fit"), "icp": prov.get("icp"),
        })
    if pending:
        q = queue or _default_queue()
        log.warning(
            "%d SAM3D job(s) pending in %s/input. Run the worker (conda sam3d-objects), "
            "then re-run this method to collect (outputs are input-hash cached):\n"
            "  conda run -n sam3d-objects python "
            "humble_ws/src_Real2USD/real2sam3d/scripts_sam3d_worker/run_sam3d_worker.py "
            "--no-current-run --use-depth --queue-dir %s --sam3d-repo "
            "humble_ws/src_Real2USD/real2sam3d/sam-3d-objects", pending, q, q,
        )

    # Surface TEASER acceptance diagnostics into run.json (per-scene) so we can judge
    # whether the gated Sim(3) does anything: how often it beats layout, and by how much.
    if registration == "teaser" and teaser_stats:
        acc = [s for s in teaser_stats if s[0]]
        improves = [s[2] - s[3] for s in acc]  # resid_before - resid_after (m)
        stats = {
            "teaser_attempts": len(teaser_stats),
            "teaser_accepted": len(acc),
            "teaser_accept_rate": len(acc) / len(teaser_stats),
            "teaser_scale_median_accepted": float(np.median([s[1] for s in acc])) if acc else float("nan"),
            "teaser_resid_improve_median_m": float(np.median(improves)) if improves else 0.0,
            "teaser_ncorr_median": float(np.median([s[4] for s in teaser_stats if s[4] is not None]))
            if any(s[4] is not None for s in teaser_stats) else 0.0,
        }
        config.setdefault("_method_stats", {})[str(getattr(source, "scene", "scene"))] = stats
        log.warning("[%s] TEASER gate: %d/%d accepted (%.0f%%), median accepted scale %.2f, "
                    "median resid improve %.1f cm", getattr(source, "scene", "?"),
                    stats["teaser_accepted"], stats["teaser_attempts"],
                    100 * stats["teaser_accept_rate"], stats["teaser_scale_median_accepted"],
                    100 * stats["teaser_resid_improve_median_m"])

    # Hand per-object placements to the runner to persist (results/<run>/placements.json):
    # each has T_world_mesh (raw object.glb verts -> world) so any visualizer can rebuild
    # the posed prediction from the cached SAM3D mesh without re-running placement.
    if placements:
        config.setdefault("_placements", {})[str(getattr(source, "scene", "scene"))] = {
            "sam3d_queue": str(queue) if queue else None,
            "method": config.get("method"),
            "objects": placements,
        }
    return preds


def sam3d_layout(source, gt, config) -> List[SceneObject]:
    return _run(source, gt, config, registration="none")


def sam3d_layout_icp(source, gt, config) -> List[SceneObject]:
    return _run(source, gt, config, registration="icp")


def sam3d_layout_teaser(source, gt, config) -> List[SceneObject]:
    """SAM3D layout + TEASER++ Sim(3) registration (recovers scale, unlike rigid ICP)."""
    return _run(source, gt, config, registration="teaser")


def sam3d_layout_scale(source, gt, config) -> List[SceneObject]:
    """SAM3D layout + depth-extent scale-fit (set metric scale from the masked-depth OBB)."""
    return _run(source, gt, config, registration="scale")


def sam3d_layout_scale_icp(source, gt, config) -> List[SceneObject]:
    """SAM3D layout + depth-extent scale-fit + rigid ICP for pose (the full scale+pose fix)."""
    return _run(source, gt, config, registration="scale_icp")
