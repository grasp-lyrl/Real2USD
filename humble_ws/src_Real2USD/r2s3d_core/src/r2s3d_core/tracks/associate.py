"""Association cascade and late merge (docs/PHASE_SPECS.md §Phase 2).

Association (first match wins):
  1. detector track id;
  2. Hungarian over cost = 1 - [0.5·IoU(det mask, track's reprojected-cloud hull) +
     0.3·appearance_cos + 0.2·centroid_gate], with gates: centroid dist <
     max(0.5 m, 0.02 m/s · seconds_since_last_seen) (drift-aware) and
     appearance_cos > 0.75 hard floor.

Late merge (reconciliation): candidate pairs by centroid < 1 m; merge if cloud-OBB
IoU > 0.3 ∨ (voxel overlap > 50% ∧ appearance_cos > 0.85). NOTE: PHASE_SPECS's
registered-*mesh*-IoU criterion needs Phase-3 registration; Phase 2 substitutes the
fused-cloud geometry (recorded in the run provenance).

``appearance_cos`` replaces PHASE_SPECS's ``clip_cos`` via the Appearance interface.
Step-2 re-ID (and thus break-healing) is disabled when ``config['reid']`` is False —
that is the ``object_track_naive`` foil.

``config['assoc_reproj']`` (default off) swaps the step-2 position gate from the legacy
isotropic 0.5 m world-space ball to an anisotropic reprojection gate — tight in pixels
(perpendicular to the ray), loose in depth ratio (along it) — since mask-median-depth
centroid error is dominated by the along-ray component. Adapted from SuperMap (RSS'26);
see docs/RELATED_WORK.md and PHASE_SPECS §Perception robustness 6a.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import fusion
from .appearance import cosine
from .types import ObjectTrack, TrackState

_TERMINAL = (TrackState.MERGED, TrackState.REJECTED)

# gates / weights (legacy isotropic world-space gate)
CENTROID_GATE_MIN_M = 0.5
DRIFT_RATE_M_PER_S = 0.02
APPEARANCE_FLOOR = 0.75
W_HULL_IOU, W_APPEARANCE, W_CENTROID = 0.5, 0.3, 0.2
MIN_CLOUD_VOXELS_FOR_HULL = 30

# reprojection-space gate (config['assoc_reproj'], PHASE_SPECS §Perception robustness 6a;
# adapted from SuperMap RSS'26, see docs/RELATED_WORK.md). Mask-median-depth centroid error
# is dominated by the along-ray component, so we gate anisotropically: tight in pixels
# (perpendicular to the ray), loose in depth ratio (along the ray).
#
# !! PROVISIONAL / NOT TUNED !! REPROJ_PIX_GATE and DEPTH_RATIO_TOL are first-guess values
# (60 px reverse-engineered from a synthetic test, 0.35 likewise). They trade fragmentation
# (too tight -> true re-IDs rejected) against ID-swaps/false-merges (too loose -> distinct
# nearby objects fused). The operating point is dataset-dependent and MUST be swept on
# scene 200 (then 137/428) against GT before these are trusted as defaults. Override per-run
# via config['reproj_pix_gate'] / config['reproj_depth_ratio_tol'] (eval.run: --reproj-pix-gate
# / --reproj-depth-ratio-tol) so a sweep needs no code edit. Owner: coworker. See STATUS §NEXT.
REPROJ_PIX_GATE = 60.0        # px; perpendicular-to-ray tolerance
DEPTH_RATIO_TOL = 0.35        # along-ray: obs_z/pred_z within [1/(1+τ), 1+τ]
W_HULL_IOU_R, W_APPEARANCE_R, W_REPROJ = 0.5, 0.2, 0.3

# late merge
MERGE_CENTROID_M = 1.0
MERGE_CLOUD_IOU = 0.3
MERGE_OVERLAP = 0.5
MERGE_APPEARANCE = 0.85


def associate_frame(tracks: List[ObjectTrack], obs_list: list, frame, config: dict
                    ) -> Tuple[Dict[int, ObjectTrack], List[int]]:
    """Return ``(matches, unmatched_obs_indices)`` where ``matches`` maps an index
    into ``obs_list`` to the track it was associated with."""
    reid = config.get("reid", True)
    matches: Dict[int, ObjectTrack] = {}
    used = set()

    # ---- step 1: detector track id
    id_index: Dict[int, ObjectTrack] = {}
    for t in tracks:
        if t.state in _TERMINAL:
            continue
        for did in t.det_track_ids:
            id_index.setdefault(did, t)
    remaining: List[int] = []
    for i, obs in enumerate(obs_list):
        t = id_index.get(obs.det_track_id) if obs.det_track_id >= 0 else None
        if t is not None and id(t) not in used:
            matches[i] = t
            used.add(id(t))
        else:
            remaining.append(i)

    # ---- step 2: Hungarian re-ID over remaining
    if reid and remaining:
        cand = [t for t in tracks if t.state not in _TERMINAL and id(t) not in used]
        if cand:
            benefit = np.full((len(remaining), len(cand)), -1.0)
            for r, i in enumerate(remaining):
                obs = obs_list[i]
                for c, t in enumerate(cand):
                    b = _pair_benefit(obs, t, frame, config)
                    if b is not None:
                        benefit[r, c] = b
            rows, cols = linear_sum_assignment(-benefit)
            for r, c in zip(rows, cols):
                if benefit[r, c] > 0.0:
                    matches[remaining[r]] = cand[c]
                    used.add(id(cand[c]))

    unmatched = [i for i in remaining if i not in matches]
    return matches, unmatched


def _pair_benefit(obs, track: ObjectTrack, frame, config: dict):
    """Association benefit for (obs, track), or None if a hard gate fails.

    ``config['assoc_reproj']`` switches the position gate from the legacy isotropic
    world-space ball to the anisotropic reprojection gate (PHASE_SPECS 6a)."""
    if track.centroid is None:
        return None
    acos = cosine(obs.appearance, track.appearance_mean)
    if acos < APPEARANCE_FLOOR:
        return None
    hull_iou = _hull_iou(obs, track, frame)

    if config.get("assoc_reproj"):
        reproj = _reproj_score(track, obs, frame, config)
        if reproj is None:
            return None
        return W_HULL_IOU_R * hull_iou + W_APPEARANCE_R * acos + W_REPROJ * reproj

    # legacy isotropic world-space centroid gate
    dt = max(obs.stamp - track.last_seen_stamp, 0.0)
    gate = max(CENTROID_GATE_MIN_M, DRIFT_RATE_M_PER_S * dt)
    dist = float(np.linalg.norm(obs.centroid_world - track.centroid))
    if dist > gate:
        return None
    centroid_score = max(0.0, 1.0 - dist / gate)
    return W_HULL_IOU * hull_iou + W_APPEARANCE * acos + W_CENTROID * centroid_score


def _hull_iou(obs, track: ObjectTrack, frame) -> float:
    """IoU of the detection mask with the track's reprojected fused-cloud silhouette
    (0.0 when the cloud is too sparse to reproject or the obs has no mask)."""
    vox = getattr(track, "voxels", None)
    if vox is not None and vox.n >= MIN_CLOUD_VOXELS_FOR_HULL and obs.mask is not None:
        hull = fusion.project_cloud_mask(track.fused_cloud, frame)
        return fusion.mask_iou(obs.mask, hull)
    return 0.0


def _project_to_cam(point_world: np.ndarray, frame):
    """Project a world point into ``frame``; return ``(u_px, v_px, z_cam)`` or None if
    it is behind the camera. Mirrors ``fusion.project_cloud_mask``'s math for one point."""
    T_cam_world = np.linalg.inv(frame.T_world_cam)
    pc = T_cam_world[:3, :3] @ point_world + T_cam_world[:3, 3]
    if pc[2] <= 1e-6:
        return None
    fx, fy, cx, cy = frame.K[0, 0], frame.K[1, 1], frame.K[0, 2], frame.K[1, 2]
    return fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy, float(pc[2])


def _reproj_score(track: ObjectTrack, obs, frame, config: dict):
    """Anisotropic reprojection gate. Compare the track's predicted image location
    (reprojected 3D centroid) against the observation's: tight in pixels (perpendicular
    to the viewing ray), loose in depth ratio (along it). Returns a score in [0, 1] or
    None if a gate fails / either centroid is behind the camera.

    Tolerances are overridable per run for tuning (see the PROVISIONAL note at the module
    constants): ``config['reproj_pix_gate']`` and ``config['reproj_depth_ratio_tol']``."""
    pix_gate = config.get("reproj_pix_gate") or REPROJ_PIX_GATE
    ratio_tol = config.get("reproj_depth_ratio_tol") or DEPTH_RATIO_TOL
    pred = _project_to_cam(track.centroid, frame)
    seen = _project_to_cam(obs.centroid_world, frame)
    if pred is None or seen is None:
        return None
    pix = float(np.hypot(pred[0] - seen[0], pred[1] - seen[1]))
    if pix > pix_gate:
        return None
    ratio = seen[2] / pred[2]
    if not (1.0 / (1.0 + ratio_tol) <= ratio <= 1.0 + ratio_tol):
        return None
    return max(0.0, 1.0 - pix / pix_gate)


# ------------------------------------------------------------------- late merge

def late_merge(tracks: List[ObjectTrack], config: dict) -> List[ObjectTrack]:
    """Merge over-fragmented tracks in place; the loser of each merge is marked
    MERGED. Returns the SAME full list (terminal tracks included) so callers/debug
    tooling can still see MERGED/REJECTED tracks; filter by state downstream.
    Greedy: repeatedly fold the best-scoring eligible pair."""
    from .view import insert_kept_view

    def alive(t):
        return t.state not in _TERMINAL and t.centroid is not None

    changed = True
    while changed:
        changed = False
        live = [t for t in tracks if alive(t)]
        best = None
        for a in range(len(live)):
            for b in range(a + 1, len(live)):
                ta, tb = live[a], live[b]
                if float(np.linalg.norm(ta.centroid - tb.centroid)) > MERGE_CENTROID_M:
                    continue
                iou = fusion.cloud_iou(ta.fused_cloud, tb.fused_cloud)
                ov = fusion.cloud_overlap(ta.voxels, tb.voxels) if (
                    ta.voxels is not None and tb.voxels is not None) else 0.0
                acos = cosine(ta.appearance_mean, tb.appearance_mean)
                if iou > MERGE_CLOUD_IOU or (ov > MERGE_OVERLAP and acos > MERGE_APPEARANCE):
                    score = max(iou, ov)
                    if best is None or score > best[0]:
                        best = (score, ta, tb, {"cloud_iou": iou, "overlap": ov, "appearance": acos})
        if best is not None:
            _, ta, tb, why = best
            _merge_into(ta, tb, why, insert_kept_view)
            changed = True

    return tracks


def _merge_into(dst: ObjectTrack, src: ObjectTrack, why: dict, insert_kept_view) -> None:
    """Fold ``src`` into ``dst``; ``src`` becomes MERGED."""
    dst.voxels.add(src.fused_cloud)
    dst.fused_cloud = dst.voxels.points()
    dst.centroid = dst.fused_cloud.mean(0) if len(dst.fused_cloud) else dst.centroid
    dst.observations.extend(src.observations)
    dst.label_votes.update(src.label_votes)
    if src.appearance_sum is not None:
        if dst.appearance_sum is None:
            dst.appearance_sum = np.zeros_like(src.appearance_sum)
        dst.appearance_sum += src.appearance_sum
        dst.n_appearance += src.n_appearance
    dst.det_track_ids |= src.det_track_ids
    dst.last_seen_stamp = max(dst.last_seen_stamp, src.last_seen_stamp)
    for o in src.kept_views:
        insert_kept_view(dst.kept_views, o)
    dst.merged_from.append(src.track_id)
    dst.merged_from.extend(src.merged_from)
    src.state = TrackState.MERGED
