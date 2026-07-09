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
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import fusion
from .appearance import cosine
from .types import ObjectTrack, TrackState

_TERMINAL = (TrackState.MERGED, TrackState.REJECTED)

# gates / weights
CENTROID_GATE_MIN_M = 0.5
DRIFT_RATE_M_PER_S = 0.02
APPEARANCE_FLOOR = 0.75
W_HULL_IOU, W_APPEARANCE, W_CENTROID = 0.5, 0.3, 0.2
MIN_CLOUD_VOXELS_FOR_HULL = 30

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
                    b = _pair_benefit(obs, t, frame)
                    if b is not None:
                        benefit[r, c] = b
            rows, cols = linear_sum_assignment(-benefit)
            for r, c in zip(rows, cols):
                if benefit[r, c] > 0.0:
                    matches[remaining[r]] = cand[c]
                    used.add(id(cand[c]))

    unmatched = [i for i in remaining if i not in matches]
    return matches, unmatched


def _pair_benefit(obs, track: ObjectTrack, frame):
    """Association benefit for (obs, track), or None if a hard gate fails."""
    if track.centroid is None:
        return None
    dt = max(obs.stamp - track.last_seen_stamp, 0.0)
    gate = max(CENTROID_GATE_MIN_M, DRIFT_RATE_M_PER_S * dt)
    dist = float(np.linalg.norm(obs.centroid_world - track.centroid))
    if dist > gate:
        return None
    acos = cosine(obs.appearance, track.appearance_mean)
    if acos < APPEARANCE_FLOOR:
        return None
    hull_iou = 0.0
    vox = getattr(track, "voxels", None)
    if vox is not None and vox.n >= MIN_CLOUD_VOXELS_FOR_HULL and obs.mask is not None:
        hull = fusion.project_cloud_mask(track.fused_cloud, frame)
        hull_iou = fusion.mask_iou(obs.mask, hull)
    centroid_score = max(0.0, 1.0 - dist / gate)
    return W_HULL_IOU * hull_iou + W_APPEARANCE * acos + W_CENTROID * centroid_score


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
