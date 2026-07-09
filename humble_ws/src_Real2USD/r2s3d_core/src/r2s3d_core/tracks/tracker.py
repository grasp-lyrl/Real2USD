"""The tracker driver: fold a posed RGB-D sequence + cached detections into a set of
persistent :class:`ObjectTrack`s.

Per frame (in trajectory order): build a candidate observation from every detection
(mask centroid, appearance, view direction, base view score), run the association
cascade, update matched tracks (fuse masked depth into the 1 cm voxel cloud, add the
view to the diverse top-K buffer, vote the label), and spawn TENTATIVE tracks for
unmatched detections. After the sequence, run late merge and finalize states.

ROS-free and torch-free — consumes a :class:`DetectionSet`, not a live detector.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from ..data.base import Frame
from . import associate, fusion, view
from .appearance import Appearance, HSVHistogram
from .types import (MAX_KEPT_VIEWS, MIN_ACTIVE_OBS, MIN_MATURE_VIEWS, Observation,
                    ObjectTrack, TrackState)

log = logging.getLogger(__name__)


def _crop(rgb: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = (int(round(v)) for v in bbox)
    H, W = rgb.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1 + 1), min(H, y1 + 1)
    if x1 <= x0 or y1 <= y0:
        return rgb[:1, :1]
    return rgb[y0:y1, x0:x1]


def _build_observation(det, frame: Frame, frame_index: int,
                       appearance: Appearance) -> Optional[Observation]:
    centroid = fusion.mask_centroid_world(frame, det.mask)
    if centroid is None:
        return None  # no valid depth under the mask -> cannot localize this detection
    crop = _crop(frame.rgb, det.bbox)
    emb = appearance.embed(frame.rgb, det.mask)
    return Observation(
        frame_index=frame_index,
        frame_id=int(frame.frame_id),
        stamp=float(frame.stamp),
        bbox=np.asarray(det.bbox, np.float64),
        det_label=str(det.label),
        det_score=float(det.score),
        det_track_id=int(det.track_id),
        centroid_world=centroid,
        appearance=emb,
        view_dir_world=frame.T_world_cam[:3, 2].copy(),  # camera +z (optical axis) in world
        view_score=view.base_view_score(det.mask, crop),
        n_mask_px=int(det.mask.sum()),
        mask=np.asarray(det.mask, bool),
        rgb_crop=crop.copy(),
    )


def _update_track(track: ObjectTrack, obs: Observation, frame: Frame) -> None:
    track.observations.append(obs)
    track.det_track_ids.add(obs.det_track_id)
    track.label_votes[obs.det_label] += 1
    track.add_appearance(obs.appearance)
    track.last_seen_stamp = obs.stamp
    # fuse masked depth into the voxel cloud (only for ACTIVE+; PHASE_SPECS)
    if track.state != TrackState.TENTATIVE and obs.mask is not None:
        track.voxels.add(fusion.backproject_mask(frame, obs.mask))
        track.fused_cloud = track.voxels.points()
    if len(track.fused_cloud):
        track.centroid = track.fused_cloud.mean(0)
    else:
        cents = np.array([o.centroid_world for o in track.observations])
        track.centroid = cents.mean(0)
    view.insert_kept_view(track.kept_views, obs, MAX_KEPT_VIEWS)


def _promote(track: ObjectTrack) -> None:
    if track.state == TrackState.TENTATIVE and track.n_obs >= MIN_ACTIVE_OBS:
        track.state = TrackState.ACTIVE
        # backfill the cloud from kept views now that we are ACTIVE (TENTATIVE didn't fuse)


def run_tracker(frames: List[Frame], detections_by_frame: Dict[int, list],
                config: dict, appearance: Optional[Appearance] = None) -> List[ObjectTrack]:
    """Run the tracker over an in-memory frames list + per-frame detections.

    ``config`` keys: ``reid`` (bool, step-2 re-ID on/off), ``late_merge`` (bool).
    """
    appearance = appearance or HSVHistogram()
    tracks: List[ObjectTrack] = []
    next_id = 0

    for fi, frame in enumerate(frames):
        dets = detections_by_frame.get(int(frame.frame_id), [])
        obs_list = []
        keep_idx = []
        for det in dets:
            obs = _build_observation(det, frame, fi, appearance)
            if obs is not None:
                obs_list.append(obs)
        if obs_list or tracks:
            matches, unmatched = associate.associate_frame(tracks, obs_list, frame, config)
        else:
            matches, unmatched = {}, list(range(len(obs_list)))

        for i, track in matches.items():
            was_tentative = track.state == TrackState.TENTATIVE
            _update_track(track, obs_list[i], frame)
            _promote(track)
            # on promotion TENTATIVE->ACTIVE, fold in the views seen while tentative
            if was_tentative and track.state == TrackState.ACTIVE:
                for o in track.kept_views:
                    if o.mask is not None:
                        track.voxels.add(fusion.backproject_mask(frames[o.frame_index], o.mask))
                track.fused_cloud = track.voxels.points()
                if len(track.fused_cloud):
                    track.centroid = track.fused_cloud.mean(0)

        for i in unmatched:
            t = ObjectTrack(track_id=next_id, voxels=fusion.VoxelCloud(
                config.get("voxel", 0.01)))
            next_id += 1
            _update_track(t, obs_list[i], frame)
            tracks.append(t)

    # finalize: ACTIVE tracks with enough views (or by sequence end) become MATURE
    for t in tracks:
        if t.state in (TrackState.MERGED, TrackState.REJECTED):
            continue
        if t.state == TrackState.ACTIVE and (len(t.kept_views) >= MIN_MATURE_VIEWS
                                             or t.n_obs >= MIN_MATURE_VIEWS):
            t.state = TrackState.MATURE
        elif t.state == TrackState.ACTIVE:
            t.state = TrackState.MATURE  # sequence end matures all ACTIVE tracks
        elif t.state == TrackState.TENTATIVE:
            t.state = TrackState.REJECTED  # never reached ACTIVE -> insufficient evidence

    if config.get("late_merge", True):
        tracks = associate.late_merge(tracks, config)

    n_mature = sum(1 for t in tracks if t.state == TrackState.MATURE)
    log.info("tracker: %d tracks (%d mature) over %d frames",
             len([t for t in tracks if t.state != TrackState.MERGED]), n_mature, len(frames))
    return tracks
