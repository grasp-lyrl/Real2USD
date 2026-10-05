"""Coverage decomposition for the real-robot ObjectTrack pipeline.

Question (paper C2): detection recall is high (hallway-1 stride-2 gt: 0.88@1m) but only
~0.58 of GT produce a MATURE track. Where do objects leak between "detected" and
"matured"? This decomposes each GT into:

  D never-detected   : 0 observations within tau of the GT (trajectory/detector coverage)
  A recalled         : >=1 MATURE track within tau
  B fragmentation    : NOT recalled, but total obs near GT >= MIN_MATURE_VIEWS, split
                       across sub-threshold / non-mature tracks -> association COULD save it
  C sparsity         : NOT recalled, total obs near GT < MIN_MATURE_VIEWS -> too few
                       detections to mature (needs denser sampling or a lower threshold,
                       NOT better association)

Each detector observation is assigned to its single nearest GT within tau so support
counts don't double. Observations are deduped by (frame_id, det_track_id).

Run:  uv run python scripts/rs_coverage_diag.py [scene] [stride]
"""
import sys, logging
from collections import Counter, defaultdict

import numpy as np

logging.basicConfig(level=logging.WARNING)
from r2s3d_core.data.registry import make_source          # noqa: E402
from r2s3d_core.detect.cache import DetectionSet           # noqa: E402
from r2s3d_core.tracks import (TrackState, run_tracker,                 # noqa: E402
                               MIN_MATURE_VIEWS, MIN_ACTIVE_OBS)
# NOTE: in batch eval the finalize step matures EVERY ACTIVE track at sequence end
# (tracker.py: the MIN_MATURE_VIEWS branch is redundant), so the real maturation gate is
# MIN_ACTIVE_OBS (a single track needs >= this many obs to leave TENTATIVE). Decompose
# against that gate, not MIN_MATURE_VIEWS.
GATE = MIN_ACTIVE_OBS

SCENE = sys.argv[1] if len(sys.argv) > 1 else "hallway-1"
STRIDE = int(sys.argv[2]) if len(sys.argv) > 2 else 2

src = make_source("realsense", SCENE, stride=STRIDE)
frames = list(src)
gt = src.gt()
ds = DetectionSet.load(f"results/detections/gt/{SCENE}", scene=SCENE)
gt_c = np.array([g.T_world_obj[:3, 3] for g in gt])        # (Ngt,3) box centers
print(f"scene={SCENE} stride={STRIDE} frames={len(frames)} detections={len(ds.detections)} "
      f"gt={len(gt)} gt_labels={dict(Counter(g.label for g in gt))}")

tracks = run_tracker(frames, ds.by_frame(), {"reid": True, "late_merge": True})
by_state = Counter(t.state.value for t in tracks)
mature = [t for t in tracks if t.state == TrackState.MATURE]
print(f"tracks={len(tracks)} states={dict(by_state)} mature={len(mature)} "
      f"MIN_MATURE_VIEWS={MIN_MATURE_VIEWS}")

# --- collect all detector observations (deduped) with world centroid ---
seen = set()
obs_xy, obs_track = [], []
for t in tracks:
    for o in t.observations:
        key = (int(o.frame_id), int(o.det_track_id))
        if key in seen:
            continue
        seen.add(key)
        obs_xy.append(o.centroid_world[:2])
        obs_track.append(t)
obs_xy = np.array(obs_xy) if obs_xy else np.zeros((0, 2))
print(f"unique detector observations placed in world: {len(obs_xy)}")


def decompose(tau):
    # nearest-GT assignment for each obs -> per-GT support count
    support = np.zeros(len(gt), dtype=int)
    if len(obs_xy):
        for xy in obs_xy:
            d = np.linalg.norm(gt_c[:, :2] - xy, axis=1)
            j = int(np.argmin(d))
            if d[j] <= tau:
                support[j] += 1
    # mature tracks per GT (greedy one-to-one so over-seg doesn't inflate recall)
    mt_xy = np.array([t.centroid[:2] for t in mature]) if mature else np.zeros((0, 2))
    recalled = np.zeros(len(gt), dtype=bool)
    used = set()
    for i in range(len(gt)):
        best, bd = -1, tau
        for j in range(len(mt_xy)):
            if j in used:
                continue
            dd = float(np.linalg.norm(mt_xy[j] - gt_c[i, :2]))
            if dd < bd:
                bd, best = dd, j
        if best >= 0:
            used.add(best); recalled[i] = True
    # per-GT: best single mature-track obs count near it, and #tracks near it (any state)
    best_mat_obs = np.zeros(len(gt), dtype=int)   # largest n_obs among mature tracks near GT
    n_tracks_near = np.zeros(len(gt), dtype=int)
    for i in range(len(gt)):
        for t in tracks:
            if t.centroid is None:
                continue
            if float(np.linalg.norm(t.centroid[:2] - gt_c[i, :2])) <= tau:
                n_tracks_near[i] += 1
                if t.state == TrackState.MATURE:
                    best_mat_obs[i] = max(best_mat_obs[i], t.n_obs)

    D = int(np.sum(support == 0))
    A = int(np.sum(recalled))
    lost = ~recalled & (support > 0)
    # localization: a mature track sits near the GT (within tau) but greedy one-to-one
    # matching gave it to another GT -> not a coverage loss
    Lz = int(np.sum(lost & (best_mat_obs > 0)))
    lost_nomat = lost & (best_mat_obs == 0)
    B = int(np.sum(lost_nomat & (support >= GATE)))   # associable: enough obs, no mature track
    C = int(np.sum(lost_nomat & (support < GATE)))    # sparsity: fewer than GATE obs total
    return support, recalled, dict(D_never_detected=D, A_recalled=A,
                                   B_fragmentation=B, C_sparsity=C,
                                   L_localization=Lz)


for tau in (0.5, 1.0):
    support, recalled, dec = decompose(tau)
    n = len(gt)
    det = n - dec["D_never_detected"]
    print(f"\n=== tau={tau} m ===")
    print(f"  detection support (>=1 obs near GT): {det}/{n} = {det/n:.2f}  "
          f"(cf. probe detection recall)")
    print(f"  mature-track recall: {dec['A_recalled']}/{n} = {dec['A_recalled']/n:.2f}")
    print(f"  DECOMPOSITION of the {n} GT:")
    for k, v in dec.items():
        print(f"     {k:18s} {v:3d}  ({v/n:.2f})")
    # of the DETECTED-but-not-recalled: localization (mature track near, matched elsewhere)
    # vs associable fragmentation (>=GATE obs, no mature track) vs sparsity (<GATE obs)
    lost = dec["L_localization"] + dec["B_fragmentation"] + dec["C_sparsity"]
    if lost:
        print(f"  of {lost} detected-but-not-recalled (gate={GATE} obs): "
              f"{dec['L_localization']} localization (mature track near, lost to greedy match), "
              f"{dec['B_fragmentation']} associable frag (>= {GATE} obs, no mature track), "
              f"{dec['C_sparsity']} sparsity (< {GATE} obs)")
    # support distribution among detected GT
    sup_det = support[support > 0]
    if len(sup_det):
        print(f"  obs-per-detected-GT: min={sup_det.min()} med={int(np.median(sup_det))} "
              f"max={sup_det.max()}  #GT with >={GATE} obs: {int(np.sum(sup_det>=GATE))}")

# --- disposition of the mature tracks themselves (over-seg / FP diagnosis) ---
def mature_disposition(tau):
    mt = [t for t in mature]
    mt_xy = np.array([t.centroid[:2] for t in mt]) if mt else np.zeros((0, 2))
    # nearest GT per mature track
    near_gt = [float(np.min(np.linalg.norm(gt_c[:, :2] - xy, axis=1))) for xy in mt_xy]
    fp = sum(1 for d in near_gt if d > tau)                    # no GT within tau
    # duplicates: >1 mature track sharing the same nearest GT within tau
    from collections import Counter as _C
    assign = [int(np.argmin(np.linalg.norm(gt_c[:, :2] - xy, axis=1)))
              for xy in mt_xy if np.min(np.linalg.norm(gt_c[:, :2] - xy, axis=1)) <= tau]
    per_gt = _C(assign)
    dup = sum(c - 1 for c in per_gt.values() if c > 1)         # excess tracks on shared GT
    uniq = sum(1 for c in per_gt.values())                     # distinct GT covered
    print(f"  mature-track disposition @tau={tau}: {len(mt)} mature -> {uniq} distinct GT "
          f"covered, {dup} duplicate (share a GT), {fp} false-positive (no GT within {tau}m)")

for tau in (0.5, 1.0):
    mature_disposition(tau)


# --- what separates FP mature tracks from TP? (to design a precision gate) ---
def feats(t):
    scores = [o.det_score for o in t.observations]
    lv = t.label_votes
    purity = max(lv.values()) / sum(lv.values()) if lv else 0.0
    ncloud = len(t.fused_cloud)
    if ncloud:
        ext = float(np.max(np.ptp(t.fused_cloud[:, :2], axis=0)))  # max XY span (m)
    else:
        ext = 0.0
    dirs = np.array([o.view_dir_world for o in t.observations])
    if len(dirs) > 1:
        md = dirs.mean(0); md /= (np.linalg.norm(md) + 1e-9)
        viewspread = float(1.0 - np.mean(dirs @ md))   # 0=all same view, higher=diverse
    else:
        viewspread = 0.0
    span = float(t.observations[-1].stamp - t.observations[0].stamp)
    return dict(n_obs=t.n_obs, mean_score=float(np.mean(scores)),
                min_score=float(np.min(scores)), purity=purity, n_cloud=ncloud,
                xy_extent=ext, viewspread=viewspread, span_s=span, label=t.label())

mt_xy = np.array([t.centroid[:2] for t in mature])
is_fp = [float(np.min(np.linalg.norm(gt_c[:, :2] - xy, axis=1))) > 1.0 for xy in mt_xy]
tp_f = [feats(t) for t, f in zip(mature, is_fp) if not f]
fp_f = [feats(t) for t, f in zip(mature, is_fp) if f]
print(f"\n=== FP-vs-TP mature-track features (TP={len(tp_f)} FP={len(fp_f)}, FP=no GT within 1m) ===")
keys = ["n_obs", "mean_score", "min_score", "purity", "n_cloud", "xy_extent", "viewspread", "span_s"]
print(f"  {'feature':11s} {'TP med':>10s} {'FP med':>10s} {'TP mean':>10s} {'FP mean':>10s}")
for k in keys:
    tv = np.array([f[k] for f in tp_f]); fv = np.array([f[k] for f in fp_f])
    print(f"  {k:11s} {np.median(tv):>10.3f} {np.median(fv):>10.3f} {tv.mean():>10.3f} {fv.mean():>10.3f}")
print("  FP labels:", Counter(f["label"] for f in fp_f))
print("  TP labels:", Counter(f["label"] for f in tp_f))

print("\nDONE")
