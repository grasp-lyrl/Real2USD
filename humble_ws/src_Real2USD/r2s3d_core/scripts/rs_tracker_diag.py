import numpy as np, logging
from collections import Counter
logging.basicConfig(level=logging.WARNING)
from r2s3d_core.data.registry import make_source
from r2s3d_core.detect.cache import DetectionSet
from r2s3d_core.tracks import TrackState, run_tracker

src = make_source("realsense", "hallway-1", stride=5)
frames = list(src)
gt = src.gt()
ds = DetectionSet.load("results/detections/gt/hallway-1", scene="hallway-1")
by_frame = ds.by_frame()
gt_c = np.array([g.T_world_obj[:3, 3] for g in gt])
gt_lab = Counter(g.label for g in gt)
print("frames=%d detections=%d gt=%d  gt_labels=%s" % (len(frames), len(ds.detections), len(gt), dict(gt_lab)))

def greedy_recall(mature, tau, labels=None):
    ms = [t for t in mature if labels is None or t.label() in labels]
    tr = np.array([t.centroid[:2] for t in ms]) if ms else np.zeros((0, 2))
    idx = [i for i, g in enumerate(gt) if labels is None or g.label in labels]
    used, hit = set(), 0
    for i in idx:
        gc = gt_c[i, :2]
        best, bd = -1, tau
        for j in range(len(tr)):
            if j in used:
                continue
            d = float(np.linalg.norm(tr[j] - gc))
            if d < bd:
                bd, best = d, j
        if best >= 0:
            used.add(best); hit += 1
    return hit, len(idx)

def frag(mature, tau=0.5):
    tr = np.array([t.centroid[:2] for t in mature]) if mature else np.zeros((0, 2))
    counts = []
    for gc in gt_c[:, :2]:
        counts.append(int(np.sum(np.linalg.norm(tr - gc, axis=1) < tau)) if len(tr) else 0)
    return counts

for name, cfg in [("naive", {"reid": False, "late_merge": False}),
                  ("full",  {"reid": True,  "late_merge": True})]:
    tracks = run_tracker(frames, by_frame, dict(cfg))
    mature = [t for t in tracks if t.state == TrackState.MATURE]
    print("\n[%s] total_tracks=%d mature=%d  tracks/GT=%.2f" % (name, len(tracks), len(mature), len(mature) / len(gt)))
    print("   mature labels:", dict(Counter(t.label() for t in mature)))
    for tau in (0.25, 0.5, 1.0):
        h, n = greedy_recall(mature, tau)
        hc, nc = greedy_recall(mature, tau, labels={"chair", "door", "table"})
        print("   per-track recall @%.2fm: all %d/%d=%.2f | chair/door/table %d/%d=%.2f" %
              (tau, h, n, h / n, hc, nc, hc / nc if nc else float("nan")))
    fr = frag(mature)
    over = sum(1 for c in fr if c > 1)
    print("   fragmentation@0.5m: %d/%d GT have >1 mature track nearby (over-seg)" % (over, len(gt)))
print("\nDONE")
