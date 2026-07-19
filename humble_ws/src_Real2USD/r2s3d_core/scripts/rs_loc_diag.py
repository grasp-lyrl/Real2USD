import numpy as np, logging
logging.basicConfig(level=logging.WARNING)
from r2s3d_core.data.registry import make_source
from r2s3d_core.detect.cache import DetectionSet
from r2s3d_core.tracks.fusion import mask_centroid_world
from r2s3d_core.tracks import TrackState, run_tracker

src = make_source("realsense", "hallway-1", stride=5)
frames = list(src)
gt = src.gt()
gt_c = np.array([g.T_world_obj[:3, 3] for g in gt])
print("GT   XYZ bbox: X[%.1f,%.1f] Y[%.1f,%.1f] Z[%.1f,%.1f]" % (
    gt_c[:,0].min(), gt_c[:,0].max(), gt_c[:,1].min(), gt_c[:,1].max(), gt_c[:,2].min(), gt_c[:,2].max()))
cams = np.array([fr.T_world_cam[:3, 3] for fr in frames])
print("cam  XYZ bbox: X[%.1f,%.1f] Y[%.1f,%.1f] Z[%.1f,%.1f]" % (
    cams[:,0].min(), cams[:,0].max(), cams[:,1].min(), cams[:,1].max(), cams[:,2].min(), cams[:,2].max()))

ds = DetectionSet.load("results/detections/gt/hallway-1", scene="hallway-1")
bf = ds.by_frame()

# back-project full valid depth of a few detection frames -> world bbox (transform sanity)
allpts = []
for fr in frames:
    if not bf.get(int(fr.frame_id)):
        continue
    ys, xs = np.where(fr.depth > 0)
    if len(xs) == 0:
        continue
    z = fr.depth[ys, xs].astype(float)
    fx, fy, cx, cy = fr.K[0,0], fr.K[1,1], fr.K[0,2], fr.K[1,2]
    pc = np.stack([(xs-cx)*z/fx, (ys-cy)*z/fy, z], 1)[::50]
    allpts.append((fr.T_world_cam[:3,:3] @ pc.T).T + fr.T_world_cam[:3,3])
    if len(allpts) >= 30:
        break
allpts = np.vstack(allpts)
print("bp   XYZ bbox: X[%.1f,%.1f] Y[%.1f,%.1f] Z[%.1f,%.1f]  (should ~overlap GT/room)" % (
    allpts[:,0].min(), allpts[:,0].max(), allpts[:,1].min(), allpts[:,1].max(), allpts[:,2].min(), allpts[:,2].max()))

# per-detection masked-depth centroid -> nearest GT (XY)
dists = []
for fr in frames:
    for det in bf.get(int(fr.frame_id), []):
        c = mask_centroid_world(fr, det.mask)
        if c is None:
            continue
        dists.append(float(np.linalg.norm(gt_c[:,:2] - c[:2], axis=1).min()))
dists = np.array(dists)
print("per-detection centroid -> nearest GT (XY): n=%d median %.2fm p10 %.2fm frac<0.5m %.2f frac<1m %.2f" % (
    len(dists), np.median(dists), np.percentile(dists,10), (dists<0.5).mean(), (dists<1.0).mean()))

# mature track centroids vs nearest GT
tracks = run_tracker(frames, bf, {"reid": True, "late_merge": True})
mature = [t for t in tracks if t.state == TrackState.MATURE]
print("\nmature track centroids vs nearest GT:")
for t in sorted(mature, key=lambda t: t.track_id):
    c = t.centroid
    d = np.linalg.norm(gt_c[:,:2] - c[:2], axis=1)
    j = int(d.argmin())
    print("  t%-3d %-6s c=(%.1f,%.1f,%.1f) nearest GT[%s] @%.2fm" % (
        t.track_id, t.label(), c[0], c[1], c[2], gt[j].label, d[j]))
print("DONE")
