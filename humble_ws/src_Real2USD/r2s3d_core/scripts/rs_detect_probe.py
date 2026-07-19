import numpy as np, cv2, os, logging
logging.basicConfig(level=logging.WARNING)
from r2s3d_core.data.registry import make_source
from r2s3d_core.detect.yoloe import run_detector
from r2s3d_core.tracks.fusion import mask_centroid_world
from r2s3d_core import frames as F

STRIDE = 2
OUT = "results/rosbag_debug"
os.makedirs(OUT, exist_ok=True)
src = make_source("realsense", "hallway-1", stride=STRIDE)
frames = list(src)                       # single bag stream
gt = src.gt()
gt_c = np.array([g.T_world_obj[:3, 3] for g in gt])
gt_vocab = sorted({g.label for g in gt if g.label})
from collections import Counter
print("frames=%d (stride %d)  gt=%d %s  vocab=%s" % (len(frames), STRIDE, len(gt), dict(Counter(g.label for g in gt)), gt_vocab))

def mv_recall(ds, tau):
    """multi-view detection recall: fraction of GT hit by >=1 detection centroid within tau (XY)."""
    bf = ds.by_frame()
    hit = np.zeros(len(gt), bool)
    for fr in frames:
        for det in bf.get(int(fr.frame_id), []):
            c = mask_centroid_world(fr, det.mask)
            if c is None:
                continue
            d = np.linalg.norm(gt_c[:, :2] - c[:2], axis=1)
            hit[d < tau] = True
    return hit.sum()

print("\n=== detector prompt-mode probe (multi-view DETECTION recall = ceiling before tracking) ===")
for prompt, vocab in [("gt", gt_vocab), ("generic", None), ("pf", None)]:
    ds = run_detector(frames, prompt=prompt, vocab=vocab)
    labs = Counter(d.label for d in ds.detections)
    r05, r10 = mv_recall(ds, 0.5), mv_recall(ds, 1.0)
    print("[%-7s] %d det, %d labels, top=%s" % (prompt, len(ds.detections), len(labs), labs.most_common(5)))
    print("           det-recall vs 57 GT: @0.5m %d/%d=%.2f  @1.0m %d/%d=%.2f" %
          (r05, len(gt), r05/len(gt), r10, len(gt), r10/len(gt)))

# ---- GT-box overlays (FIXED pose): pick frames with the most GT boxes in view ----
def corners(g):
    e = g.extents/2; s = np.array([[i,j,k] for i in(-1,1) for j in(-1,1) for k in(-1,1)])*e
    return (g.T_world_obj[:3,:3]@s.T).T + g.T_world_obj[:3,3]
EDGES=[(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
scored=[]
for k,fr in enumerate(frames):
    Tcw=F.invert(fr.T_world_cam); iv=[]
    for g in gt:
        c3=(Tcw[:3,:3]@corners(g).T).T+Tcw[:3,3]
        if (c3[:,2]<=0.2).any(): continue
        uvp=(fr.K@c3.T); uvp=(uvp[:2]/uvp[2]).T; cx,cy=uvp.mean(0)
        if 0<cx<640 and 0<cy<480: iv.append((g,uvp))
    if iv: scored.append((len(iv),k,iv))
scored.sort(reverse=True)
print("\n=== GT-box overlays (top frames by #boxes-in-view) ===")
for rank,(nb,k,iv) in enumerate(scored[:4]):
    fr=frames[k]; vis=cv2.cvtColor(fr.rgb,cv2.COLOR_RGB2BGR).copy()
    d=fr.depth; m=d>0
    if m.any():
        dn=np.clip((d-np.percentile(d[m],2))/max(np.percentile(d[m],98)-np.percentile(d[m],2),1e-6),0,1)
        dc=cv2.applyColorMap((dn*255).astype(np.uint8),cv2.COLORMAP_JET); dc[~m]=0; vis[m]=cv2.addWeighted(vis[m],0.7,dc[m],0.3,0)
    for g,uv in iv:
        col=(0,255,0) if g.label in("chair","door","table") else (0,180,255)
        for a,b in EDGES: cv2.line(vis,tuple(uv[a].astype(int)),tuple(uv[b].astype(int)),col,2)
        cv2.putText(vis,g.label,tuple(uv[0].astype(int)),cv2.FONT_HERSHEY_SIMPLEX,0.5,col,2)
    p=f"{OUT}/RS_hallway_gtbox_fixed_{rank}_f{fr.frame_id}.png"; cv2.imwrite(p,vis)
    print("  frame %d: %d GT boxes -> %s"%(fr.frame_id,nb,p))
print("DONE")
