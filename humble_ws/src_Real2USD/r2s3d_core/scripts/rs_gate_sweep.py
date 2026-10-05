"""Sweep the track precision gate (--track-gate) thresholds on a real Go2 scene.

Runs the tracker ONCE, then for a grid of (gate_min_obs, gate_min_score) applies the gate
to the mature set and reports greedy one-to-one centroid precision / recall / F1 vs the
Supervisely GT (track-set quality, before SAM3D/placement). Pick the knee, then do one full
eval.run --track-gate at that setting for placement numbers.

Run:  uv run python scripts/rs_gate_sweep.py [scene] [stride]
"""
import sys, logging
import numpy as np

logging.basicConfig(level=logging.WARNING)
from r2s3d_core.data.registry import make_source           # noqa: E402
from r2s3d_core.detect.cache import DetectionSet            # noqa: E402
from r2s3d_core.tracks import TrackState, run_tracker        # noqa: E402
from r2s3d_core.tracks.tracker import _passes_precision_gate  # noqa: E402

SCENE = sys.argv[1] if len(sys.argv) > 1 else "hallway-1"
STRIDE = int(sys.argv[2]) if len(sys.argv) > 2 else 2

src = make_source("realsense", SCENE, stride=STRIDE)
frames = list(src)
gt = src.gt()
gt_c = np.array([g.T_world_obj[:3, 3] for g in gt])
ds = DetectionSet.load(f"results/detections/{'gt'}/{SCENE}", scene=SCENE)
tracks = run_tracker(frames, ds.by_frame(), {"reid": True, "late_merge": True})
mature_all = [t for t in tracks if t.state == TrackState.MATURE]
print(f"scene={SCENE} stride={STRIDE} gt={len(gt)} mature(ungated)={len(mature_all)}")


def prf(mature, tau):
    mt = np.array([t.centroid[:2] for t in mature]) if mature else np.zeros((0, 2))
    used, tp = set(), 0
    for i in range(len(gt)):
        best, bd = -1, tau
        for j in range(len(mt)):
            if j in used:
                continue
            d = float(np.linalg.norm(mt[j] - gt_c[i, :2]))
            if d < bd:
                bd, best = d, j
        if best >= 0:
            used.add(best); tp += 1
    n_pred, n_gt = len(mature), len(gt)
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gt if n_gt else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return tp, n_pred, p, r, f


print(f"\n{'min_obs':>7} {'min_score':>9} {'#kept':>6} "
      f"{'P@1':>6} {'R@1':>6} {'F1@1':>6}   {'P@.5':>6} {'R@.5':>6} {'F1@.5':>6}")
for mo in (3, 6, 8, 10, 12, 15):
    for ms in (0.0, 0.40, 0.45, 0.50):
        cfg = {"gate_min_obs": mo, "gate_min_score": ms}
        kept = [t for t in mature_all if _passes_precision_gate(t, cfg)]
        _, _, p1, r1, f1 = prf(kept, 1.0)
        _, _, p5, r5, f5 = prf(kept, 0.5)
        print(f"{mo:>7} {ms:>9.2f} {len(kept):>6} "
              f"{p1:>6.2f} {r1:>6.2f} {f1:>6.2f}   {p5:>6.2f} {r5:>6.2f} {f5:>6.2f}")

print("\nbaseline (ungated) row = min_obs=3 min_score=0.00")
print("DONE")
