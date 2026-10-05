"""Export viewable lite GLBs for the Phase-2 object_track runs.

Re-runs the tracker + placement (SAM3D outputs are input-hash cached, so no inference)
and writes lite pred/gt/compare GLBs into each results/phase2_<name>/ dir. Lite =
textures stripped + decimated to ~6k faces/object (needs `uv sync --extra viz`).

  uv run python scripts/export_phase2_glb.py [scene ...]        # default room0 room1

Open the *_compare_lite.glb (pred = blue, GT = translucent gray) in any glTF viewer
(VS Code glTF Tools, https://gltf-viewer.donmccurdy.com, Blender).
"""
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from r2s3d_core.baselines import get_method
from r2s3d_core.data.registry import make_source
from r2s3d_core.recon.scene_glb import export_pred_vs_gt

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
ROOT = os.path.join(os.path.dirname(__file__), "..")
DET = os.path.join(ROOT, "results", "detections", "gt")

SCENES = sys.argv[1:] or ["room0", "room1"]
# (method, results-subdir suffix); full == object_track (reid+late_merge)
RUNS = [("object_track", ""), ("object_track_naive", "_naive")]


def main():
    for scene in SCENES:
        for method, suffix in RUNS:
            name = f"{scene}_gt{suffix}"
            out_dir = os.path.join(ROOT, "results", f"phase2_{name}")
            if not os.path.isdir(out_dir):
                print(f"skip {name}: {out_dir} missing (run the eval first)")
                continue
            src = make_source("replica", scene, stride=20)
            gt = src.gt()
            cfg = {"detections": DET, "full_frame": True, "compute_geometry": False,
                   "reid": None, "late_merge": None}
            preds = get_method(method)(src, gt, cfg)
            if not preds:
                print(f"{name}: 0 placed objects (SAM3D jobs still pending?) — skipping GLB")
                continue
            paths = export_pred_vs_gt(preds, gt, out_dir, lite=True, full=False)
            print(f"{name}: {len(preds)} pred / {len(gt)} gt -> "
                  f"{os.path.relpath(paths['compare_lite'], ROOT)}")
    print("\nopen *_compare_lite.glb in a glTF viewer (pred=blue, GT=translucent gray)")


if __name__ == "__main__":
    main()
