#!/usr/bin/env python3
"""
SAM3D worker: reads job directories from queue_dir/input, runs SAM3D inference,
writes results to queue_dir/output/<job_id>/.

Run on the host in the sam3d-objects conda env (see config/USER_NEXT_STEPS.md):
  conda activate sam3d-objects
  python run_sam3d_worker.py --queue-dir /data/sam3d_queue --sam3d-repo /path/to/sam-3d-objects [--once] [--dry-run]

--sam3d-repo: path to the sam-3d-objects repo clone (required for real inference; has notebook/inference.py and checkpoints/).
--once: process one job and exit (for testing).
--dry-run: only validate job and write dummy pose.json (no SAM3D dependency).
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

# Allow importing ply_frame_utils when run from any cwd (e.g. share/real2sam3d/scripts_sam3d_worker/)
_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))


def _move_to_processed(job_path: Path, processed_dir: Path) -> None:
    """Move a completed job from input/ to input_processed/ so we don't reprocess it."""
    dest = processed_dir / job_path.name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.move(str(job_path), str(dest))


def load_job(job_path: Path):
    """Load rgb, mask, depth, meta from a job directory."""
    rgb_path = job_path / "rgb.png"
    mask_path = job_path / "mask.png"
    depth_path = job_path / "depth.npy"
    meta_path = job_path / "meta.json"
    for p, name in [(rgb_path, "rgb.png"), (mask_path, "mask.png"), (depth_path, "depth.npy"), (meta_path, "meta.json")]:
        if not p.exists():
            raise FileNotFoundError(f"Missing {name} in {job_path}")
    import numpy as np
    import cv2
    rgb = cv2.imread(str(rgb_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    depth = np.load(str(depth_path))
    with open(meta_path) as f:
        meta = json.load(f)
    return rgb, mask, depth, meta


def run_sam3d_inference(rgb, mask, meta, sam3d_repo: Path = None, config_path: str = None):
    """
    Call SAM3D-Objects inference. Requires sam3d-objects env and repo path (--sam3d-repo).
    Returns dict with 'gs' (gaussian splat) or 'mesh' etc. for export.
    """
    repo = sam3d_repo or (Path(__file__).resolve().parents[2] / "sam-3d-objects")
    repo = Path(repo).resolve()
    notebook_dir = repo / "notebook"
    if not notebook_dir.is_dir():
        raise RuntimeError(
            f"SAM3D repo path must contain notebook/ (inference module). Got: {repo}"
        )
    # So "from inference import ..." finds notebook/inference.py
    if str(notebook_dir) not in sys.path:
        sys.path.insert(0, str(notebook_dir))
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    try:
        from inference import Inference, load_image, load_single_mask
    except ImportError as e:
        raise RuntimeError(
            "SAM3D inference not available. Run from sam3d-objects env with --sam3d-repo /path/to/sam-3d-objects. "
            f"Import error: {e}"
        ) from e

    tag = "hf"
    config_path = config_path or str(repo / "checkpoints" / tag / "pipeline.yaml")
    inference = Inference(config_path, compile=False)

    # SAM3D expects image (RGBA or RGB) and mask; demo uses load_image (from file) and load_single_mask (dir + index).
    # We have numpy rgb and mask; save to temp and load, or pass in-memory if API allows.
    import tempfile
    import cv2
    with tempfile.TemporaryDirectory() as tmp:
        img_path = Path(tmp) / "image.png"
        cv2.imwrite(str(img_path), rgb)
        mask_dir = Path(tmp) / "masks"
        mask_dir.mkdir()
        # SAM3D load_single_mask(dir, index=0) expects file named "0.png" in dir
        mask_path = mask_dir / "0.png"
        cv2.imwrite(str(mask_path), mask)
        image = load_image(str(img_path))
        mask_loaded = load_single_mask(str(mask_dir), index=0)
        output = inference(image, mask_loaded, seed=42)
    return output


def process_one_job(job_path: Path, output_dir: Path, dry_run: bool, sam3d_repo: Path = None) -> bool:
    """Process a single job directory. Returns True on success."""
    job_id = job_path.name
    out_path = output_dir / job_id
    out_path.mkdir(parents=True, exist_ok=True)

    try:
        rgb, mask, depth, meta = load_job(job_path)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return False

    # Pose: keep robot/camera pose when image was retrieved (from odom); useful for later (e.g. registration).
    odom = meta["odometry"]
    position = list(odom["position"])
    orientation = list(odom["orientation"])  # qx, qy, qz, qw

    if dry_run:
        pose = {
            "position": position,
            "orientation": orientation,
            "job_id": job_id,
            "track_id": meta.get("track_id"),
            "label": meta.get("label"),
        }
        with open(out_path / "pose.json", "w") as f:
            json.dump(pose, f, indent=2)
        (out_path / "object.ply").write_text("# dry-run placeholder\n")
        try:
            if (job_path / "rgb.png").exists():
                shutil.copy2(str(job_path / "rgb.png"), str(out_path / "rgb.png"))
        except OSError:
            pass
        print(f"[dry-run] Wrote pose.json and placeholder object.ply for {job_id}")
        return True

    try:
        output = run_sam3d_inference(rgb, mask, meta, sam3d_repo=sam3d_repo)
    except Exception as e:
        print(f"[ERROR] SAM3D inference failed for {job_id}: {e}", file=sys.stderr)
        return False

    # Export: PLY and GLB centered at origin only (no odom transform). Registration places object later.
    object_ply_path = out_path / "object.ply"
    object_glb_path = out_path / "object.glb"
    if "gs" in output:
        output["gs"].save_ply(str(object_ply_path))
        try:
            from ply_frame_utils import center_ply_to_origin
            center_ply_to_origin(object_ply_path, out_path=object_ply_path, z_up=True)
        except Exception as e:
            print(f"[WARN] Could not center PLY to origin: {e}", file=sys.stderr)
    else:
        object_ply_path.write_text("# no gs in output (keys: {})\n".format(list(output.keys()) if isinstance(output, dict) else type(output).__name__))
        print(f"[WARN] SAM3D output had no 'gs'; wrote placeholder object.ply for {job_id}. Keys: {list(output.keys()) if isinstance(output, dict) else 'N/A'}", file=sys.stderr)

    # Export GLB centered at origin, Z-up world frame (SAM3D/glTF use Y-up).
    if "glb" in output:
        try:
            import numpy as np
            from ply_frame_utils import vertices_y_up_to_z_up
            glb_mesh = output["glb"]
            if hasattr(glb_mesh, "vertices") and hasattr(glb_mesh, "export"):
                verts = np.asarray(glb_mesh.vertices, dtype=np.float64)
                centroid = np.mean(verts, axis=0)
                verts_local = verts - centroid
                verts_local = vertices_y_up_to_z_up(verts_local)
                glb_mesh.vertices = verts_local
                glb_mesh.export(str(object_glb_path))
                print(f"[OK] Wrote object.glb (origin, Z-up)", file=sys.stderr)
            else:
                print(f"[WARN] output['glb'] is not a trimesh (no vertices/export); skipping GLB.", file=sys.stderr)
        except Exception as e:
            print(f"[WARN] Could not export object.glb: {e}", file=sys.stderr)

    pose = {
        "position": position,
        "orientation": orientation,
        "job_id": job_id,
        "track_id": meta.get("track_id"),
        "label": meta.get("label"),
    }
    with open(out_path / "pose.json", "w") as f:
        json.dump(pose, f, indent=2)

    # Copy input crop to output so result dir is self-contained (for FAISS indexer / multi-view)
    try:
        src_rgb = job_path / "rgb.png"
        if src_rgb.exists():
            shutil.copy2(str(src_rgb), str(out_path / "rgb.png"))
    except OSError as e:
        print(f"[WARN] Could not copy rgb.png to output: {e}", file=sys.stderr)

    outputs = "pose.json + object.ply"
    if object_glb_path.exists():
        outputs += " + object.glb"
    print(f"[OK] Wrote {out_path} ({outputs}, origin)", file=sys.stderr)
    return True


def main():
    parser = argparse.ArgumentParser(description="SAM3D worker: process jobs from queue_dir/input")
    parser.add_argument("--queue-dir", type=str, default="/data/sam3d_queue", help="Base queue directory")
    parser.add_argument("--sam3d-repo", type=str, default=None, help="Path to sam-3d-objects repo (required for real inference; has notebook/ and checkpoints/)")
    parser.add_argument("--once", action="store_true", help="Process one job and exit")
    parser.add_argument("--dry-run", action="store_true", help="Only validate job and write pose (no SAM3D)")
    args = parser.parse_args()

    sam3d_repo = Path(args.sam3d_repo).resolve() if args.sam3d_repo else None
    if not args.dry_run and sam3d_repo is None:
        print("Warning: --sam3d-repo not set; real inference will fail. Use --dry-run to test without SAM3D.", file=sys.stderr)

    queue_dir = Path(args.queue_dir).resolve()
    input_dir = queue_dir / "input"
    output_dir = queue_dir / "output"
    processed_dir = queue_dir / "input_processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    print(f"Queue dir: {queue_dir}", file=sys.stderr)
    print(f"Output dir: {output_dir}", file=sys.stderr)

    if not input_dir.exists():
        print(f"Input dir does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)

    while True:
        jobs = [p for p in input_dir.iterdir() if p.is_dir()]
        # Process oldest first (FIFO by directory mtime)
        jobs.sort(key=lambda p: p.stat().st_mtime)
        if not jobs:
            if args.once:
                print("No jobs in queue, exiting.")
                sys.exit(0)
            time.sleep(1.0)
            continue

        job_path = jobs[0]
        print(f"Processing job: {job_path.name} ...", file=sys.stderr)
        success = process_one_job(job_path, output_dir, args.dry_run, sam3d_repo=sam3d_repo)
        if success and args.once:
            _move_to_processed(job_path, processed_dir)
            sys.exit(0)
        if success:
            _move_to_processed(job_path, processed_dir)
        if args.once:
            sys.exit(0 if success else 1)
        time.sleep(0.5)


if __name__ == "__main__":
    main()
