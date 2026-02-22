#!/usr/bin/env python3
"""
SAM3D worker: reads job directories from queue_dir/input, runs SAM3D inference,
writes results to queue_dir/output/<job_id>/.

Run on the host in the sam3d-objects conda env (see config/USER_NEXT_STEPS.md):
  conda activate sam3d-objects
  python run_sam3d_worker.py --queue-dir /data/sam3d_queue --sam3d-repo /path/to/sam-3d-objects [--once] [--dry-run] [--use-depth]

--sam3d-repo: path to the sam-3d-objects repo clone (required for real inference; has notebook/inference.py and checkpoints/).
--once: process one job and exit (for testing).
--dry-run: only validate job and write dummy pose.json (no SAM3D dependency).
--use-depth: pass depth to SAM3D inference when available (ablation: omit for no pointmap/depth; data is always saved).
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Allow importing ply_frame_utils when run from any cwd (e.g. share/real2sam3d/scripts_sam3d_worker/)
_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

RUN_CONFIG_JSON = "run_config.json"


def _save_worker_args_to_run_config(queue_dir: Path, args: argparse.Namespace) -> None:
    """Merge worker arguments into run_config.json in the run directory (same file launch writes)."""
    config_path = queue_dir / RUN_CONFIG_JSON
    config = {}
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    sam3d_repo_val = getattr(args, "sam3d_repo", None)
    worker_args = {
        "use_depth": getattr(args, "use_depth", False),
        "queue_dir": str(queue_dir.resolve()),
        "no_current_run": getattr(args, "no_current_run", False),
        "sam3d_repo": str(Path(sam3d_repo_val).resolve()) if sam3d_repo_val else None,
        "once": getattr(args, "once", False),
        "dry_run": getattr(args, "dry_run", False),
        "use_init_odom": getattr(args, "use_init_odom", False),
        "write_demo_go2_compare": getattr(args, "write_demo_go2_compare", False),
    }
    config["worker"] = worker_args
    config["worker_updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
    except OSError as e:
        print(f"[WARN] Could not write {config_path}: {e}", file=sys.stderr)


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


def depth_to_pointmap(depth_image: np.ndarray, cam_info: dict, crop_bbox=None):
    """
    Convert depth image (H,W) to pointmap (H,W,3) in PyTorch3D convention using camera intrinsics.
    Matches demo_go2.py: https://github.com/christopher-hsu/sam-3d-objects/blob/main/demo_go2.py
    depth_image: in meters, may be cropped (then crop_bbox [x_min, y_min, x_max, y_max] is required).
    cam_info: dict with "K" or "k" (9 values, row-major 3x3).
    """
    K = np.array(cam_info.get("K") or cam_info.get("k"), dtype=np.float64).reshape(3, 3)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    height, width = depth_image.shape
    u = np.arange(width, dtype=np.float64)
    v = np.arange(height, dtype=np.float64)
    uu, vv = np.meshgrid(u, v)
    if crop_bbox is not None and len(crop_bbox) >= 4:
        x_min, y_min = int(crop_bbox[0]), int(crop_bbox[1])
        uu = uu + x_min
        vv = vv + y_min
    Z = np.asarray(depth_image, dtype=np.float64)
    Z[Z <= 0] = np.nan
    X = (uu - cx) * Z / fx
    Y = (vv - cy) * Z / fy
    # PyTorch3D: +x right, +y UP, +z forward => stack(-X, -Y, Z) from image coords
    pointmap = np.stack((-X, -Y, Z), axis=-1)
    return pointmap


def run_sam3d_inference(
    rgb,
    mask,
    meta,
    depth=None,
    use_depth: bool = False,
    sam3d_repo: Path = None,
    config_path: str = None,
):
    """
    Call SAM3D-Objects inference. Requires sam3d-objects env and repo path (--sam3d-repo).
    Returns dict with 'gs' (gaussian splat) or 'mesh' etc. for export.

    When use_depth is True and depth is not None, depth is passed as pointmap= to the inference call.
    Data is always saved in the job/output; this flag only controls whether SAM3D uses it.
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
        if use_depth and depth is not None:
            # pointmap must be (H,W,3) XYZ in PyTorch3D convention (see demo_go2 depth_to_pointmap)
            cam_info = meta.get("camera_info") or {}
            k_flat = cam_info.get("K") or cam_info.get("k")
            if k_flat is not None and len(k_flat) == 9:
                crop_bbox = meta.get("crop_bbox")
                pointmap_np = depth_to_pointmap(depth, cam_info, crop_bbox=crop_bbox)
                pointmap = torch.from_numpy(pointmap_np.astype(np.float32)).float()
                output = inference(image, mask_loaded, seed=42, pointmap=pointmap)
            else:
                print(
                    "[WARN] --use-depth set but meta has no camera_info.K (9 values); running without pointmap",
                    file=sys.stderr,
                )
                output = inference(image, mask_loaded, seed=42)
        else:
            output = inference(image, mask_loaded, seed=42)
    return output


def process_one_job(
    job_path: Path,
    output_dir: Path,
    dry_run: bool,
    sam3d_repo: Path = None,
    queue_dir: Path = None,
    use_init_odom: bool = False,
    write_demo_go2_compare: bool = False,
    use_depth: bool = False,
) -> bool:
    """Process a single job directory. Returns True on success."""
    job_id = job_path.name
    out_path = output_dir / job_id
    out_path.mkdir(parents=True, exist_ok=True)

    try:
        rgb, mask, depth, meta = load_job(job_path)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return False

    # Pose: persist raw odom from the job. Injector applies init_odom normalization.
    odom = meta["odometry"]
    if use_init_odom:
        print(
            "[WARN] --use-init-odom is deprecated in worker; init_odom is applied by injector.",
            file=sys.stderr,
        )

    if dry_run:
        pose = {
            "frame": "sam3d_raw",
            "job_id": job_id,
            "track_id": meta.get("track_id"),
            "label": meta.get("label"),
            "go2_odom_position": list(odom["position"]),
            "go2_odom_orientation": list(odom["orientation"]),
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
        output = run_sam3d_inference(
            rgb, mask, meta,
            depth=depth,
            use_depth=use_depth,
            sam3d_repo=sam3d_repo,
        )
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

    # Export GLB: vanilla (raw SAM3D mesh, no frame conversion). Transform is done per-slot in sam3d_injector_node.
    if "glb" in output:
        try:
            glb_mesh = output["glb"]
            if hasattr(glb_mesh, "vertices") and hasattr(glb_mesh, "export"):
                verts = np.asarray(glb_mesh.vertices, dtype=np.float64)
                glb_mesh.vertices = verts
                if hasattr(glb_mesh, "volume") and glb_mesh.volume < 0:
                    glb_mesh.invert()
                glb_mesh.export(str(object_glb_path))
                print(f"[OK] Wrote object.glb (vanilla SAM3D)", file=sys.stderr)
            else:
                print(f"[WARN] output['glb'] is not a trimesh (no vertices/export); skipping GLB.", file=sys.stderr)
        except Exception as e:
            print(f"[WARN] Could not export object.glb: {e}", file=sys.stderr)

    # pose.json: raw data for injector to transform. frame=sam3d_raw; injector writes z_up_odom + initial_* after transform.
    pose = {
        "frame": "sam3d_raw",
        "job_id": job_id,
        "track_id": meta.get("track_id"),
        "label": meta.get("label"),
        "go2_odom_position": list(odom["position"]),
        "go2_odom_orientation": list(odom["orientation"]),
    }
    # Save SAM3D transform inputs so injector can run our frame conversion (demo_go2-style) per slot.
    scale = output.get("scale")
    rot = output.get("rotation")
    trans = output.get("translation")
    if scale is not None and rot is not None and trans is not None:
        try:
            if getattr(scale, "shape", None) and scale.shape[0] == 1:
                scale = scale[0]
        except (IndexError, TypeError):
            pass
        try:
            if getattr(trans, "shape", None) and trans.shape[0] == 1:
                trans = trans[0]
        except (IndexError, TypeError):
            pass
        try:
            rot = getattr(rot, "squeeze", lambda: rot)()
        except Exception:
            pass
        if hasattr(scale, "cpu"):
            scale = np.asarray(scale.cpu().numpy(), dtype=np.float64).ravel()
        else:
            scale = np.asarray(scale, dtype=np.float64).ravel()
        if hasattr(rot, "cpu"):
            rot = np.asarray(rot.cpu().numpy(), dtype=np.float64).ravel()
        else:
            rot = np.asarray(rot, dtype=np.float64).ravel()
        if hasattr(trans, "cpu"):
            trans = np.asarray(trans.cpu().numpy(), dtype=np.float64).ravel()
        else:
            trans = np.asarray(trans, dtype=np.float64).ravel()
        pose["sam3d_scale"] = scale.tolist() if hasattr(scale, "tolist") else list(scale)
        pose["sam3d_rotation"] = rot.tolist() if hasattr(rot, "tolist") else list(rot)  # w,x,y,z
        pose["sam3d_translation"] = trans.tolist() if hasattr(trans, "tolist") else list(trans)[:3]

        # Optional worker-side demo_go2 parity export for debugging:
        # Write per-job GLBs transformed with the same row-vector chain as demo_go2.py
        # so we can compare against injector/object_odom.glb without running demo_go2 separately.
        if write_demo_go2_compare and "glb" in output and hasattr(output["glb"], "vertices"):
            try:
                from ply_frame_utils import (
                    GO2_R_WORLD_TO_CAM,
                    GO2_T_WORLD_TO_CAMERA,
                    R_FLIP_Z,
                    R_PYTORCH3D_TO_CAM,
                    R_YUP_TO_ZUP,
                    demo_go2_build_T_raw_to_odom,
                    demo_go2_transform_vertices_to_odom,
                )
                init_odom = None
                if queue_dir is not None:
                    init_path = queue_dir / "init_odom.json"
                    if init_path.exists():
                        with open(init_path) as f:
                            init_odom = json.load(f)

                # IMPORTANT: reload from disk object.glb and flatten node transforms so the
                # compare path uses the same effective raw geometry as injector.
                import trimesh
                loaded = trimesh.load(str(object_glb_path), process=False)
                raw_mesh = None
                if isinstance(loaded, trimesh.Scene):
                    # merge flattened geometries to one mesh for deterministic compare
                    flat = []
                    for node_name in loaded.graph.nodes_geometry:
                        try:
                            node_tf, geom_name = loaded.graph[node_name]
                            geom = loaded.geometry.get(geom_name)
                            if isinstance(geom, trimesh.Trimesh):
                                g = geom.copy()
                                g.apply_transform(np.asarray(node_tf, dtype=np.float64))
                                flat.append(g)
                        except Exception:
                            continue
                    if flat:
                        raw_mesh = trimesh.util.concatenate(flat)
                elif isinstance(loaded, trimesh.Trimesh):
                    raw_mesh = loaded
                if raw_mesh is None:
                    raise RuntimeError("Could not load object.glb geometry for demo-go2 compare")

                raw_verts = np.asarray(raw_mesh.vertices, dtype=np.float64)
                verts_cam, verts_odom = demo_go2_transform_vertices_to_odom(
                    raw_verts,
                    scale=scale,
                    rotation_quat_wxyz=rot,
                    translation=trans,
                    odom=odom,
                    init_odom=init_odom,
                )

                mesh_cam = raw_mesh.copy()
                mesh_cam.vertices = verts_cam
                if hasattr(mesh_cam, "volume") and mesh_cam.volume < 0:
                    mesh_cam.invert()
                cam_out = out_path / "object_demo_go2_cam.glb"
                mesh_cam.export(str(cam_out))

                mesh_odom = raw_mesh.copy()
                mesh_odom.vertices = verts_odom
                if hasattr(mesh_odom, "volume") and mesh_odom.volume < 0:
                    mesh_odom.invert()
                odom_out = out_path / "object_demo_go2_odom.glb"
                mesh_odom.export(str(odom_out))

                # Build apples-to-apples numeric debug payload with the same structure
                # as injector transform_debug.json so we can diff the two directly.
                T_raw_to_odom = demo_go2_build_T_raw_to_odom(
                    scale=scale,
                    rotation_quat_wxyz=rot,
                    translation=trans,
                    odom=odom,
                    init_odom=init_odom,
                )
                basis = np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                )
                basis_cam, basis_odom_direct = demo_go2_transform_vertices_to_odom(
                    basis,
                    scale=scale,
                    rotation_quat_wxyz=rot,
                    translation=trans,
                    odom=odom,
                    init_odom=init_odom,
                )
                ones = np.ones((basis.shape[0], 1), dtype=np.float64)
                basis_h = np.hstack([basis, ones])
                basis_odom_matrix = (T_raw_to_odom @ basis_h.T).T[:, :3]

                compare = {
                    "centroid": np.asarray(mesh_odom.centroid).tolist(),
                    "aabb_bounds": np.asarray(mesh_odom.bounds).tolist(),
                    "init_odom_position": init_odom.get("position") if isinstance(init_odom, dict) else None,
                    "constants": {
                        "R_flip_z": np.asarray(R_FLIP_Z, dtype=np.float64).tolist(),
                        "R_yup_to_zup": np.asarray(R_YUP_TO_ZUP, dtype=np.float64).tolist(),
                        "R_pytorch3d_to_cam": np.asarray(R_PYTORCH3D_TO_CAM, dtype=np.float64).tolist(),
                        "R_world_to_cam": np.asarray(GO2_R_WORLD_TO_CAM, dtype=np.float64).tolist(),
                        "t_world_to_camera": np.asarray(GO2_T_WORLD_TO_CAMERA, dtype=np.float64).tolist(),
                    },
                    "inputs": {
                        "odom_position": list(odom["position"]),
                        "odom_orientation": list(odom["orientation"]),
                        "init_odom_position": init_odom.get("position") if isinstance(init_odom, dict) else None,
                        "sam3d_scale": np.asarray(scale, dtype=np.float64).tolist(),
                        "sam3d_rotation_wxyz": np.asarray(rot, dtype=np.float64).tolist(),
                        "sam3d_translation": np.asarray(trans, dtype=np.float64).tolist(),
                    },
                    "matrices": {
                        "T_raw_to_odom": np.asarray(T_raw_to_odom, dtype=np.float64).tolist(),
                    },
                    "basis_test": {
                        "raw_basis_points": basis.tolist(),
                        "cam_direct_chain": np.asarray(basis_cam, dtype=np.float64).tolist(),
                        "odom_direct_chain": np.asarray(basis_odom_direct, dtype=np.float64).tolist(),
                        "odom_matrix_chain": np.asarray(basis_odom_matrix, dtype=np.float64).tolist(),
                        "odom_direct_minus_matrix": np.asarray(basis_odom_direct - basis_odom_matrix, dtype=np.float64).tolist(),
                    },
                }
                with open(out_path / "demo_go2_compare.json", "w") as f:
                    json.dump(compare, f, indent=2)
                print(f"[OK] Wrote demo-go2 compare GLBs: {cam_out.name}, {odom_out.name}", file=sys.stderr)
            except Exception as e:
                print(f"[WARN] Could not write demo-go2 compare GLBs: {e}", file=sys.stderr)
    with open(out_path / "pose.json", "w") as f:
        json.dump(pose, f, indent=2)

    # Copy input crop and registration data to output so result dir is self-contained
    # (for FAISS indexer / multi-view and for segment-PC-based registration)
    for name in ("rgb.png", "depth.npy", "mask.png", "meta.json"):
        try:
            src = job_path / name
            if src.exists():
                shutil.copy2(str(src), str(out_path / name))
        except OSError as e:
            print(f"[WARN] Could not copy {name} to output: {e}", file=sys.stderr)

    outputs = "pose.json + object.ply"
    if object_glb_path.exists():
        outputs += " + object.glb"
    print(f"[OK] Wrote {out_path} ({outputs}, origin)", file=sys.stderr)
    return True


def main():
    parser = argparse.ArgumentParser(description="SAM3D worker: process jobs from queue_dir/input")
    parser.add_argument("--queue-dir", type=str, default="/data/sam3d_queue", help="Queue directory (used only with --no-current-run)")
    parser.add_argument("--no-current-run", action="store_true", help="Do not use current_run.json; use --queue-dir (and paths) explicitly")
    parser.add_argument("--sam3d-repo", type=str, default=None, help="Path to sam-3d-objects repo (required for real inference; has notebook/ and checkpoints/)")
    parser.add_argument("--once", action="store_true", help="Process one job and exit")
    parser.add_argument("--dry-run", action="store_true", help="Only validate job and write pose (no SAM3D)")
    parser.add_argument("--use-init-odom", action="store_true", help="Deprecated: kept for compatibility; injector now applies init_odom normalization")
    parser.add_argument("--write-demo-go2-compare", action="store_true", help="Also write object_demo_go2_cam.glb/object_demo_go2_odom.glb + demo_go2_compare.json per job for transform parity checks")
    parser.add_argument("--use-depth", action="store_true", dest="use_depth", help="Pass depth to SAM3D inference as pointmap (ablation: omit for no pointmap; data is always saved)")
    args = parser.parse_args()

    use_current_run = not args.no_current_run
    try:
        from current_run import resolve_queue_and_index
        queue_dir, _ = resolve_queue_and_index(use_current_run, args.queue_dir, None)
    except ImportError:
        queue_dir = Path(args.queue_dir).resolve()
        if use_current_run:
            print("Warning: current_run module not found; using --queue-dir. Use --no-current-run to silence.", file=sys.stderr)

    sam3d_repo = Path(args.sam3d_repo).resolve() if args.sam3d_repo else None
    if not args.dry_run and sam3d_repo is None:
        print("Warning: --sam3d-repo not set; real inference will fail. Use --dry-run to test without SAM3D.", file=sys.stderr)
    input_dir = queue_dir / "input"
    output_dir = queue_dir / "output"
    processed_dir = queue_dir / "input_processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    print(f"Queue dir: {queue_dir}", file=sys.stderr)
    print(f"Output dir: {output_dir}", file=sys.stderr)
    _save_worker_args_to_run_config(queue_dir, args)

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
        success = process_one_job(
            job_path, output_dir, args.dry_run, sam3d_repo=sam3d_repo,
            queue_dir=queue_dir, use_init_odom=args.use_init_odom,
            write_demo_go2_compare=args.write_demo_go2_compare,
            use_depth=args.use_depth,
        )
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
