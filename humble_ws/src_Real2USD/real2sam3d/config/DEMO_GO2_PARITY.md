# demo_go2 vs sam3d_only Pipeline Parity

This document compares [demo_go2.py](https://github.com/christopher-hsu/sam-3d-objects/blob/main/demo_go2.py) with the **sam3d_only** path in this repo (worker → injector → scene_sam3d_only). The transform chain is intended to match demo_go2 so that `scene_sam3d_only.glb` / `object_odom.glb` are equivalent to running demo_go2 on the same inputs.

## 1. What Each Side Does

### demo_go2.py (reference)

- Loads per-step: `rgb.png`, `depth.png`, `odom.json`, `camera_info.json`, optional `points.ply`/`points.npy`.
- Runs YOLO segmentation → per-mask SAM3D inference with `pointmap` when depth is available.
- **transform_glb**: raw GLB vertices → flip Z → Y-up→Z-up → scale, rotate, translate (PyTorch3D) → `R_pytorch3d_to_cam` → **camera frame**.
- **transform_glb_to_world**: camera → world (`R_world_to_cam.T`, `t_world_to_camera`) → odom (with `t -= [init_odom.t[0], init_odom.t[1], 0]`), then export `glbs_world/<step>/<label>.glb` and combined `scene_world_<step>.glb`.

### sam3d_only pipeline (this repo)

- **Job writer** writes cropped `rgb.png`, `mask.png`, `depth.npy` (meters), `meta.json` (camera_info + odometry as `position` / `orientation`).
- **Worker** runs SAM3D inference (same `Inference` + optional `pointmap`), saves **raw** `object.glb` and `pose.json` (scale, rotation wxyz, translation, go2_odom_*).
- **Injector** applies the same math as demo_go2 (in `real2sam3d/ply_frame_utils.py`): raw → pointmap → camera → world → odom; writes `transform_odom_from_raw`, `initial_position`/`initial_orientation`, and **object_odom.glb**.
- **Scene buffer** builds `scene_sam3d_only.glb` by applying `transform_odom_from_raw` to each raw `object.glb` and merging.

So **object_odom.glb** and **scene_sam3d_only.glb** should match demo_go2’s world-frame GLBs when inputs and conventions align.

## 2. Parity Checklist

| Item | demo_go2 | sam3d_only | Match? |
|------|----------|------------|--------|
| Inference | `Inference(config_path)`, `inference(img, mask, seed=42, pointmap=pointmap)` | Same (worker uses same repo/notebook) | ✓ |
| Pointmap | `depth_to_pointmap`: Z in **mm** → `/1000`; then `(-X, -Y, Z)` | Worker expects depth in **meters**; job writer saves `depth/1000` from 16UC1 | ✓ (pipeline in m) |
| Raw→Camera | flip Z, Y-up→Z-up, scale, rotate, translate, then `R_pytorch3d_to_cam` | `transform_mesh_vertices` + `@ R_pytorch3d_to_cam` in injector | ✓ |
| Camera→Odom | `R_world_to_cam` (3×3), `t_world_to_camera` [0.285, 0, 0.01]; `t_odom -= [init_odom.t[0], init_odom.t[1], 0]` | Same constants and init_odom subtraction in `_demo_go2_cam_to_odom_vertices` | ✓ |
| Odom quat | `R.from_quat(q)` (scipy: x,y,z,w) | Default xyzw; set `REAL2SAM3D_ODOM_QUAT_WXYZ=1` if source is wxyz | ✓ |
| init_odom | First step’s odom; subtract (x,y,0) from all t | `use_init_odom:=true` + first job or `init_odom.json` | ✓ |

## 3. How to Get Similar Results to demo_go2

1. **Use the same conventions**
   - Run with **use_init_odom:=true** so poses are first-frame-relative (like demo_go2).
   - Ensure odom is in the same frame and quaternion convention (xyzw unless your driver uses wxyz; then set `REAL2SAM3D_ODOM_QUAT_WXYZ=1`).

2. **Use depth for SAM3D (pointmap)**
   - Worker: run with **--use-depth** so inference gets `pointmap` (job writer already saves depth in meters).
   - Launch/config: ensure depth and camera_info are in the job (default pipeline does this).

3. **Compare numerically**
   - Worker: `--write-demo-go2-compare` → writes `object_demo_go2_cam.glb`, `object_demo_go2_odom.glb`, and `demo_go2_compare.json` per job.
   - Injector: set `REAL2SAM3D_DEBUG_DUMP=1` → writes `transform_debug.json` per slot.
   - For the same job/slot, compare `demo_go2_compare.json` and `transform_debug.json`: `basis_test.odom_direct_*` and matrices should match within float tolerance.

4. **Outputs to compare**
   - demo_go2: `glbs_world/<step>/<label>.glb` and combined `scene_world_<step>.glb`.
   - sam3d_only: `output/<job_id>/object_odom.glb`, and merged `scene_sam3d_only.glb` in the queue dir.

## 4. Differences That Can Affect Results (Not Transform Bugs)

- **Image/mask size**: demo_go2 resizes to (640, 480); we pass cropped ROI as-is. Different resolution can change SAM3D output.
- **Segmentation**: demo_go2 uses YOLO in-script; we use segmentation from the pipeline (e.g. lidar_cam_node) and write a single mask per job. Different masks → different geometry.
- **Depth source**: If you feed a dataset that stores depth in **mm** (e.g. uint16 PNG), the job must provide depth in **meters** to the worker (e.g. convert when writing jobs), or the worker’s `depth_to_pointmap` would need to scale; our job writer already converts 16UC1 mm → meters.

## 5. Code References

- demo_go2: `transform_glb`, `transform_glb_to_world`, `depth_to_pointmap` (lines ~100–175).
- Our transform chain: `real2sam3d/ply_frame_utils.py` (`transform_mesh_vertices`, `transform_glb_to_world`, `build_T_raw_to_odom`, `_demo_go2_cam_to_odom_vertices`).
- Our injector: `sam3d_injector_node.py` uses these to write `object_odom.glb` and pose fields.
- Worker parity path: `scripts_sam3d_worker/run_sam3d_worker.py` (`--write-demo-go2-compare`) and `scripts_sam3d_worker/ply_frame_utils.py` (`demo_go2_transform_vertices_to_odom`, `demo_go2_build_T_raw_to_odom`).
