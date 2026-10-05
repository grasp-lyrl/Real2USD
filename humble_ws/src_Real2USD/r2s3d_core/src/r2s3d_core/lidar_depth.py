"""Project an accumulated world point cloud into a camera to synthesize depth.

The Go2 bags carry only RGB (``/camera/image_raw``) + a sparse LiDAR cloud
(``/point_cloud2``, already in the odom/world frame) — there is no depth camera.
The :class:`~r2s3d_core.data.rosbag.RosbagSource` accumulates the LiDAR sweeps into
one world cloud and calls :func:`project_cloud_to_depth` per RGB frame to produce a
registered depth image, so the Go2 data flows through the same posed-RGB-D
``SequenceSource`` interface as Replica / ProcTHOR.

This is the ROS-free port of v1 ``scripts_r2s3d/utils.ProjectionUtils.lidar2depth``,
with two deliberate changes: depth is returned in **meters** as float32 (v1 packed
``z * 256`` into uint16), and a **nearest-point-wins z-buffer** replaces v1's
last-write-wins so foreground surfaces occlude background points at shared pixels.

Frame convention matches :mod:`r2s3d_core.data.base`: camera = OpenCV optical
(x right, y down, z forward); ``T_cam_world`` maps world column-vectors into the
camera. Pair it with ``T_world_cam = frames.T_odom_cam_go2(odom)``.
"""

from __future__ import annotations

import numpy as np


def project_cloud_to_depth(
    world_pts: np.ndarray,
    T_cam_world: np.ndarray,
    K: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """Render a metric depth image by projecting a world cloud into a camera.

    Parameters
    ----------
    world_pts : (N, 3) float
        Points in the gravity-aligned Z-up world (odom) frame.
    T_cam_world : (4, 4) float
        World -> camera (OpenCV optical). ``invert(T_world_cam)``.
    K : (3, 3) float
        Pinhole intrinsics for the target resolution.
    height, width : int
        Output image size (must match the RGB frame these are paired with).

    Returns
    -------
    depth : (height, width) float32
        Per-pixel nearest surface depth in meters; ``0.0`` where no point projects.
    """
    depth = np.zeros((int(height), int(width)), dtype=np.float32)
    pts = np.asarray(world_pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return depth

    # world -> camera
    cam = (T_cam_world[:3, :3] @ pts.T).T + T_cam_world[:3, 3]
    z = cam[:, 2]
    front = z > 0.0
    cam = cam[front]
    z = z[front]
    if cam.shape[0] == 0:
        return depth

    # camera -> pixels
    uvw = K @ cam.T
    u = uvw[0] / uvw[2]
    v = uvw[1] / uvw[2]
    ui = np.floor(u).astype(np.int64)  # matches v1 truncation on positive coords
    vi = np.floor(v).astype(np.int64)
    inb = (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    ui, vi, z = ui[inb], vi[inb], z[inb]
    if ui.size == 0:
        return depth

    # nearest-point-wins z-buffer: assign far->near so the nearest survives when
    # several points hit the same pixel (numpy keeps the last write per index).
    order = np.argsort(-z)
    depth[vi[order], ui[order]] = z[order].astype(np.float32)
    return depth
