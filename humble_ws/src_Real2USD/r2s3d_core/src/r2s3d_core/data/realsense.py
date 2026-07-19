"""RealSense SequenceSource for the Go2 real-robot scenes (primary real-robot leg).

The Go2 RealSense bags carry proper aligned RGB-D:
  * ``/camera[/camera]/color/image_raw``                 rgb8, 640x480
  * ``/camera[/camera]/aligned_depth_to_color/image_raw`` 16UC1 mm, aligned to color
  * ``/camera[/camera]/color/camera_info``               intrinsics (plumb_bob, D=0)
  * ``/utlidar/robot_pose``                              PoseStamped in the odom/GT frame

This is strictly better than the lidar-projection path (:class:`data.rosbag.RosbagSource`):
depth is dense (~90% valid) and hardware-aligned to color, color is pre-rectified, and
the pose already lives in the same absolute-odom frame the Supervisely GT boxes use.

Iteration is **depth-triggered** (mirroring the v1 ``realsense_cam_node``): each aligned
depth frame pairs with the most recent color frame and the nearest robot pose, yielding one
:class:`~r2s3d_core.data.base.Frame`. Camera pose uses :func:`r2s3d_core.frames.T_odom_cam_go2`
(the v1 RealSense extrinsic default ``[0.285, 0, 0.01]``); an ``extrinsic`` override is
exposed for a future RS-specific refinement.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np

from .. import frames
from .base import Frame, GTObject
from .rosbag import _stamp_to_sec, _typestore
from .supervisely import load_supervisely_gt

log = logging.getLogger(__name__)

_COLOR = ("/camera/camera/color/image_raw", "/camera/color/image_raw")
_DEPTH = ("/camera/camera/aligned_depth_to_color/image_raw", "/camera/aligned_depth_to_color/image_raw")
_INFO = ("/camera/camera/color/camera_info", "/camera/color/camera_info")
_POSE = "/utlidar/robot_pose"

# Go2 RealSense scenes with Supervisely 3D-cuboid GT (color rgb8 + aligned depth,
# pose in the GT odom frame — all verified 2026-07-18). Maps scene id ->
# (bag dir under $R2S3D_RS_BAGS, Supervisely json basename).
_RS_BAG_ROOT = Path(os.environ.get("R2S3D_RS_BAGS", "/data/go2/rs"))
_RS_SCENES = {
    "lounge-0": ("lounge-0-4132025", "lounge-0_voxel_pointcloud.pcd.json"),
    "smalloffice-0": ("smalloffice-0-4132025", "smalloffice-0_voxel_pointcloud.pcd.json"),
    "smalloffice-1": ("smalloffice-1-4132025", "smalloffice-1_voxel_pointcloud.pcd.json"),
    "hallway-1": ("hallway_rs_01", "hallway-1_voxel_pointcloud.pcd.json"),
}


def resolve_rs_scene(scene: str, root=None):
    """Resolve a scene id to (RealSense bag_path, Supervisely gt_json). Tolerates
    aliases (``lounge0`` / ``smalloffice0`` / ``hallway1``)."""
    from .rosbag import _supervisely_dir

    key = scene.strip().lower()
    if key not in _RS_SCENES:
        alias = {"smalloffice0": "smalloffice-0", "smalloffice1": "smalloffice-1",
                 "lounge0": "lounge-0", "hallway1": "hallway-1"}
        key = alias.get(key.replace("_", "").replace("-", ""), key)
    if key not in _RS_SCENES:
        raise ValueError(f"unknown Go2 RealSense scene {scene!r}; have {sorted(_RS_SCENES)}")
    bag_name, gt_name = _RS_SCENES[key]
    bag_root = Path(root) if root else _RS_BAG_ROOT
    gt = _supervisely_dir() / gt_name
    return bag_root / bag_name, (gt if gt.is_file() else None)


class RealSenseSource:
    """A single Go2 RealSense bag as a :class:`~r2s3d_core.data.base.SequenceSource`.

    Parameters
    ----------
    bag_path : rosbag2 directory (``metadata.yaml`` + ``*.db3``).
    gt_json : Supervisely ``.pcd.json`` for this scene (``None`` -> no GT).
    stride : yield every ``stride``-th depth-triggered frame.
    depth_min, depth_max : reliability gate (m); depth outside -> invalid (0).
    canonical_only : restrict GT to the v1 chair/table/door classes.
    extrinsic : optional (3,) camera translation in the odom body frame; defaults
        to the v1 RealSense value via :func:`frames.T_odom_cam_go2`.
    """

    def __init__(
        self,
        bag_path: os.PathLike | str,
        gt_json: os.PathLike | str | None = None,
        stride: int = 1,
        depth_min: float = 0.2,
        depth_max: float = 5.0,
        canonical_only: bool = False,
        extrinsic: Optional[tuple] = None,
        scene: str = "scene",
    ) -> None:
        self.bag_path = Path(bag_path).expanduser()
        self.scene = scene  # used by detect/eval for per-scene output + SAM3D job keys
        if not self.bag_path.exists():
            raise FileNotFoundError(f"rosbag not found: {self.bag_path}")
        self.gt_json = Path(gt_json).expanduser() if gt_json else None
        self.stride = int(stride)
        self.depth_min = float(depth_min)
        self.depth_max = float(depth_max)
        self.canonical_only = canonical_only
        self.extrinsic = extrinsic

        self._gt_cache: Optional[List[GTObject]] = None
        self.K: Optional[np.ndarray] = None
        self.width = 0
        self.height = 0
        self._pose_t: Optional[np.ndarray] = None
        self._pose: List[dict] = []
        self._n_depth = 0
        self._topics: dict = {}

        self._read_static()

    # ------------------------------------------------------------------ static
    def _pick(self, conns, names):
        for c in conns:
            if c.topic in names:
                return c
        return None

    def _read_static(self) -> None:
        from rosbags.highlevel import AnyReader

        pose_t: List[float] = []
        with AnyReader([self.bag_path], default_typestore=_typestore()) as r:
            info = self._pick(r.connections, _INFO)
            color = self._pick(r.connections, _COLOR)
            depth = self._pick(r.connections, _DEPTH)
            pose = self._pick(r.connections, (_POSE,))
            if not all([info, color, depth, pose]):
                raise RuntimeError(
                    f"{self.bag_path.name}: missing RealSense topics "
                    f"(info={bool(info)} color={bool(color)} depth={bool(depth)} pose={bool(pose)})"
                )
            self._topics = {"color": color.topic, "depth": depth.topic, "pose": pose.topic}
            self._n_depth = int(getattr(depth, "msgcount", 0))
            # intrinsics from the first color camera_info
            for conn, _ts, raw in r.messages(connections=[info]):
                m = r.deserialize(raw, conn.msgtype)
                self.K = np.array(m.k, dtype=np.float64).reshape(3, 3)
                self.width, self.height = int(m.width), int(m.height)
                break
            # all poses (small). Sync by BAG RECORD TIME (tsn), not header.stamp:
            # /utlidar/robot_pose stamps its header on the robot's internal clock,
            # which is offset from the camera wall clock by ~months even though both
            # cover the same recording window — so header-stamp matching pins every
            # frame to one pose. The recorder's write time is a single clock for all
            # topics, so it pairs depth<->pose correctly.
            for conn, tsn, raw in r.messages(connections=[pose]):
                m = r.deserialize(raw, conn.msgtype)
                p, o = m.pose.position, m.pose.orientation
                pose_t.append(float(tsn))  # nanoseconds, bag record time
                self._pose.append({"position": [p.x, p.y, p.z], "orientation": [o.x, o.y, o.z, o.w]})

        if self.K is None:
            raise RuntimeError(f"{self.bag_path.name}: no color camera_info")
        if not self._pose:
            raise RuntimeError(f"{self.bag_path.name}: no {_POSE}")
        order = np.argsort(pose_t)
        self._pose_t = np.asarray(pose_t)[order]
        self._pose = [self._pose[i] for i in order]
        log.info("%s: %d depth-frames, %d poses, K fx=%.1f %dx%d",
                 self.bag_path.name, self._n_depth, len(self._pose), self.K[0, 0], self.width, self.height)

    def _nearest_pose(self, stamp: float) -> dict:
        i = int(np.searchsorted(self._pose_t, stamp))
        if i <= 0:
            return self._pose[0]
        if i >= len(self._pose_t):
            return self._pose[-1]
        lo, hi = i - 1, i
        return self._pose[lo if (stamp - self._pose_t[lo]) <= (self._pose_t[hi] - stamp) else hi]

    def T_world_cam(self, pose: dict) -> np.ndarray:
        kw = {"t_body_cam": tuple(self.extrinsic)} if self.extrinsic is not None else {}
        return frames.T_odom_cam_go2(pose, odom_quat="xyzw", **kw)

    # ------------------------------------------------------------------ frames
    def __len__(self) -> int:
        # count of depth-triggered frames after stride (upper bound: n_depth)
        return (self._n_depth + self.stride - 1) // self.stride if self._n_depth else 0

    def __iter__(self) -> Iterator[Frame]:
        from rosbags.highlevel import AnyReader

        with AnyReader([self.bag_path], default_typestore=_typestore()) as r:
            color_c = self._pick(r.connections, _COLOR)
            depth_c = self._pick(r.connections, _DEPTH)
            latest_rgb: Optional[np.ndarray] = None
            depth_i = 0
            for conn, tsn, raw in r.messages(connections=[color_c, depth_c]):
                if conn.topic == color_c.topic:
                    latest_rgb = self._decode_rgb(r.deserialize(raw, conn.msgtype))
                    continue
                # depth message -> emit a frame if we have a color yet. Pair to the
                # nearest pose by BAG RECORD TIME (tsn), see _read_static.
                idx = depth_i
                depth_i += 1
                if latest_rgb is None or (idx % self.stride):
                    continue
                m = r.deserialize(raw, conn.msgtype)
                depth = self._decode_depth(m)
                T_wc = self.T_world_cam(self._nearest_pose(float(tsn)))
                yield Frame(
                    rgb=latest_rgb.copy(), depth=depth, K=self.K.copy(),
                    T_world_cam=T_wc, stamp=float(tsn) * 1e-9, frame_id=idx,
                )

    def _decode_rgb(self, msg) -> np.ndarray:
        h, w = int(msg.height), int(msg.width)
        img = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(h, w, 3)
        enc = (msg.encoding or "rgb8").lower()
        if enc == "bgr8":
            img = img[:, :, ::-1]
        elif enc != "rgb8":
            log.warning("unexpected color encoding %r; assuming rgb8", msg.encoding)
        return np.ascontiguousarray(img)

    def _decode_depth(self, msg) -> np.ndarray:
        h, w = int(msg.height), int(msg.width)
        raw = np.frombuffer(bytes(msg.data), dtype=np.uint16).reshape(h, w)
        depth_m = raw.astype(np.float32) / 1000.0  # RealSense 16UC1 mm -> m
        depth_m[(depth_m < self.depth_min) | (depth_m > self.depth_max)] = 0.0
        return depth_m

    # ---------------------------------------------------------------------- GT
    def gt(self) -> Optional[List[GTObject]]:
        if self.gt_json is None:
            return None
        if self._gt_cache is None:
            self._gt_cache = load_supervisely_gt(self.gt_json, canonical_only=self.canonical_only)
        return self._gt_cache
