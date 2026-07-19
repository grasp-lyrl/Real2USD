"""Go2 rosbag SequenceSource backend (real-robot leg).

Turns a recorded Unitree Go2 ROS2 bag into the same posed-RGB-D stream Replica /
ProcTHOR expose, so the real-robot scenes flow through one pipeline + eval path.

The Go2 has **no depth camera** — the bag carries RGB (``/camera/image_raw``) and a
LiDAR cloud (``/point_cloud2``) that is *already in the odom frame*. So we:

1. accumulate every ``/point_cloud2`` sweep into one odom-frame cloud (concat +
   voxel downsample) — the depth source;
2. read camera intrinsics from ``/camera/camera_info``;
3. for each RGB frame, find the nearest ``/odom`` and build ``T_world_cam`` via
   :func:`r2s3d_core.frames.T_odom_cam_go2` (raw odom, no re-centering — GT boxes
   are in absolute odom too);
4. project the accumulated cloud into that camera to synthesize a metric depth
   image (:func:`r2s3d_core.lidar_depth.project_cloud_to_depth`).

Reading is pure-Python via ``rosbags`` (the ``rosbag`` uv extra) — no ROS install,
and only the standard ``sensor_msgs``/``nav_msgs`` topics are deserialized (the
custom ``go2_interfaces`` types are never touched). GT is the Supervisely 3D
cuboids (:func:`r2s3d_core.data.supervisely.load_supervisely_gt`).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterator, List, Optional

import cv2
import numpy as np

from .. import frames
from ..lidar_depth import project_cloud_to_depth
from .base import Frame, GTObject
from .supervisely import load_supervisely_gt

log = logging.getLogger(__name__)

RGB_TOPIC = "/camera/image_raw"
INFO_TOPIC = "/camera/camera_info"
ODOM_TOPIC = "/odom"
CLOUD_TOPIC = "/point_cloud2"

_TYPESTORE = None

# Go2 real-robot scenes with Supervisely 3D-cuboid GT. Maps a scene id ->
# (bag directory under $R2S3D_GO2_BAGS, Supervisely json basename). Bag roots and
# GT dir are overridable via env for portability off this workstation.
_GO2_BAG_ROOT = Path(os.environ.get("R2S3D_GO2_BAGS", "/data/go2/lidar"))
_SCENES = {
    "smalloffice-0": ("smalloffice-0-4132025", "smalloffice-0_voxel_pointcloud.pcd.json"),
    "smalloffice-1": ("smalloffice-1-4132025", "smalloffice-1_voxel_pointcloud.pcd.json"),
    "lounge-0": ("lounge-0_4132025", "lounge-0_voxel_pointcloud.pcd.json"),
    # hallway-1 GT corresponds to the second hallway capture (…_01); verify odom
    # overlap with GT before trusting (see the alignment validation).
    "hallway-1": ("hallway_lidar_cam_202544_01", "hallway-1_voxel_pointcloud.pcd.json"),
}


def _supervisely_dir() -> Path:
    """Locate the Supervisely GT dir (env override, in-repo, then v1 backup)."""
    env = os.environ.get("R2S3D_SUPERVISELY_DIR")
    cands = [Path(env)] if env else []
    cands += [
        Path(__file__).resolve().parents[5] / "evaluations" / "supervisely",
        Path("/data/home_backup/hsu/repos/Real2USD/humble_ws/evaluations/supervisely"),
    ]
    for c in cands:
        if c.is_dir():
            return c
    return cands[0]


def resolve_scene(scene: str, root: str | os.PathLike | None = None) -> tuple[Path, Optional[Path]]:
    """Resolve a Go2 scene id to (bag_path, gt_json_path). Tolerates aliases like
    ``smalloffice0`` / ``lounge0`` / ``hallway1``."""
    key = scene.strip().lower()
    if key not in _SCENES:
        alias = {
            "smalloffice0": "smalloffice-0", "smalloffice1": "smalloffice-1",
            "lounge0": "lounge-0", "hallway1": "hallway-1",
        }
        key = alias.get(key.replace("_", "").replace("-", ""), key)
    if key not in _SCENES:
        raise ValueError(f"unknown Go2 scene {scene!r}; have {sorted(_SCENES)}")
    bag_name, gt_name = _SCENES[key]
    bag_root = Path(root) if root else _GO2_BAG_ROOT
    gt = _supervisely_dir() / gt_name
    return bag_root / bag_name, (gt if gt.is_file() else None)


def _typestore():
    global _TYPESTORE
    if _TYPESTORE is None:
        from rosbags.typesys import Stores, get_typestore
        _TYPESTORE = get_typestore(Stores.ROS2_HUMBLE)
    return _TYPESTORE


def _stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _cloud_xyz(msg) -> np.ndarray:
    """Extract (N,3) float32 xyz from a PointCloud2, honoring field offsets."""
    off = {f.name: (f.offset, f.datatype) for f in msg.fields}
    step = int(msg.point_step)
    n = int(msg.width) * int(msg.height)
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)[: n * step].reshape(n, step)
    cols = []
    for name in ("x", "y", "z"):
        o, _dt = off[name]
        cols.append(raw[:, o : o + 4].copy().view(np.float32).reshape(-1))
    xyz = np.stack(cols, axis=1).astype(np.float64)
    return xyz[np.isfinite(xyz).all(axis=1)]


class RosbagSource:
    """A single Go2 bag as a :class:`~r2s3d_core.data.base.SequenceSource`.

    Parameters
    ----------
    bag_path : rosbag2 directory (containing ``metadata.yaml`` + ``*.db3``).
    gt_json : Supervisely ``.pcd.json`` for this scene (``None`` -> no GT).
    stride : yield every ``stride``-th RGB frame.
    cloud_voxel : voxel size (m) for the accumulated depth cloud.
    max_range : optional; drop cloud points farther than this from the trajectory
        centroid (m) to trim stray far returns. ``None`` keeps all.
    cloud_cache : cache the downsampled cloud to ``$R2S3D_DATA/rosbag_cache``.
    canonical_only : restrict GT to the v1 chair/table/door classes.
    """

    def __init__(
        self,
        bag_path: os.PathLike | str,
        gt_json: os.PathLike | str | None = None,
        stride: int = 1,
        cloud_voxel: float = 0.03,
        max_range: Optional[float] = None,
        cloud_cache: bool = True,
        canonical_only: bool = False,
        scene: str = "scene",
    ) -> None:
        self.bag_path = Path(bag_path).expanduser()
        self.scene = scene  # used by detect/eval for per-scene output + SAM3D job keys
        if not self.bag_path.exists():
            raise FileNotFoundError(f"rosbag not found: {self.bag_path}")
        self.gt_json = Path(gt_json).expanduser() if gt_json else None
        self.stride = int(stride)
        self.cloud_voxel = float(cloud_voxel)
        self.max_range = max_range
        self.cloud_cache = cloud_cache
        self.canonical_only = canonical_only

        self._gt_cache: Optional[List[GTObject]] = None
        self.K: Optional[np.ndarray] = None
        self.D: Optional[np.ndarray] = None
        self._undistort_maps: Optional[tuple] = None
        self.width = 0
        self.height = 0
        self._odom_t: Optional[np.ndarray] = None      # (M,) stamps
        self._odom_pose: List[dict] = []               # aligned {position, orientation(xyzw)}
        self._n_rgb = 0

        self._read_static()  # K, odom index, accumulated cloud
        self._indices = list(range(0, self._n_rgb, self.stride))

    # ------------------------------------------------------------------ static
    def _read_static(self) -> None:
        from rosbags.highlevel import AnyReader

        odom_t: List[float] = []
        clouds: List[np.ndarray] = []
        with AnyReader([self.bag_path], default_typestore=_typestore()) as r:
            topics = {c.topic for c in r.connections}
            for t in (RGB_TOPIC, INFO_TOPIC, ODOM_TOPIC, CLOUD_TOPIC):
                if t not in topics:
                    log.warning("bag %s missing topic %s (have %s)", self.bag_path.name, t, sorted(topics))
            # RGB blobs are large; count them from metadata rather than streaming here.
            n_rgb = sum(int(getattr(c, "msgcount", 0)) for c in r.connections if c.topic == RGB_TOPIC)
            for conn, _ts, raw in r.messages(
                connections=[c for c in r.connections
                             if c.topic in (INFO_TOPIC, ODOM_TOPIC, CLOUD_TOPIC)]
            ):
                msg = r.deserialize(raw, conn.msgtype)
                if conn.topic == INFO_TOPIC and self.K is None:
                    self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
                    self.D = np.asarray(msg.d, dtype=np.float64)
                    self.width, self.height = int(msg.width), int(msg.height)
                elif conn.topic == ODOM_TOPIC:
                    p = msg.pose.pose.position
                    q = msg.pose.pose.orientation
                    odom_t.append(_stamp_to_sec(msg.header.stamp))
                    self._odom_pose.append({
                        "position": [p.x, p.y, p.z],
                        "orientation": [q.x, q.y, q.z, q.w],  # xyzw
                    })
                elif conn.topic == CLOUD_TOPIC:
                    xyz = _cloud_xyz(msg)
                    if xyz.size:
                        clouds.append(xyz)

        if self.K is None:
            raise RuntimeError(f"{self.bag_path.name}: no {INFO_TOPIC}; cannot form intrinsics")
        # Undistort RGB into the pinhole-K frame so it aligns with the K-projected
        # lidar depth (the Go2 camera has plumb_bob k1~-0.34 barrel distortion).
        if self.D is not None and np.any(np.abs(self.D) > 1e-8):
            self._undistort_maps = cv2.initUndistortRectifyMap(
                self.K, self.D, None, self.K, (self.width, self.height), cv2.CV_16SC2
            )
        if not self._odom_pose:
            raise RuntimeError(f"{self.bag_path.name}: no {ODOM_TOPIC}; cannot pose frames")
        self._odom_t = np.asarray(odom_t, dtype=np.float64)
        order = np.argsort(self._odom_t)
        self._odom_t = self._odom_t[order]
        self._odom_pose = [self._odom_pose[i] for i in order]
        self._n_rgb = n_rgb
        self._cloud = self._accumulate_cloud(clouds)
        log.info(
            "%s: %d rgb, %d odom, cloud %d pts (voxel %.3f), K fx=%.1f %dx%d",
            self.bag_path.name, n_rgb, len(self._odom_pose), self._cloud.shape[0],
            self.cloud_voxel, self.K[0, 0], self.width, self.height,
        )

    def _accumulate_cloud(self, clouds: List[np.ndarray]) -> np.ndarray:
        cache = self._cloud_cache_path()
        if cache and cache.is_file():
            try:
                return np.load(cache)
            except Exception as e:  # pragma: no cover
                log.warning("cloud cache load failed (%s); rebuilding", e)
        if not clouds:
            raise RuntimeError(f"{self.bag_path.name}: no {CLOUD_TOPIC} points; no depth source")
        pts = np.concatenate(clouds, axis=0)
        if self.max_range is not None:
            c = np.median(pts, axis=0)
            pts = pts[np.linalg.norm(pts - c, axis=1) <= self.max_range]
        pts = self._voxel_downsample(pts, self.cloud_voxel)
        if cache:
            try:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, pts)
            except Exception as e:  # pragma: no cover
                log.warning("cloud cache save failed: %s", e)
        return pts

    @staticmethod
    def _voxel_downsample(pts: np.ndarray, voxel: float) -> np.ndarray:
        try:
            import open3d as o3d
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(pts)
            return np.asarray(pc.voxel_down_sample(voxel).points)
        except Exception:
            # numpy voxel-grid dedup fallback (no open3d in the light env)
            keys = np.floor(pts / voxel).astype(np.int64)
            _, idx = np.unique(keys, axis=0, return_index=True)
            return pts[np.sort(idx)]

    def _cloud_cache_path(self) -> Optional[Path]:
        if not self.cloud_cache:
            return None
        root = Path(os.environ.get("R2S3D_DATA", str(Path.home() / "Data" / "datasets")))
        tag = f"{self.bag_path.name}_v{self.cloud_voxel:.3f}"
        if self.max_range is not None:
            tag += f"_r{self.max_range:g}"
        return root / "rosbag_cache" / f"{tag}.npy"

    # ------------------------------------------------------------------ poses
    def _nearest_odom(self, stamp: float) -> dict:
        i = int(np.searchsorted(self._odom_t, stamp))
        if i <= 0:
            return self._odom_pose[0]
        if i >= len(self._odom_t):
            return self._odom_pose[-1]
        lo, hi = i - 1, i
        return self._odom_pose[lo if (stamp - self._odom_t[lo]) <= (self._odom_t[hi] - stamp) else hi]

    def T_world_cam(self, odom: dict) -> np.ndarray:
        return frames.T_odom_cam_go2(odom, odom_quat="xyzw")

    # ------------------------------------------------------------------ frames
    def __len__(self) -> int:
        return len(self._indices)

    def __iter__(self) -> Iterator[Frame]:
        from rosbags.highlevel import AnyReader

        keep = set(self._indices)
        with AnyReader([self.bag_path], default_typestore=_typestore()) as r:
            conns = [c for c in r.connections if c.topic == RGB_TOPIC]
            rgb_i = 0
            for conn, _ts, raw in r.messages(connections=conns):
                idx = rgb_i
                rgb_i += 1
                if idx not in keep:
                    continue
                msg = r.deserialize(raw, conn.msgtype)
                rgb = self._decode_rgb(msg)
                stamp = _stamp_to_sec(msg.header.stamp)
                odom = self._nearest_odom(stamp)
                T_wc = self.T_world_cam(odom)
                depth = project_cloud_to_depth(
                    self._cloud, frames.invert(T_wc), self.K, self.height, self.width
                )
                yield Frame(
                    rgb=rgb, depth=depth, K=self.K.copy(),
                    T_world_cam=T_wc, stamp=stamp, frame_id=idx,
                )

    def _decode_rgb(self, msg) -> np.ndarray:
        h, w = int(msg.height), int(msg.width)
        buf = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(h, msg.step)[:, : w * 3]
        img = buf.reshape(h, w, 3)
        enc = (msg.encoding or "bgr8").lower()
        if enc == "bgr8":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif enc != "rgb8":
            log.warning("unexpected image encoding %r; assuming rgb8", msg.encoding)
        if self._undistort_maps is not None:
            img = cv2.remap(img, self._undistort_maps[0], self._undistort_maps[1], cv2.INTER_LINEAR)
        return np.ascontiguousarray(img)

    # ---------------------------------------------------------------------- GT
    def gt(self) -> Optional[List[GTObject]]:
        if self.gt_json is None:
            return None
        if self._gt_cache is None:
            self._gt_cache = load_supervisely_gt(self.gt_json, canonical_only=self.canonical_only)
        return self._gt_cache
