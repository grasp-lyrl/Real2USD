"""ProcTHOR / MolmoSpaces SequenceSource backend (AI2-THOR native).

Loads a ProcTHOR-10k house by numeric id via ``prior``, drives an ``ai2thor``
Controller over a deterministic reachable-position trajectory, and yields posed
RGB-D ``Frame``s + oriented-box ``GTObject``s in this package's Z-up OpenCV-optical
convention (see ``data/base.py``).

Why this backend exists: a coworker's holistic 3D-scene-graph benchmark evaluates
methods (Hydra/Khronos/ConceptGraphs/...) on a fixed slice of ProcTHOR-10k houses
sourced through Ai2's MolmoSpaces. This backend lets Real2USD produce the two
row-groups it can honestly answer on the *same scenes* — Objects (P/R/F1, counts,
class-free geo recall) and Mesh (Chamfer / footprint IoU, once GT meshes are wired).
Rooms/Places/Building/Trajectory/Grounding are out of scope until those layers exist.

Coordinate conventions (LOAD-BEARING — validated by a back-projection round-trip test,
tests/test_procthor_frames.py; do not touch without re-running it):

* AI2-THOR world is Unity: **left-handed, Y-up**, meters, degrees. Objects and the
  camera live here. We map Unity -> our world W (**right-handed, Z-up**) by
  ``W = (x_u, z_u, y_u)`` (permutation ``_M_WU``, det = -1, so it also fixes the
  left->right handedness flip). Gravity: Unity +Y -> our +Z.
* AI2-THOR camera: forward = +Z, up = +Y, right = +X (Unity left-handed). We emit the
  OpenCV optical frame (x right, y down, z forward). Camera orientation comes from the
  agent yaw (``rotation.y``, 0 faces Unity +Z, increasing clockwise seen from above)
  and ``cameraHorizon`` (pitch; positive tilts the camera down).
* Depth is metric meters; ``>= _THOR_FAR`` (far clip) and ``0`` are masked to NaN.
"""

from __future__ import annotations

import os
from typing import Iterator, List, Optional

import numpy as np
import trimesh

from .base import Frame, GTObject

# --- Unity(left-handed, Y-up) -> our world (right-handed, Z-up) --------------------
# W = M_WU @ p_unity  =>  (x, y, z)_W = (x, z, y)_unity.  det(M_WU) = -1 (LH->RH).
_M_WU = np.array([[1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0],
                  [0.0, 1.0, 0.0]])

_THOR_FAR = 20.0  # AI2-THOR default far clip; depth returns exactly this beyond range.

# Object types that are scene structure / not countable "objects" for the Objects
# rows. Kept explicit + loud (logged) because the exact object set is a comparability
# knob vs the external benchmark (see docs/ACTION_ITEMS.md AI-procthor-metrics).
_THOR_EXCLUDE_TYPES = {
    "floor", "wall", "window", "doorway", "door", "room",
}

# The MolmoSpaces / coworker slice (9 confirmed ids; a 10th + the split are pending —
# docs/ACTION_ITEMS.md). Split defaults to "train"; override via kwarg once confirmed.
PROCTHOR_SCENES = ["137", "200", "428", "534", "569", "573", "683", "771", "912"]


def intrinsics_from_fov(height: int, width: int, fov_deg: float,
                        fov_axis: str = "vertical") -> np.ndarray:
    """Pinhole K (square pixels, principal point at image center) from a Unity FOV.

    ``fov_axis`` selects whether ``fov_deg`` is the vertical or horizontal field of
    view. AI2-THOR reports ``metadata['fov']`` as the **vertical** FOV; the round-trip
    test confirms this choice for our render resolution.
    """
    f_from = height if fov_axis == "vertical" else width
    f = (f_from / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    return np.array([[f, 0.0, width / 2.0],
                     [0.0, f, height / 2.0],
                     [0.0, 0.0, 1.0]])


def thor_camera_to_world(camera_position: dict, yaw_deg: float,
                         horizon_deg: float) -> np.ndarray:
    """Build ``T_world_cam`` (OpenCV-optical camera -> our Z-up world) from AI2-THOR pose.

    Derives the Unity camera basis in Unity world, converts it to the OpenCV optical
    convention, then maps into our right-handed Z-up world via ``_M_WU``.
    """
    yaw = np.radians(yaw_deg)
    pitch = np.radians(horizon_deg)  # positive => looking down

    # Camera forward in Unity world: yaw about +Y (0 -> +Z, 90 -> +X), pitched down by
    # `horizon` (so +horizon lowers the +Z/+X forward toward -Y).
    fwd = np.array([np.sin(yaw) * np.cos(pitch),
                    -np.sin(pitch),
                    np.cos(yaw) * np.cos(pitch)])
    # Right is the horizontal forward rotated -90 deg about +Y (independent of pitch).
    right = np.array([np.cos(yaw), 0.0, -np.sin(yaw)])
    # Unity up = right x forward is left-handed; use fwd x right for a consistent up.
    up = np.cross(fwd, right)

    # OpenCV optical axes expressed in Unity world: x=right, y=-up (y is down), z=fwd.
    R_unity_cam = np.column_stack([right, -up, fwd])          # cols map cam-optical -> unity world
    R_world_cam = _M_WU @ R_unity_cam                          # -> our Z-up world
    t = _M_WU @ np.array([camera_position["x"],
                          camera_position["y"],
                          camera_position["z"]])
    T = np.eye(4)
    T[:3, :3] = R_world_cam
    T[:3, 3] = t
    return T


def _obb_from_corner_points(corner_points) -> tuple[np.ndarray, np.ndarray]:
    """(T_world_obj, extents) from 8 OOBB corner points given in *our* world frame.

    Order-agnostic via PCA: the covariance eigenvectors of a box's corners are its
    local axes (robust to aspect ratio, unlike nearest-neighbour edge picking, where
    a long box's shortest face-diagonal can be shorter than its longest edge). Axis
    labelling/sign is arbitrary — downstream rotation error is axis-labeling-invariant
    (see [[box-rotation-metric-gotcha]]).
    """
    c = np.asarray(corner_points, float)
    center = c.mean(axis=0)
    X = c - center
    _, V = np.linalg.eigh(X.T @ X)   # columns: principal axes (orthonormal)
    if np.linalg.det(V) < 0:         # keep a right-handed frame
        V[:, 2] = -V[:, 2]
    proj = X @ V
    extents = proj.max(axis=0) - proj.min(axis=0)
    T = np.eye(4)
    T[:3, :3] = V
    T[:3, 3] = center
    return T, extents


class ProcThorSource:
    """Posed RGB-D over a ProcTHOR-10k house, rendered live by AI2-THOR.

    ``scene`` is the house id as a string (e.g. ``"137"``); ``root`` is unused (houses
    come from the ``prior`` cache) but kept for the registry signature.
    """

    def __init__(self, root=None, scene: str = "137", split: str = "train",
                 stride: int = 1, load_gt: bool = True, width: int = 640,
                 height: int = 480, position_stride: int = 20,
                 yaws=(0, 90, 180, 270), horizons=(0, 30),
                 platform: Optional[str] = None, quality: str = "Low",
                 max_frames: Optional[int] = None, gt_mesh: str = "box",
                 native_masks: bool = True):
        self.scene = str(scene)
        self.split = split
        self.stride = stride
        self._load_gt = load_gt
        # native_masks: expose AI2-THOR's pixel-perfect instance masks (via native_mask())
        # so sam3d_layout feeds SAM3D the TRUE object silhouette (occlusion-aware) instead
        # of the box-silhouette proxy — a faithful GT-detector ceiling. Stores one seg
        # image per frame + a global objectId->color map, decoding per-object on demand
        # (avoids holding one mask per object per frame in RAM).
        self.native_masks = native_masks
        # gt_mesh: "box" attaches the OBB as a trimesh so the sam3d_layout GT-mask path
        # (render_instance_mask / select_best_view) works before real asset meshes exist
        # (AI-8). "none" leaves mesh=None. Box silhouettes are a superset of the true
        # object -> a faithful GT-box-detector ceiling for the Objects rows; do NOT read
        # Chamfer/F-score off box meshes (run those methods with --no-geometry).
        self.gt_mesh = gt_mesh
        self.width = width
        self.height = height
        self.position_stride = position_stride
        self.yaws = tuple(yaws)
        self.horizons = tuple(horizons)
        self.platform = platform
        self.quality = quality
        self.max_frames = max_frames
        self._controller = None
        self._gt: Optional[List[GTObject]] = None
        self._n_frames: Optional[int] = None
        self._seg_by_frame: dict = {}          # frame_id -> instance-seg image (uint8 HxWx3)
        self._inst_color: dict = {}            # instance_id -> np.array([r,g,b])
        self._objid_by_instance: dict = {}     # instance_id -> THOR objectId
        self.K = intrinsics_from_fov(height, width, 90.0)  # refreshed from metadata on start

    # -- controller lifecycle ------------------------------------------------------
    def _house(self):
        import prior
        ds = prior.load_dataset("procthor-10k")
        houses = ds[self.split]
        return houses[int(self.scene)]

    def _start(self):
        if self._controller is not None:
            return self._controller
        from ai2thor.controller import Controller
        kwargs = dict(scene=self._house(), renderDepthImage=True,
                      renderInstanceSegmentation=True, width=self.width,
                      height=self.height, gridSize=0.25, quality=self.quality)
        if self.platform:
            from ai2thor import platform as _plat
            kwargs["platform"] = getattr(_plat, self.platform)
        self._controller = Controller(**kwargs)
        fov = self._controller.last_event.metadata.get("fov", 90.0)
        self.K = intrinsics_from_fov(self.height, self.width, fov)
        return self._controller

    def close(self):
        if self._controller is not None:
            self._controller.stop()
            self._controller = None

    def _trajectory_positions(self):
        c = self._start()
        rp = c.step(action="GetReachablePositions").metadata["actionReturn"] or []
        rp = sorted(rp, key=lambda p: (round(p["x"], 3), round(p["z"], 3)))
        return rp[:: max(1, self.position_stride)]

    # -- native GT instance masks (pixel-perfect, occlusion-aware) ------------------
    def native_mask(self, frame_id: int, instance_id: int):
        """Uint8 0/255 mask of GT ``instance_id`` in frame ``frame_id`` from AI2-THOR's
        instance segmentation, or None if that object isn't visible there. Requires
        ``native_masks=True`` and that the frame has been iterated (seg captured)."""
        if not self.native_masks:
            return None
        if not self._inst_color:
            self.gt()  # build instance_id -> color / objectId maps
        color = self._inst_color.get(instance_id)
        seg = self._seg_by_frame.get(frame_id)
        if color is None or seg is None:
            return None
        m = np.all(seg == color, axis=-1)
        return (m.astype(np.uint8) * 255) if m.any() else None

    # -- SequenceSource protocol ---------------------------------------------------
    def __iter__(self) -> Iterator[Frame]:
        c = self._start()
        if self.native_masks and not self._inst_color:
            self.gt()  # ensure instance->color map is ready before capturing seg
        self._seg_by_frame = {}
        positions = self._trajectory_positions()
        fid = 0
        emitted = 0
        for pos in positions:
            for yaw in self.yaws:
                for horizon in self.horizons:
                    ev = c.step(action="Teleport", position=pos,
                                rotation=dict(x=0, y=float(yaw), z=0),
                                horizon=float(horizon), standing=True)
                    if not ev.metadata["lastActionSuccess"]:
                        continue
                    depth = np.asarray(ev.depth_frame, np.float32).copy()
                    depth[(depth <= 0) | (depth >= _THOR_FAR)] = np.nan
                    T = thor_camera_to_world(ev.metadata["cameraPosition"],
                                             float(yaw), float(horizon))
                    if self.native_masks:
                        self._seg_by_frame[fid] = np.asarray(
                            ev.instance_segmentation_frame, np.uint8)
                    yield Frame(
                        rgb=np.asarray(ev.frame, np.uint8),
                        depth=depth,
                        K=self.K.copy(),
                        T_world_cam=T,
                        stamp=float(fid),
                        frame_id=fid,
                    )
                    fid += 1
                    emitted += 1
                    if self.max_frames and emitted >= self.max_frames:
                        self._n_frames = emitted
                        return
        self._n_frames = emitted

    def __len__(self) -> int:
        if self._n_frames is not None:
            return self._n_frames
        positions = self._trajectory_positions()
        n = len(positions) * len(self.yaws) * len(self.horizons)
        return n if not self.max_frames else min(n, self.max_frames)

    def gt(self) -> Optional[List[GTObject]]:
        if not self._load_gt:
            return None
        if self._gt is not None:
            return self._gt
        c = self._start()
        obj_color = getattr(c.last_event, "object_id_to_color", {}) or {}
        objs: List[GTObject] = []
        self._inst_color = {}
        self._objid_by_instance = {}
        n_excluded = 0
        for o in c.last_event.metadata["objects"]:
            if o["objectType"].lower() in _THOR_EXCLUDE_TYPES:
                n_excluded += 1
                continue
            oobb = o.get("objectOrientedBoundingBox")
            if oobb and oobb.get("cornerPoints"):
                corners_w = (_M_WU @ np.asarray(oobb["cornerPoints"], float).T).T
                T_world_obj, extents = _obb_from_corner_points(corners_w)
            else:
                aabb = o["axisAlignedBoundingBox"]
                corners_w = (_M_WU @ np.asarray(aabb["cornerPoints"], float).T).T
                T_world_obj, extents = _obb_from_corner_points(corners_w)
            mesh = None
            if self.gt_mesh == "box":
                # OBB as a mesh: enables the sam3d_layout GT-mask path. NOT a real shape
                # (Mesh rows need asset meshes, AI-8) -> run those methods --no-geometry.
                mesh = trimesh.creation.box(extents=extents, transform=T_world_obj)
            inst_id = len(objs)
            col = obj_color.get(o["objectId"])
            if col is not None:
                self._inst_color[inst_id] = np.asarray(col, np.uint8)
                self._objid_by_instance[inst_id] = o["objectId"]
            objs.append(GTObject(
                instance_id=inst_id,
                label=o["objectType"].lower(),
                T_world_obj=T_world_obj,
                extents=extents,
                mesh=mesh,
            ))
        self._gt = objs
        return objs
