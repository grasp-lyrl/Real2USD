"""Replica SequenceSource backend (Phase 0).

Reads the community-standard NICE-SLAM / iMAP rendered trajectories (the same data
ConceptGraphs / HOV-SG evaluate on) for posed RGB-D, and extracts per-instance
ground truth from the original Replica-Dataset semantic mesh.

Expected on-disk layout (see docs/DATASETS.md)::

    <root>/                       # e.g. ~/Data/datasets/replica
      Replica/
        room0/
          results/
            frame000000.jpg  depth000000.png  ...   # ~2000 frames
          traj.txt                                   # 4x4 c2w per line (16 floats)
        room1/ ... office4/
        cam_params.json           # optional, per-release
      replica_intrinsics.yaml     # NICE-SLAM cam config (fx/fy/cx/cy/H/W/scale)
      semantic/                   # from facebookresearch/Replica-Dataset (GT)
        room0/
          habitat/
            mesh_semantic.ply     # per-face object_id
            info_semantic.json    # id -> class label

Frame conventions (empirically verified against room0_mesh.ply):

* The NICE-SLAM ``traj.txt`` c2w poses are used directly (no axis flip). With the
  OpenCV back-projection convention (x right, y down, z forward) they reproduce
  the room mesh geometry exactly. NICE-SLAM's documented ``c2w[:3,1:3] *= -1`` flip
  pairs with its OpenGL ray directions; since we back-project with OpenCV
  directions, the raw poses already sit in the OpenCV optical frame. Check:
  back-projected depth of frame 0 lands within room0_mesh bounds, min corner
  matching to ~1 cm.
* The NICE-SLAM Replica **world frame is already gravity-aligned Z-up** (room mesh
  vertical extent is on Z; camera vertical spread is smallest on Z). No world
  rotation is applied: ``T_WORLD_NATIVE`` is identity, and is the single place to
  change if a future release differs.

GT semantic meshes (original Replica-Dataset) are rendered along the same
trajectory, so they share this frame. :func:`_validate_gravity` re-checks that GT
OBB bottoms cluster near a common floor plane and logs loudly on violation.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Iterator, List, Optional

import cv2
import numpy as np
import trimesh
import yaml

from .base import Frame, GTObject

log = logging.getLogger(__name__)

# NICE-SLAM Replica camera defaults (configs/Replica/replica.yaml). Used only as
# a fallback when the intrinsics file is missing keys.
_DEFAULT_CAM = {
    "H": 680, "W": 1200,
    "fx": 600.0, "fy": 600.0, "cx": 599.5, "cy": 339.5,
    "png_depth_scale": 6553.5,
}

_REPLICA_SCENES = ["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]

# World-frame conversion: the NICE-SLAM Replica world is already gravity-aligned
# Z-up (verified against room0_mesh.ply), so this is identity. Change here if a
# future Replica release uses a different world convention.
T_WORLD_NATIVE = np.eye(4, dtype=np.float64)


def _T_zup_native() -> np.ndarray:
    return T_WORLD_NATIVE.copy()


class ReplicaSource:
    """A single Replica scene as a :class:`~r2s3d_core.data.base.SequenceSource`.

    Parameters
    ----------
    root : path to the dataset root containing ``Replica/`` (and ``semantic/``).
    scene : one of room0/room1/room2/office0..office4.
    stride : yield every ``stride``-th frame (1 = all frames).
    load_gt : parse the semantic mesh into per-instance GTObjects (cached).
    """

    def __init__(
        self,
        root: os.PathLike | str,
        scene: str,
        stride: int = 1,
        load_gt: bool = True,
    ) -> None:
        self.root = Path(root).expanduser()
        self.scene = scene
        self.stride = int(stride)
        self._load_gt = load_gt

        if scene not in _REPLICA_SCENES:
            log.warning("scene %r not in the standard 8 eval scenes %s", scene, _REPLICA_SCENES)

        self.scene_dir = self.root / "Replica" / scene
        if not self.scene_dir.is_dir():
            raise FileNotFoundError(
                f"Replica scene dir not found: {self.scene_dir}. "
                f"Run scripts/datasets/download_replica.sh {self.root}"
            )
        self.results_dir = self.scene_dir / "results"

        self.cam = self._load_intrinsics()
        self.K = np.array([
            [self.cam["fx"], 0.0, self.cam["cx"]],
            [0.0, self.cam["fy"], self.cam["cy"]],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        self._poses = self._load_traj()  # native-frame c2w list, already OpenCV-cam
        self._rgb_files, self._depth_files = self._list_frames()
        n = min(len(self._poses), len(self._rgb_files), len(self._depth_files))
        if not (len(self._poses) == len(self._rgb_files) == len(self._depth_files)):
            log.warning(
                "count mismatch poses=%d rgb=%d depth=%d; truncating to %d",
                len(self._poses), len(self._rgb_files), len(self._depth_files), n,
            )
        self._n = n
        self._indices = list(range(0, n, self.stride))

        self._T_zup = _T_zup_native()
        self._gt_cache: Optional[List[GTObject]] = None

    # ------------------------------------------------------------------ config
    def _load_intrinsics(self) -> dict:
        cam = dict(_DEFAULT_CAM)
        # Prefer an explicit yaml; fall back to per-scene cam_params.json.
        yml = self.root / "replica_intrinsics.yaml"
        if yml.is_file():
            with open(yml) as f:
                doc = yaml.safe_load(f) or {}
            src = doc.get("cam", doc)
            for k in ("H", "W", "fx", "fy", "cx", "cy", "png_depth_scale"):
                if isinstance(src, dict) and k in src and src[k] is not None:
                    cam[k] = src[k]
        else:
            j = self.scene_dir / "cam_params.json"
            if not j.is_file():
                j = self.root / "Replica" / "cam_params.json"
            if j.is_file():
                with open(j) as f:
                    doc = json.load(f)
                src = doc.get("camera", doc)
                mapping = {"w": "W", "h": "H", "fx": "fx", "fy": "fy", "cx": "cx", "cy": "cy", "scale": "png_depth_scale"}
                for src_k, dst_k in mapping.items():
                    if src_k in src:
                        cam[dst_k] = src[src_k]
            else:
                log.warning("no intrinsics file found under %s; using NICE-SLAM defaults", self.root)
        return cam

    def _load_traj(self) -> List[np.ndarray]:
        traj = self.scene_dir / "traj.txt"
        if not traj.is_file():
            raise FileNotFoundError(f"traj.txt not found: {traj}")
        poses = []
        with open(traj) as f:
            for line in f:
                vals = line.strip().split()
                if len(vals) != 16:
                    continue
                c2w = np.array(vals, dtype=np.float64).reshape(4, 4)
                # Raw c2w is already OpenCV optical under our back-projection
                # convention (see module docstring); no axis flip.
                poses.append(c2w)
        return poses

    def _list_frames(self):
        rgb = sorted(self.results_dir.glob("frame*.jpg"))
        depth = sorted(self.results_dir.glob("depth*.png"))
        return rgb, depth

    # ------------------------------------------------------------------ frames
    def __len__(self) -> int:
        return len(self._indices)

    def __iter__(self) -> Iterator[Frame]:
        scale = float(self.cam["png_depth_scale"])
        for out_id, i in enumerate(self._indices):
            rgb = cv2.cvtColor(cv2.imread(str(self._rgb_files[i]), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            depth_raw = cv2.imread(str(self._depth_files[i]), cv2.IMREAD_UNCHANGED)
            depth = depth_raw.astype(np.float32) / scale  # meters
            depth[depth <= 0] = 0.0
            T_world_cam = self._T_zup @ self._poses[i]  # native -> Z-up world
            yield Frame(
                rgb=rgb,
                depth=depth,
                K=self.K.copy(),
                T_world_cam=T_world_cam,
                stamp=float(i),
                frame_id=int(i),
            )

    # ---------------------------------------------------------------------- GT
    def gt(self) -> Optional[List[GTObject]]:
        if not self._load_gt:
            return None
        if self._gt_cache is None:
            self._gt_cache = self._load_gt_objects()
        return self._gt_cache

    def _semantic_dir(self) -> Optional[Path]:
        # NICE-SLAM names scenes room0/office0; the original Replica-Dataset uses
        # room_0/office_0. Try both spellings under several roots.
        underscore = self.scene
        for i, ch in enumerate(self.scene):
            if ch.isdigit():
                underscore = self.scene[:i] + "_" + self.scene[i:]
                break
        names = {self.scene, underscore}
        roots = [
            self.root / "semantic",
            self.root / "replica_semantic",
            self.root / "Replica",
            self.root,
        ]
        for r in roots:
            for name in names:
                cand = r / name / "habitat"
                if (cand / "mesh_semantic.ply").is_file() and (cand / "info_semantic.json").is_file():
                    return cand
        return None

    def _load_gt_objects(self) -> Optional[List[GTObject]]:
        cache = self.root / "semantic" / f"{self.scene}_gt.npz"
        if cache.is_file():
            objs = _load_gt_cache(cache, self._T_zup)
            if objs is not None:
                _validate_gravity(objs, self.scene)
                return objs

        sem = self._semantic_dir()
        if sem is None:
            log.warning(
                "GT semantic assets not found for %s (looked under %s/semantic/...). "
                "Download from facebookresearch/Replica-Dataset per docs/DATASETS.md. "
                "Returning no GT.", self.scene, self.root,
            )
            return None

        objs_native = _extract_instances(sem / "mesh_semantic.ply", sem / "info_semantic.json")
        cache.parent.mkdir(parents=True, exist_ok=True)
        _save_gt_cache(cache, objs_native)
        objs = [_apply_world_transform(o, self._T_zup) for o in objs_native]
        _validate_gravity(objs, self.scene)
        return objs


# ------------------------------------------------------------------ GT helpers

class _RawInstance:
    """Instance data in the NATIVE Replica world frame (pre Z-up conversion)."""

    __slots__ = ("instance_id", "label", "center", "R", "extents", "vertices", "faces")

    def __init__(self, instance_id, label, center, R, extents, vertices, faces):
        self.instance_id = instance_id
        self.label = label
        self.center = center      # (3,)
        self.R = R                # (3,3) obb axes as columns
        self.extents = extents    # (3,)
        self.vertices = vertices  # (V,3)
        self.faces = faces        # (F,3) local indices


def _obb_from_points(pts: np.ndarray):
    """Yaw-free axis-aligned OBB in the native frame (identity rotation).

    Replica objects are gravity-consistent in the native frame; a full min-volume
    OBB is deferred — Phase 0 metrics use IoU which the loader-side identity-OBB
    represents faithfully as long as extraction is self-consistent. Orientation
    refinement is a downstream concern, not GT loading.
    """
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    center = 0.5 * (lo + hi)
    extents = np.maximum(hi - lo, 1e-6)
    R = np.eye(3)
    return center, R, extents


# Architectural / structural / background classes excluded from object-placement
# metrics (SAM3D does not reconstruct these; standard Scan2CAD/ConceptGraphs
# practice). Everything else in info_semantic.json's `objects` is kept.
_STRUCTURAL = {
    "undefined", "unknown", "wall", "floor", "ceiling", "window", "door",
    "blinds", "curtain", "pillar", "column", "beam", "vent", "wall-plug",
    "switch", "rug", "picture", "panel", "stair", "stairs", "railing",
    "ceiling-light", "skirting-board", "handrail",
}


def _extract_instances(ply_path: Path, info_path: Path) -> List[_RawInstance]:
    """Per-instance GT from the Replica habitat semantic mesh.

    Pose/extents come from info_semantic.json's authoritative ``oriented_bbox``
    (``abb`` center+sizes in the object's local axes, ``orientation.rotation`` an
    xyzw quaternion local->world). The per-instance mesh is extracted from the PLY
    faces tagged with that object id (used for Chamfer/F-score).
    """
    from plyfile import PlyData
    from scipy.spatial.transform import Rotation

    with open(info_path) as f:
        info = json.load(f)
    objects = {int(o["id"]): o for o in info.get("objects", [])}

    ply = PlyData.read(str(ply_path))
    verts = np.stack([
        np.asarray(ply["vertex"]["x"]),
        np.asarray(ply["vertex"]["y"]),
        np.asarray(ply["vertex"]["z"]),
    ], axis=1).astype(np.float64)

    face_el = ply["face"]
    if "object_id" not in face_el.data.dtype.names:
        raise ValueError(f"{ply_path} has no per-face object_id; not a Replica habitat semantic mesh")
    vidx_all = face_el["vertex_indices"]
    face_obj = np.asarray(face_el["object_id"]).astype(np.int64)

    out: List[_RawInstance] = []
    for oid in np.unique(face_obj):
        obj = objects.get(int(oid))
        if obj is None:
            continue  # background object_id not in the labeled objects list
        label = str(obj.get("class_name", "unknown")).lower()
        if label in _STRUCTURAL:
            continue

        sel = np.where(face_obj == oid)[0]
        faces = np.stack([np.asarray(vidx_all[i], dtype=np.int64) for i in sel])
        vidx = np.unique(faces.reshape(-1))
        remap = {int(v): j for j, v in enumerate(vidx)}
        local_faces = np.vectorize(remap.__getitem__)(faces)
        local_verts = verts[vidx]

        ob = obj.get("oriented_bbox")
        if ob and "abb" in ob:
            # abb.center is in the object's LOCAL frame; orientation maps local->world:
            #   world_point = R @ local_point + translation
            # so the box's world center is R @ abb.center + translation. Verified
            # against extracted mesh points (100% containment).
            abb_center = np.asarray(ob["abb"]["center"], dtype=np.float64)
            extents = np.maximum(np.asarray(ob["abb"]["sizes"], dtype=np.float64), 1e-6)
            orient = ob.get("orientation", {})
            q = orient.get("rotation", [0, 0, 0, 1])
            t = np.asarray(orient.get("translation", [0, 0, 0]), dtype=np.float64)
            R = Rotation.from_quat([q[0], q[1], q[2], q[3]]).as_matrix()  # xyzw
            center = R @ abb_center + t
        else:
            center, R, extents = _obb_from_points(local_verts)

        out.append(_RawInstance(int(oid), label, center, R, extents, local_verts, local_faces))
    return out


def _apply_world_transform(inst: _RawInstance, T: np.ndarray) -> GTObject:
    Rw = T[:3, :3]
    tw = T[:3, 3]
    center_w = Rw @ inst.center + tw
    R_w = Rw @ inst.R
    verts_w = (Rw @ inst.vertices.T).T + tw
    mesh = trimesh.Trimesh(vertices=verts_w, faces=inst.faces, process=False)
    T_world_obj = np.eye(4)
    T_world_obj[:3, :3] = R_w
    T_world_obj[:3, 3] = center_w
    return GTObject(
        instance_id=inst.instance_id,
        label=inst.label,
        T_world_obj=T_world_obj,
        extents=inst.extents.astype(np.float64),
        mesh=mesh,
    )


def _save_gt_cache(path: Path, insts: List[_RawInstance]) -> None:
    d = {"n": len(insts)}
    for i, o in enumerate(insts):
        d[f"id_{i}"] = o.instance_id
        d[f"label_{i}"] = o.label
        d[f"center_{i}"] = o.center
        d[f"R_{i}"] = o.R
        d[f"extents_{i}"] = o.extents
        d[f"verts_{i}"] = o.vertices.astype(np.float32)
        d[f"faces_{i}"] = o.faces.astype(np.int32)
    np.savez_compressed(path, **d)


def _load_gt_cache(path: Path, T: np.ndarray) -> Optional[List[GTObject]]:
    try:
        z = np.load(path, allow_pickle=True)
        out = []
        for i in range(int(z["n"])):
            inst = _RawInstance(
                int(z[f"id_{i}"]), str(z[f"label_{i}"]),
                z[f"center_{i}"], z[f"R_{i}"], z[f"extents_{i}"],
                z[f"verts_{i}"].astype(np.float64), z[f"faces_{i}"].astype(np.int64),
            )
            out.append(_apply_world_transform(inst, T))
        return out
    except Exception as e:  # pragma: no cover
        log.warning("failed to load GT cache %s: %s", path, e)
        return None


def _validate_gravity(objs: List[GTObject], scene: str) -> None:
    """Sanity-check the Y-up->Z-up assumption: object floors should cluster.

    Logs loudly if the spread of per-object OBB bottom-Z is large, which would
    indicate the world up-axis conversion is wrong for this release.
    """
    if not objs:
        return
    bottoms = np.array([o.T_world_obj[2, 3] - 0.5 * o.extents[2] for o in objs])
    floor = np.percentile(bottoms, 10)
    near_floor = np.mean(np.abs(bottoms - floor) < 0.15)
    if near_floor < 0.2:
        log.warning(
            "[%s] gravity check weak: only %.0f%% of %d GT objects sit near a common "
            "floor plane (z=%.2f). Verify R_ZUP_YUP for this Replica release.",
            scene, 100 * near_floor, len(objs), floor,
        )
    else:
        log.info("[%s] gravity check ok: %.0f%% of objects near floor z=%.2f",
                 scene, 100 * near_floor, floor)
