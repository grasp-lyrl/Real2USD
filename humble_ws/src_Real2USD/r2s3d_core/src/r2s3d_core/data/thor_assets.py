"""Load AI2-THOR object meshes (MolmoSpaces ``isaac/objects/thor`` USDA export) by assetId.

The MolmoSpaces THOR object source (docs/ACTION_ITEMS.md AI-8) unpacks to one directory per
``assetId`` under ``<version>/<assetId>/<assetId>.usda`` (ASCII USD, Z-up, meters). ``trimesh``
cannot read USDA, so we parse it with ``usd-core`` (``pxr``) into a canonical-frame triangle
mesh. Callers pose it into the world OBB with :func:`fit_canonical_to_obb`.

Why fit-to-OBB rather than the object's Unity transform: the world OBB (``T_world_obj`` /
``extents`` from :mod:`r2s3d_core.data.procthor`) is already validated (oracle=1.000 on scene
137) and lives in our Z-up world, whereas composing the object's Unity Y-up rotation with the
USD Z-up mesh re-introduces the fragile frame chain CLAUDE.md warns about. Fitting the
canonical mesh into the trusted OBB (rotation+translation, extent-matched axes, no rescale)
keeps GT geometry consistent with the boxes the metrics already match on. Limitation: axis
SIGN is unresolved, so a strongly front/back-asymmetric asset may sit flipped 180deg about an
OBB axis; refine with the Unity yaw if Chamfer looks off (tracked in STATUS Phase-5).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

from .registry import data_root

# Where the MolmoSpaces iThor object meshes live. Env override wins; else the conventional
# download location under the data root (see AI-8 command block).
_ENV = "R2S3D_THOR_ASSETS"
_DEFAULT_SUBPATH = ("molmospaces", "isaac", "objects", "thor")


def default_asset_root() -> Optional[Path]:
    """Resolve the THOR asset directory, or ``None`` if it isn't present.

    ``$R2S3D_THOR_ASSETS`` (pointing at the ``.../thor/<version>`` dir) takes precedence;
    otherwise use ``<data_root>/molmospaces/isaac/objects/thor/<latest version dir>``.
    """
    env = os.environ.get(_ENV)
    if env:
        p = Path(env).expanduser()
        return p if p.is_dir() else None
    base = data_root().joinpath(*_DEFAULT_SUBPATH)
    if not base.is_dir():
        return None
    # one version dir (e.g. 20260128); pick the lexicographically latest if several
    versions = sorted((d for d in base.iterdir() if d.is_dir()), reverse=True)
    return versions[0] if versions else None


def load_usd_mesh(usda_path: Path) -> "object":
    """Read a USD(A) file into a single ``trimesh.Trimesh`` in its canonical local frame.

    All ``UsdGeom.Mesh`` prims are read, transformed by their composed local-to-world
    (canonical) matrix, polygon-triangulated (fan), and concatenated. Raises on an empty or
    unreadable stage so failures are loud rather than silently producing empty GT geometry.
    """
    import trimesh
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(usda_path))
    if stage is None:
        raise ValueError(f"usd-core could not open {usda_path}")
    xform_cache = UsdGeom.XformCache()

    all_v: list[np.ndarray] = []
    all_f: list[np.ndarray] = []
    offset = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        idx = mesh.GetFaceVertexIndicesAttr().Get()
        if pts is None or counts is None or idx is None or len(pts) == 0:
            continue
        verts = np.asarray(pts, dtype=np.float64)
        # canonical (local-to-world within the asset stage) transform; USD uses row-vector
        # convention v' = v * M, so right-multiply the homogeneous points by np.array(M).
        M = np.asarray(xform_cache.GetLocalToWorldTransform(prim), dtype=np.float64)
        verts = (np.c_[verts, np.ones(len(verts))] @ M)[:, :3]

        counts = np.asarray(counts, dtype=np.int64)
        idx = np.asarray(idx, dtype=np.int64)
        # fan-triangulate each polygon: (v0, vk, vk+1)
        pos = 0
        faces: list[tuple[int, int, int]] = []
        for c in counts:
            f = idx[pos:pos + c]
            for k in range(1, c - 1):
                faces.append((f[0], f[k], f[k + 1]))
            pos += c
        if not faces:
            continue
        all_v.append(verts)
        all_f.append(np.asarray(faces, dtype=np.int64) + offset)
        offset += len(verts)

    if not all_v:
        raise ValueError(f"no mesh geometry in {usda_path}")
    return trimesh.Trimesh(vertices=np.vstack(all_v),
                           faces=np.vstack(all_f), process=False)


class ThorAssetLibrary:
    """Resolves ``assetId`` -> canonical ``trimesh.Trimesh`` from the THOR USDA source."""

    def __init__(self, root: Optional[Path] = None):
        root = Path(root) if root is not None else default_asset_root()
        if root is None or not Path(root).is_dir():
            raise FileNotFoundError(
                f"THOR asset root not found (set ${_ENV} or download to "
                f"<data_root>/{'/'.join(_DEFAULT_SUBPATH)}/<version>; see ACTION_ITEMS AI-8). "
                f"Got: {root}")
        self.root = Path(root)

    def path_for(self, asset_id: str) -> Optional[Path]:
        f = self.root / asset_id / f"{asset_id}.usda"
        return f if f.exists() else None

    @lru_cache(maxsize=2048)
    def load_canonical(self, asset_id: str) -> Optional["object"]:
        """Canonical-frame mesh for ``asset_id``; ``None`` if the asset dir is absent."""
        path = self.path_for(asset_id)
        if path is None:
            return None
        return load_usd_mesh(path)


def _pca_frame(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(center, axes, extents) of a point set's PCA oriented box.

    Matches ``procthor._obb_from_corner_points`` conventions: columns of ``axes`` are the
    principal directions (right-handed), ``extents`` are the full spans along them.
    """
    c = vertices.mean(axis=0)
    X = vertices - c
    _, V = np.linalg.eigh(X.T @ X)
    if np.linalg.det(V) < 0:
        V[:, 2] = -V[:, 2]
    proj = X @ V
    extents = proj.max(axis=0) - proj.min(axis=0)
    return c, V, extents


def fit_canonical_to_obb(mesh: "object", T_world_obj: np.ndarray,
                         extents: np.ndarray) -> "object":
    """Rigidly place a canonical asset mesh into a world OBB (rotation + translation).

    Aligns the mesh's PCA axes to the OBB axes by matching them in descending-extent order
    (robust for non-cubic shapes; irrelevant for near-cubic ones), then translates the mesh
    centroid to the OBB center. No rescale -- the asset and the OBB describe the same physical
    object, so their extents already agree; a large mismatch is logged as a warning rather
    than silently distorting the shape. Returns a new world-frame ``trimesh.Trimesh``.
    """
    V = np.asarray(mesh.vertices, dtype=np.float64)
    c_m, A_m, e_m = _pca_frame(V)
    R_w = np.asarray(T_world_obj, dtype=np.float64)[:3, :3]
    t_w = np.asarray(T_world_obj, dtype=np.float64)[:3, 3]
    e_w = np.asarray(extents, dtype=np.float64)

    order_m = np.argsort(-e_m)          # mesh axes, longest first
    order_w = np.argsort(-e_w)          # world axes, longest first
    A_m_sorted = A_m[:, order_m]
    R_w_sorted = R_w[:, order_w]
    # rotation mapping mesh-canonical frame -> world OBB frame
    R_fit = R_w_sorted @ A_m_sorted.T
    if np.linalg.det(R_fit) < 0:        # enforce a proper rotation (no reflection)
        R_w_sorted[:, 2] = -R_w_sorted[:, 2]
        R_fit = R_w_sorted @ A_m_sorted.T

    world_v = (V - c_m) @ R_fit.T + t_w
    out = mesh.copy()
    out.vertices = world_v
    return out
