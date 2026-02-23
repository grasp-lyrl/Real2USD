import json
import ast
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from label_matching import resolve_label_with_learning


def _pose_to_matrix(position_xyz, quat_xyzw):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
    T[:3, 3] = np.asarray(position_xyz, dtype=np.float64)
    return T


def _raw_to_zup_matrix() -> np.ndarray:
    try:
        from real2sam3d.ply_frame_utils import R_flip_z, R_yup_to_zup

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = (np.asarray(R_flip_z) @ np.asarray(R_yup_to_zup)).T
        return T
    except Exception:
        return np.eye(4, dtype=np.float64)


def _yaw_normalization_matrix(yaw_rad: float) -> np.ndarray:
    """
    Rotate scene by -yaw around Z so dominant GT orientation aligns with axes.
    """
    T = np.eye(4, dtype=np.float64)
    if abs(float(yaw_rad)) <= 1e-12:
        return T
    c = float(np.cos(-yaw_rad))
    s = float(np.sin(-yaw_rad))
    T[:3, :3] = np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return T


def estimate_supervisely_global_yaw(gt_json: str) -> float:
    """
    Estimate dominant scene yaw from Supervisely cuboid rotations.
    Uses weighted circular mean with XY footprint area as weight.
    """
    with open(gt_json) as f:
        data = json.load(f)
    angles = []
    weights = []
    for fig in data.get("figures", []):
        if fig.get("geometryType") not in (None, "cuboid_3d"):
            continue
        geom = fig.get("geometry", {})
        rot = geom.get("rotation", {})
        d = geom.get("dimensions", {})
        yaw = float(rot.get("z", 0.0))
        w = max(float(d.get("x", 0.0)) * float(d.get("y", 0.0)), 1e-6)
        angles.append(yaw)
        weights.append(w)
    if not angles:
        return 0.0
    angles = np.asarray(angles, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    s = float(np.sum(weights * np.sin(angles)))
    c = float(np.sum(weights * np.cos(angles)))
    return float(np.arctan2(s, c))


def _load_mesh_vertices(data_path: str) -> np.ndarray:
    import trimesh

    path = Path(data_path)
    if path.suffix.lower() == ".ply":
        glb = path.parent / "object.glb"
        if glb.exists():
            path = glb
    scene = trimesh.load(str(path), process=False)
    verts = []
    if isinstance(scene, trimesh.Scene):
        for node_name in scene.graph.nodes_geometry:
            node_tf, geom_name = scene.graph[node_name]
            geom = scene.geometry.get(geom_name)
            if isinstance(geom, trimesh.Trimesh):
                g = geom.copy()
                g.apply_transform(np.asarray(node_tf, dtype=np.float64))
                verts.append(np.asarray(g.vertices, dtype=np.float64))
    elif isinstance(scene, trimesh.Trimesh):
        verts.append(np.asarray(scene.vertices, dtype=np.float64))
    if not verts:
        raise RuntimeError(f"No mesh vertices found in {path}")
    return np.vstack(verts)


def _normalize_obb_vertical_last(
    center: np.ndarray, dims: np.ndarray, R: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reorder OBB axes so the one most aligned with world Z is last. Returns (center, dims_ordered, R_ordered)."""
    center = np.asarray(center, dtype=np.float64).ravel()[:3]
    dims = np.asarray(dims, dtype=np.float64).ravel()[:3]
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    z_component = np.abs(R[2, :])
    k_z = int(np.argmax(z_component))
    i_h = [i for i in range(3) if i != k_z]
    perm = i_h + [k_z]
    return center.copy(), dims[perm].copy(), R[:, perm].copy()


def _boxes_from_vertices(verts: np.ndarray, T: np.ndarray) -> Dict:
    ones = np.ones((verts.shape[0], 1), dtype=np.float64)
    vh = np.hstack([verts, ones])
    transformed = (T @ vh.T).T[:, :3]
    vmin = transformed.min(axis=0)
    vmax = transformed.max(axis=0)
    out = {
        "aabb_min": vmin.tolist(),
        "aabb_max": vmax.tolist(),
        "center": ((vmin + vmax) / 2.0).tolist(),
        "dimensions": (vmax - vmin).tolist(),
    }
    try:
        import trimesh

        obb = trimesh.points.PointCloud(transformed).bounding_box_oriented
        T_obb = np.asarray(obb.primitive.transform, dtype=np.float64)
        center = T_obb[:3, 3]
        dims = np.asarray(obb.primitive.extents, dtype=np.float64)
        R = T_obb[:3, :3]
        center, dims, R = _normalize_obb_vertical_last(center, dims, R)
        out["obb_center"] = center.tolist()
        out["obb_dimensions"] = dims.tolist()
        out["obb_rotation_matrix"] = R.tolist()
    except Exception:
        out["obb_center"] = out["center"]
        out["obb_dimensions"] = out["dimensions"]
        out["obb_rotation_matrix"] = np.eye(3, dtype=np.float64).tolist()
    return out


def load_predictions_scene_graph(
    scene_graph_json: str,
    canonical: set,
    alias_map: Dict[str, str],
    contains_rules: Dict[str, str],
    learned_alias_map: Dict[str, str],
    use_embedding_for_unresolved: bool,
    learn_new_aliases: bool,
    learned_alias_path: str,
    embedding_min_score: float,
    label_source: str = "prefer_retrieved_else_label",
    normalize_yaw_rad: float = 0.0,
) -> List[Dict]:
    raw_to_zup = _raw_to_zup_matrix()
    T_norm = _yaw_normalization_matrix(normalize_yaw_rad)
    with open(scene_graph_json) as f:
        data = json.load(f)
    objs = data.get("objects", [])
    out = []
    for obj in objs:
        label_raw = obj.get("label")
        if label_source == "prefer_retrieved_else_label" and obj.get("retrieved_object_label"):
            label_raw = obj.get("retrieved_object_label")
        slot_label_raw = obj.get("label")
        retrieved_label_raw = obj.get("retrieved_object_label")
        label_canonical, label_source_used = resolve_label_with_learning(
            label_raw or "",
            canonical=canonical,
            alias_map=alias_map,
            contains_rules=contains_rules,
            learned_alias_map=learned_alias_map,
            use_embedding_for_unresolved=use_embedding_for_unresolved,
            learn_new_aliases=learn_new_aliases,
            learned_alias_path=learned_alias_path,
            embedding_min_score=embedding_min_score,
        )

        slot_label_canonical, _ = resolve_label_with_learning(
            slot_label_raw or "",
            canonical=canonical,
            alias_map=alias_map,
            contains_rules=contains_rules,
            learned_alias_map=learned_alias_map,
            use_embedding_for_unresolved=use_embedding_for_unresolved,
            learn_new_aliases=learn_new_aliases,
            learned_alias_path=learned_alias_path,
            embedding_min_score=embedding_min_score,
        )
        retrieved_label_canonical = None
        if retrieved_label_raw:
            retrieved_label_canonical, _ = resolve_label_with_learning(
                retrieved_label_raw,
                canonical=canonical,
                alias_map=alias_map,
                contains_rules=contains_rules,
                learned_alias_map=learned_alias_map,
                use_embedding_for_unresolved=use_embedding_for_unresolved,
                learn_new_aliases=learn_new_aliases,
                learned_alias_path=learned_alias_path,
                embedding_min_score=embedding_min_score,
            )

        try:
            verts = _load_mesh_vertices(obj["data_path"])
            if obj.get("transform_odom_from_raw") is not None:
                T = np.asarray(obj["transform_odom_from_raw"], dtype=np.float64)
            else:
                T_pose = _pose_to_matrix(obj["position"], obj["orientation"])
                T = T_pose @ raw_to_zup
            T = T_norm @ T
            box = _boxes_from_vertices(verts, T)
        except Exception:
            # Fallback: degenerate box at pose center, keeps run analyzable even with missing mesh.
            pos = np.asarray(obj.get("position", [0, 0, 0]), dtype=np.float64)
            pos_h = np.hstack([pos, np.array([1.0], dtype=np.float64)])
            pos = (T_norm @ pos_h)[:3]
            box = {
                "aabb_min": pos.tolist(),
                "aabb_max": pos.tolist(),
                "center": pos.tolist(),
                "dimensions": [0.0, 0.0, 0.0],
                "obb_center": pos.tolist(),
                "obb_dimensions": [0.0, 0.0, 0.0],
                "obb_rotation_matrix": T_norm[:3, :3].tolist(),
            }

        out.append(
            {
                "id": obj.get("id"),
                "data_path": obj.get("data_path"),
                "job_id": obj.get("job_id"),
                "position": obj.get("position"),
                "orientation": obj.get("orientation"),
                "label_raw": label_raw,
                "slot_label_raw": slot_label_raw,
                "retrieved_label_raw": retrieved_label_raw,
                "label_canonical": label_canonical,
                "slot_label_canonical": slot_label_canonical,
                "retrieved_label_canonical": retrieved_label_canonical,
                "retrieval_swapped": bool(retrieved_label_raw),
                "label_resolution": label_source_used,
                **box,
            }
        )
    return out


def load_predictions_clio_graphml(
    graphml_file: str,
    canonical: set,
    alias_map: Dict[str, str],
    contains_rules: Dict[str, str],
    learned_alias_map: Dict[str, str],
    use_embedding_for_unresolved: bool,
    learn_new_aliases: bool,
    learned_alias_path: str,
    embedding_min_score: float,
    normalize_yaw_rad: float = 0.0,
) -> List[Dict]:
    import networkx as nx

    T_norm = _yaw_normalization_matrix(normalize_yaw_rad)
    G = nx.read_graphml(graphml_file)
    out = []
    for node_id, data in G.nodes(data=True):
        if data.get("node_type") != "object":
            continue

        raw_label = str(data.get("name", "")).replace("find the ", "").strip()
        if not raw_label:
            continue

        pos_str = data.get("position", "")
        dim_str = data.get("bbox_dim", "")
        rot_str = data.get("bbox_orientation", "")
        if not pos_str or not dim_str:
            continue
        try:
            pos = np.asarray(ast.literal_eval(pos_str), dtype=np.float64).reshape(-1)[:3]
            dim = np.asarray(ast.literal_eval(dim_str), dtype=np.float64).reshape(-1)[:3]
        except Exception:
            continue
        if pos.shape[0] != 3 or dim.shape[0] != 3:
            continue
        try:
            Rm = np.asarray(ast.literal_eval(rot_str), dtype=np.float64).reshape(3, 3) if rot_str else np.eye(3)
        except Exception:
            Rm = np.eye(3, dtype=np.float64)

        label_canonical, label_source_used = resolve_label_with_learning(
            raw_label,
            canonical=canonical,
            alias_map=alias_map,
            contains_rules=contains_rules,
            learned_alias_map=learned_alias_map,
            use_embedding_for_unresolved=use_embedding_for_unresolved,
            learn_new_aliases=learn_new_aliases,
            learned_alias_path=learned_alias_path,
            embedding_min_score=embedding_min_score,
        )

        T_obj = np.eye(4, dtype=np.float64)
        T_obj[:3, :3] = Rm
        T_obj[:3, 3] = pos
        hx, hy, hz = (dim / 2.0).tolist()
        corners = np.array(
            [
                [-hx, -hy, -hz],
                [-hx, -hy, hz],
                [-hx, hy, -hz],
                [-hx, hy, hz],
                [hx, -hy, -hz],
                [hx, -hy, hz],
                [hx, hy, -hz],
                [hx, hy, hz],
            ],
            dtype=np.float64,
        )
        box = _boxes_from_vertices(corners, T_norm @ T_obj)
        out.append(
            {
                "id": data.get("obj_id", node_id),
                "data_path": graphml_file,
                "job_id": None,
                "position": pos.tolist(),
                "orientation": None,
                "label_raw": raw_label,
                "slot_label_raw": raw_label,
                "retrieved_label_raw": None,
                "label_canonical": label_canonical,
                "slot_label_canonical": label_canonical,
                "retrieved_label_canonical": None,
                "retrieval_swapped": False,
                "label_resolution": label_source_used,
                **box,
            }
        )
    return out


def load_supervisely_gt(
    gt_json: str,
    canonical: set,
    alias_map: Dict[str, str],
    contains_rules: Dict[str, str],
    learned_alias_map: Dict[str, str],
    use_embedding_for_unresolved: bool,
    learn_new_aliases: bool,
    learned_alias_path: str,
    embedding_min_score: float,
    normalize_yaw_rad: float = 0.0,
) -> List[Dict]:
    with open(gt_json) as f:
        data = json.load(f)
    T_norm = _yaw_normalization_matrix(normalize_yaw_rad)

    id_to_label = {obj["id"]: obj.get("classTitle", "") for obj in data.get("objects", [])}
    out = []
    for fig in data.get("figures", []):
        if fig.get("geometryType") not in (None, "cuboid_3d"):
            continue
        geom = fig.get("geometry", {})
        p = geom.get("position", {})
        d = geom.get("dimensions", {})
        pos = np.array([p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0)], dtype=np.float64)
        dim = np.array([d.get("x", 0.0), d.get("y", 0.0), d.get("z", 0.0)], dtype=np.float64)
        rot = geom.get("rotation", {})
        rot_xyz = np.array(
            [rot.get("x", 0.0), rot.get("y", 0.0), rot.get("z", 0.0)],
            dtype=np.float64,
        )
        Rm = R.from_euler("xyz", rot_xyz, degrees=False).as_matrix()
        T_obj = np.eye(4, dtype=np.float64)
        T_obj[:3, :3] = Rm
        T_obj[:3, 3] = pos
        # 8 local corners of cuboid centered at origin.
        hx, hy, hz = (dim / 2.0).tolist()
        corners = np.array(
            [
                [-hx, -hy, -hz],
                [-hx, -hy, hz],
                [-hx, hy, -hz],
                [-hx, hy, hz],
                [hx, -hy, -hz],
                [hx, -hy, hz],
                [hx, hy, -hz],
                [hx, hy, hz],
            ],
            dtype=np.float64,
        )
        box = _boxes_from_vertices(corners, T_norm @ T_obj)
        raw_label = id_to_label.get(fig.get("objectId"), "")
        label_canonical, label_source_used = resolve_label_with_learning(
            raw_label,
            canonical=canonical,
            alias_map=alias_map,
            contains_rules=contains_rules,
            learned_alias_map=learned_alias_map,
            use_embedding_for_unresolved=use_embedding_for_unresolved,
            learn_new_aliases=learn_new_aliases,
            learned_alias_path=learned_alias_path,
            embedding_min_score=embedding_min_score,
        )
        out.append(
            {
                "id": fig.get("id"),
                "label_raw": raw_label,
                "label_canonical": label_canonical,
                "label_resolution": label_source_used,
                **box,
            }
        )
    return out
