"""
Refine scene_graph.json by deduplicating objects using label similarity (CLIP or exact)
and 3D overlap (IoU from mesh AABBs or position-distance fallback).

Combines ideas from:
  - demo_reduce_scene_graph.py (sam-3d-objects): CLIP label similarity + 3D IoU grouping.
  - usd_buffer_node: mesh-based bounds (GLB/PLY load, transform by pose) and label handling.

Input: scene_graph.json with {"objects": [{"id", "data_path", "position", "orientation", "label", ...}]}.
Output: scene_graph_reduced.json (same schema, fewer objects). Representative per group is chosen
  by lowest id (stable) or by optional confidence/score if present.

Usage (from repo root or humble_ws with real2sam3d on PYTHONPATH):

  # Full: CLIP + mesh-based IoU (needs torch, transformers, trimesh)
  python3 -m real2sam3d.refine_scene_graph --input /path/to/run_dir/scene_graph.json \\
      --output /path/to/run_dir/scene_graph_reduced.json --iou-threshold 0.2 --clip-threshold 0.8

  # No CLIP: exact label match only (faster, no GPU)
  python3 -m real2sam3d.refine_scene_graph --input scene_graph.json --output scene_graph_reduced.json \\
      --no-clip --iou-threshold 0.2

  # Position-only overlap (no mesh load; use when data_path not available or for speed)
  python3 -m real2sam3d.refine_scene_graph --input scene_graph.json --output scene_graph_reduced.json \\
      --position-only --position-radius 0.5 --clip-threshold 0.85

Suggested path:
  1. Produce scene_graph.json (pipeline or scripts_offline_scene_buffer).
  2. Run this script with CLIP + mesh IoU for best quality; use --report-overlaps to tune thresholds.
  3. Use scene_graph_reduced.json for evaluation, USDA export, or downstream tools.
  4. Optional: keep scene_graph.json as the full slot list and use reduced only where dedup matters.
"""

from __future__ import annotations

import argparse
import json
import re
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np

# Optional CLIP
try:
    import torch
    from transformers import CLIPModel, CLIPTokenizer
    _CLIP_AVAILABLE = True
except ImportError:
    _CLIP_AVAILABLE = False

# Optional trimesh for mesh AABB
try:
    import trimesh
    _TRIMESH_AVAILABLE = True
except ImportError:
    _TRIMESH_AVAILABLE = False


DEVICE = "cuda" if (_CLIP_AVAILABLE and torch.cuda.is_available()) else "cpu"
CLIP_MODEL = None
CLIP_TOKENIZER = None
MODEL_NAME = "openai/clip-vit-base-patch32"


def initialize_clip_model():
    global CLIP_MODEL, CLIP_TOKENIZER
    if not _CLIP_AVAILABLE:
        return False
    try:
        CLIP_MODEL = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
        CLIP_TOKENIZER = CLIPTokenizer.from_pretrained(MODEL_NAME)
        return True
    except Exception as e:
        print(f"CLIP load failed: {e}")
        return False


def clean_label(label: str) -> str:
    """Normalize label: lowercase, strip; remove trailing _1, _2, _A, etc."""
    if not label:
        return ""
    label = label.lower().strip()
    label = re.sub(r"(_|\s)\d+$", "", label)
    label = re.sub(r"(_|\s)[a-zA-Z]$", "", label)
    return label.strip()


def get_text_embedding(text: str) -> "np.ndarray":
    if not _CLIP_AVAILABLE or CLIP_MODEL is None:
        raise RuntimeError("CLIP not initialized")
    cleaned = clean_label(text)
    inputs = CLIP_TOKENIZER([cleaned], padding=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        feats = CLIP_MODEL.get_text_features(**inputs)
    feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
    return feats.cpu().numpy().squeeze(0)


def are_labels_similar_clip(label1: str, label2: str, threshold: float) -> bool:
    if not _CLIP_AVAILABLE or CLIP_MODEL is None:
        return label1.lower().strip() == label2.lower().strip()
    e1 = get_text_embedding(label1)
    e2 = get_text_embedding(label2)
    sim = float(np.dot(e1, e2))
    return sim >= threshold


def load_mesh_local_bounds(data_path: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Load mesh from GLB/PLY and return (local_center, local_extents). Returns None if unavailable."""
    if not _TRIMESH_AVAILABLE:
        return None
    path = Path(data_path)
    if not path.exists():
        return None
    suf = path.suffix.lower()
    if suf not in (".glb", ".ply"):
        return None
    try:
        scene = trimesh.load(str(path), process=False)
        if isinstance(scene, trimesh.Scene):
            meshes = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
        elif isinstance(scene, trimesh.Trimesh):
            meshes = [scene]
        else:
            meshes = []
        if not meshes:
            return None
        verts = np.vstack([np.asarray(m.vertices, dtype=np.float64) for m in meshes if len(m.vertices) > 0])
        if len(verts) == 0:
            return None
        min_b = np.min(verts, axis=0)
        max_b = np.max(verts, axis=0)
        center = (min_b + max_b) / 2.0
        extents = (max_b - min_b) / 2.0
        return center, extents
    except Exception:
        return None


def quat_xyzw_from_orientation(orientation: list[float]) -> np.ndarray:
    """Our scene_graph may store [w,x,y,z]. Convert to (x,y,z,w) for scipy."""
    o = np.asarray(orientation, dtype=np.float64)
    if len(o) != 4:
        return np.array([0.0, 0.0, 0.0, 1.0])
    # If first component is close to 1 and others small, likely WXYZ
    if abs(o[0]) > 0.9 and np.max(np.abs(o[1:])) < 0.5:
        return np.array([o[1], o[2], o[3], o[0]])
    return o


def aabb_from_pose(
    position: list[float] | np.ndarray,
    orientation: list[float] | np.ndarray,
    local_center: np.ndarray,
    local_extents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """World-axis-aligned bounding box from pose (position + quat) and local box (center, extents)."""
    try:
        from scipy.spatial.transform import Rotation as R
    except ImportError:
        raise ImportError("scipy is required for mesh-based AABB; use --position-only to skip.")

    pos = np.asarray(position, dtype=np.float64).ravel()[:3]
    q = quat_xyzw_from_orientation(orientation)
    R_obj = R.from_quat(q)
    rot = R_obj.as_matrix()
    corners_local = np.array(
        [
            local_center + np.array([sx * local_extents[0], sy * local_extents[1], sz * local_extents[2]])
            for sx in (-1, 1)
            for sy in (-1, 1)
            for sz in (-1, 1)
        ]
    )
    corners_world = (rot @ corners_local.T).T + pos
    min_b = np.min(corners_world, axis=0)
    max_b = np.max(corners_world, axis=0)
    return min_b, max_b


def position_box_aabb(position: list[float] | np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """AABB as a cube around position with half-side radius."""
    p = np.asarray(position, dtype=np.float64).ravel()[:3]
    r = float(radius)
    return p - r, p + r


def calculate_3d_iou(
    bbox1_bounds: tuple[np.ndarray, np.ndarray],
    bbox2_bounds: tuple[np.ndarray, np.ndarray],
) -> float:
    """3D IoU of two AABBs: (min, max) each."""
    min_a = np.asarray(bbox1_bounds[0])
    max_a = np.asarray(bbox1_bounds[1])
    min_b = np.asarray(bbox2_bounds[0])
    max_b = np.asarray(bbox2_bounds[1])
    inter_min = np.maximum(min_a, min_b)
    inter_max = np.minimum(max_a, max_b)
    inter_dims = np.maximum(0, inter_max - inter_min)
    inter_vol = np.prod(inter_dims)
    vol_a = np.prod(max_a - min_a)
    vol_b = np.prod(max_b - min_b)
    if vol_a <= 1e-9 and vol_b <= 1e-9:
        return 1.0
    if vol_a <= 1e-9 or vol_b <= 1e-9:
        return 0.0
    union = vol_a + vol_b - inter_vol
    if union <= 0:
        return 0.0
    return float(inter_vol / union)


def build_object_list_with_bounds(
    objects: list[dict[str, Any]],
    use_mesh: bool,
    position_radius: float,
) -> list[dict[str, Any]]:
    """Attach aabb_bounds to each object. use_mesh=True: load from data_path; else position box."""
    out = []
    for obj in objects:
        pos = obj.get("position")
        ori = obj.get("orientation", [0.0, 0.0, 0.0, 1.0])
        if not pos or len(pos) < 3:
            continue
        bounds = None
        if use_mesh:
            dp = obj.get("data_path")
            if dp:
                local = load_mesh_local_bounds(dp)
                if local is not None:
                    center, extents = local
                    bounds = aabb_from_pose(pos, ori, center, extents)
        if bounds is None:
            bounds = position_box_aabb(pos, position_radius)
        rec = {**obj, "aabb_bounds": [bounds[0].tolist(), bounds[1].tolist()]}
        out.append(rec)
    return out


def deduplicate_scene_graph(
    objects_with_bounds: list[dict[str, Any]],
    iou_threshold: float,
    label_similar_fn: Callable[[str, str], bool],
    pick_representative: Callable[[list[dict]], dict] | None = None,
) -> list[dict[str, Any]]:
    """
    Group objects by label similarity + 3D IoU; emit one representative per group.
    objects_with_bounds must have "aabb_bounds", "label" (or use "id" as fallback), and full object payload.
    """
    if pick_representative is None:
        def pick_representative(group):
            return min(group, key=lambda o: (o.get("id"), str(o.get("id", ""))))

    groups: list[list[dict]] = []
    for obj in objects_with_bounds:
        label = obj.get("label") or str(obj.get("id", ""))
        bounds = obj["aabb_bounds"]
        bounds_t = (np.array(bounds[0]), np.array(bounds[1]))
        merged = False
        for group in groups:
            rep = group[0]
            rep_label = rep.get("label") or str(rep.get("id", ""))
            if not label_similar_fn(label, rep_label):
                continue
            rep_bounds = (np.array(rep["aabb_bounds"][0]), np.array(rep["aabb_bounds"][1]))
            iou = calculate_3d_iou(bounds_t, rep_bounds)
            if iou >= iou_threshold:
                group.append(obj)
                merged = True
                break
        if not merged:
            groups.append([obj])

    result = []
    for group in groups:
        rep = pick_representative(group)
        # Remove internal key and optionally record merged ids
        out = {k: v for k, v in rep.items() if k != "aabb_bounds"}
        merged_ids = [o.get("id") for o in group if o.get("id") != rep.get("id")]
        if merged_ids:
            out["merged_from_ids"] = merged_ids
        result.append(out)
    return result


def check_overlaps(objects: list[dict], iou_threshold: float) -> list[dict]:
    """Report pairs with IoU > threshold (for diagnostics). objects must have aabb_bounds."""
    overlaps = []
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            a = objects[i]["aabb_bounds"]
            b = objects[j]["aabb_bounds"]
            iou = calculate_3d_iou((np.array(a[0]), np.array(a[1])), (np.array(b[0]), np.array(b[1])))
            if iou > iou_threshold:
                overlaps.append({"id1": objects[i].get("id"), "id2": objects[j].get("id"), "iou": iou})
    return overlaps


def main():
    parser = argparse.ArgumentParser(
        description="Refine scene_graph.json: deduplicate by label similarity (CLIP or exact) + 3D IoU."
    )
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input scene_graph.json")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output scene_graph_reduced.json")
    parser.add_argument("--iou-threshold", type=float, default=0.2, help="IoU threshold for merging (default 0.2)")
    parser.add_argument("--clip-threshold", type=float, default=0.8, help="CLIP similarity threshold (default 0.8)")
    parser.add_argument("--no-clip", action="store_true", help="Use exact label match only (no CLIP)")
    parser.add_argument(
        "--position-only",
        action="store_true",
        help="Use position-radius box instead of loading meshes for AABB",
    )
    parser.add_argument(
        "--position-radius",
        type=float,
        default=0.5,
        help="Half-side of position box in meters when --position-only (default 0.5)",
    )
    parser.add_argument(
        "--report-overlaps",
        action="store_true",
        help="Print remaining overlaps in reduced graph (for tuning)",
    )
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)
    objects = data.get("objects", [])
    if not objects:
        print("No objects in input.")
        return

    use_clip = not args.no_clip
    if use_clip and _CLIP_AVAILABLE:
        if not initialize_clip_model():
            print("Falling back to exact label match (CLIP failed).")
            use_clip = False
    elif use_clip and not _CLIP_AVAILABLE:
        print("CLIP not available (torch/transformers). Using exact label match.")
        use_clip = False

    def label_similar(l1: str, l2: str) -> bool:
        if use_clip:
            return are_labels_similar_clip(l1 or "", l2 or "", args.clip_threshold)
        return (l1 or "").lower().strip() == (l2 or "").lower().strip()

    use_mesh = not args.position_only and _TRIMESH_AVAILABLE
    if args.position_only:
        print("Using position-only AABB (radius=%.2f m)." % args.position_radius)
    elif not _TRIMESH_AVAILABLE:
        print("trimesh not available; using position-only AABB.")
        use_mesh = False

    objects_with_bounds = build_object_list_with_bounds(
        objects, use_mesh=use_mesh, position_radius=args.position_radius
    )
    if len(objects_with_bounds) < len(objects):
        print("Skipped %d objects (missing position or bounds)." % (len(objects) - len(objects_with_bounds)))

    refined = deduplicate_scene_graph(objects_with_bounds, args.iou_threshold, label_similar)
    print("Refined %d -> %d objects (iou_thr=%.2f, clip=%s)." % (len(objects), len(refined), args.iou_threshold, use_clip))

    out_data = {"objects": refined}
    if data.get("source"):
        out_data["source"] = data["source"]
    out_data["refined_from"] = str(args.input)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out_data, f, indent=2)
    print("Wrote %s" % args.output)

    if args.report_overlaps:
        # Rebuild bounds for refined list for overlap check (position-only style to avoid re-loading)
        refined_with_bounds = build_object_list_with_bounds(refined, use_mesh=False, position_radius=args.position_radius)
        overlaps = check_overlaps(refined_with_bounds, args.iou_threshold)
        if overlaps:
            print("Remaining overlaps (IoU > %.2f): %d" % (args.iou_threshold, len(overlaps)))
            for o in overlaps[:20]:
                print("  %s vs %s IoU %.3f" % (o["id1"], o["id2"], o["iou"]))
        else:
            print("No remaining overlaps above IoU threshold.")


if __name__ == "__main__":
    main()
