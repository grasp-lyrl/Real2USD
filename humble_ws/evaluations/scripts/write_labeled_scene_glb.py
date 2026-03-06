#!/usr/bin/env python3
"""
Write scene GLBs and labeled scene-graph JSONs that include only objects with resolved labels (same as overlay plot).

Uses the same label resolution as run_eval / plot_overlays: canonical_labels + aliases
(+ optional embedding for unresolved). Objects without a resolved label_canonical are
excluded. When --run-dir is set, GLBs also include the accumulated RealSense point cloud
from run_dir/diagnostics/pointclouds/realsense (or pointscloud/realsense).

Outputs (when using --run-dir):
  - scene_labeled.glb, scene_sam3d_only_labeled.glb  (meshes + optional RealSense point cloud)
  - scene_graph_labeled.json, scene_graph_sam3d_only_labeled.json  (objects with label/label_canonical)

Usage (from evaluations/ or with EVAL_ROOT on path):

  python scripts/write_labeled_scene_glb.py --run-dir /path/to/run_dir [--eval-config ./eval_config.json]

  # Optional: explicit paths
  python scripts/write_labeled_scene_glb.py --scene-graph /path/to/scene_graph.json \\
      --scene-graph-sam3d /path/to/scene_graph_sam3d_only.json \\
      --out-scene-glb /path/to/scene_labeled.glb --out-sam3d-glb /path/to/scene_sam3d_only_labeled.glb \\
      --out-scene-json /path/to/scene_graph_labeled.json --out-sam3d-json /path/to/scene_graph_sam3d_only_labeled.json \\
      --eval-config ./eval_config.json

PYTHONPATH=../src_Real2USD/real2sam3d python3 scripts/write_labeled_scene_glb.py \
  --run-dir /data/sam3d_queue/run_20260224_053223

Requires: real2sam3d on PYTHONPATH for GLB writing (e.g. PYTHONPATH=humble_ws/src_Real2USD/real2sam3d).
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Allow imports from parent (evaluations/)
_EVAL_ROOT = Path(__file__).resolve().parent.parent
if str(_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_EVAL_ROOT))

from eval_common import load_predictions_scene_graph
from label_matching import load_label_configs, load_learned_aliases


def _has_label(box: Dict[str, Any]) -> bool:
    """True if box has a non-empty label (label_canonical or label). Matches plot_overlays."""
    lbl = box.get("label_canonical") or box.get("label")
    return bool(lbl) and bool(str(lbl).strip())


def _load_eval_config(config_path: str, eval_dir: str, cwd: str = ".") -> Dict:
    p = Path(config_path)
    if not p.is_absolute():
        # Try cwd-relative first (e.g. evaluations/eval_config.json from repo root), then eval_dir-relative
        p_cwd = Path(cwd) / p
        if p_cwd.exists():
            p = p_cwd.resolve()
        else:
            p = (Path(eval_dir) / p).resolve()
    with open(p) as f:
        cfg = json.load(f)
    base = p.parent
    for key in ("canonical_labels", "label_aliases", "learned_label_aliases"):
        if key in cfg.get("paths", {}):
            rel = cfg["paths"][key]
            if not Path(rel).is_absolute():
                cfg["paths"][key] = str((base / rel).resolve())
    return cfg


def _scene_list_from_filtered_predictions(
    preds: List[Dict],
    raw_objects_by_id: Dict[str, Dict],
) -> List[Dict]:
    """Build list of scene-graph-style objects for write_joint_glb_from_list_standalone."""
    out = []
    for p in preds:
        oid = p.get("id")
        raw = raw_objects_by_id.get(str(oid)) or raw_objects_by_id.get(oid, p)
        entry = {
            "id": oid,
            "data_path": p.get("data_path") or raw.get("data_path"),
            "position": p.get("position") or raw.get("position"),
            "orientation": p.get("orientation") or raw.get("orientation"),
        }
        if raw.get("transform_odom_from_raw") is not None:
            entry["transform_odom_from_raw"] = raw["transform_odom_from_raw"]
        out.append(entry)
    return out


def _labeled_objects_for_json(
    preds: List[Dict],
    raw_objects_by_id: Dict[str, Dict],
) -> List[Dict]:
    """Build list of objects for scene_graph_labeled.json (includes label/label_canonical)."""
    out = []
    for p in preds:
        oid = p.get("id")
        raw = raw_objects_by_id.get(str(oid)) or raw_objects_by_id.get(oid, p)
        entry = {
            "id": oid,
            "data_path": p.get("data_path") or raw.get("data_path"),
            "position": p.get("position") or raw.get("position"),
            "orientation": p.get("orientation") or raw.get("orientation"),
        }
        if raw.get("transform_odom_from_raw") is not None:
            entry["transform_odom_from_raw"] = raw["transform_odom_from_raw"]
        # Write "label" so run_eval's load_predictions_scene_graph has label_raw to resolve (it uses obj.get("label")).
        # Preds from load_predictions_scene_graph use "label_raw", not "label".
        label_for_json = p.get("label_raw") or p.get("label") or p.get("label_canonical")
        if label_for_json is not None and str(label_for_json).strip():
            entry["label"] = label_for_json
        if p.get("label_canonical") is not None:
            entry["label_canonical"] = p["label_canonical"]
        out.append(entry)
    return out


def write_labeled_scene_glb(
    scene_graph_path: Path,
    out_glb_path: Path,
    config: Dict,
    eval_root: Path,
    label_source: str = "prefer_retrieved_else_label",
    normalize_yaw_rad: float = 0.0,
    run_dir: Optional[Path] = None,
    out_json_path: Optional[Path] = None,
    realsense_voxel_size: Optional[float] = 0.02,
    realsense_use_all_files: bool = False,
    realsense_z_offset: float = 0.0,
    log=None,
    skip_glb: bool = False,
) -> int:
    """
    Load scene_graph, resolve labels, filter to resolved-only, write GLB (and optional labeled JSON).
    When skip_glb=True, only the labeled JSON is written (no GLB); use e.g. with --no-scene-glb to get
    scene_graph_labeled.json for run_eval without writing the heavy GLB.
    When run_dir is set, the GLB includes accumulated RealSense point cloud from run_dir (voxel-downsampled by default for speed).
    realsense_use_all_files: if True, merge all realsense_*.npy files; if False, use the single file with most points.
    realsense_z_offset: translation in meters applied to point cloud z (e.g. -0.2 to move down by 0.2). Default 0.
    Returns number of objects included.
    """
    if log:
        log.info("[1/5] Loading scene graph: %s" % scene_graph_path)
    paths = config.get("paths", {})
    canonical_path = paths.get("canonical_labels")
    aliases_path = paths.get("label_aliases")
    learned_path = paths.get("learned_label_aliases")
    if not canonical_path or not aliases_path:
        raise ValueError("eval config paths must include canonical_labels and label_aliases")
    if log:
        log.info("[2/5] Loading label config (canonical + aliases)...")
    canonical, alias_map, contains_rules = load_label_configs(canonical_path, aliases_path)
    learned_alias_map = load_learned_aliases(learned_path or "")
    label_cfg = config.get("label_resolution", {}) or {}
    use_embedding = bool(label_cfg.get("use_embedding_for_unresolved", False))
    learn_new = bool(label_cfg.get("learn_new_aliases", False))
    embedding_min = float(label_cfg.get("embedding_min_score", 0.5))

    with open(scene_graph_path) as f:
        raw_data = json.load(f)
    raw_objs = raw_data.get("objects", [])
    raw_by_id = {str(o.get("id", i)): o for i, o in enumerate(raw_objs)}
    if log:
        log.info("      Scene graph has %d objects." % len(raw_objs))

    if log:
        log.info("[3/5] Resolving labels (canonical + aliases)...")
    scene_base_dir = str(scene_graph_path.resolve().parent)
    preds = load_predictions_scene_graph(
        str(scene_graph_path),
        canonical=canonical,
        alias_map=alias_map,
        contains_rules=contains_rules,
        learned_alias_map=learned_alias_map,
        use_embedding_for_unresolved=use_embedding,
        learn_new_aliases=learn_new,
        learned_alias_path=learned_path or "",
        embedding_min_score=embedding_min,
        label_source=label_source,
        normalize_yaw_rad=normalize_yaw_rad,
        base_dir=scene_base_dir,
    )
    filtered = [p for p in preds if _has_label(p)]
    if log:
        log.info("      %d objects with resolved labels (of %d total)." % (len(filtered), len(preds)))
    scene_list = _scene_list_from_filtered_predictions(filtered, raw_by_id)
    if not scene_list:
        if log:
            log.warn("No objects with resolved labels; skipping GLB %s" % out_glb_path)
        return 0

    if out_json_path is not None:
        if log:
            log.info("[4/5] Writing labeled JSON: %s" % out_json_path)
        labeled_objs = _labeled_objects_for_json(filtered, raw_by_id)
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json_path, "w") as f:
            json.dump({"objects": labeled_objs}, f, indent=2)
        if log:
            log.info("      Wrote %s (%d objects)" % (out_json_path, len(labeled_objs)))
    else:
        if log:
            log.info("[4/5] Skipping labeled JSON (no path).")

    if skip_glb:
        if log:
            log.info("[5/5] Skipping GLB (skip_glb=True).")
        return len(scene_list)

    try:
        from real2sam3d.glb_export_standalone import write_joint_glb_from_list_standalone
    except ImportError as e:
        raise ImportError(
            "real2sam3d is required for GLB writing (no ROS needed). "
            "Set PYTHONPATH to include real2sam3d package root, e.g. PYTHONPATH=src_Real2USD/real2sam3d python ..."
        ) from e

    if log:
        log.info("[5/5] Writing GLB: %s (%d meshes%s) — may take a while for large scenes or with point cloud."
                % (out_glb_path, len(scene_list), "; including RealSense point cloud" if run_dir else ""))
    out_glb_path.parent.mkdir(parents=True, exist_ok=True)
    sig = inspect.signature(write_joint_glb_from_list_standalone)
    kwargs = {"run_dir": run_dir, "log": log}
    if "realsense_voxel_size" in sig.parameters:
        kwargs["realsense_voxel_size"] = realsense_voxel_size
    if "realsense_use_all_files" in sig.parameters:
        kwargs["realsense_use_all_files"] = realsense_use_all_files
    if "realsense_z_offset" in sig.parameters:
        kwargs["realsense_z_offset"] = realsense_z_offset
    write_joint_glb_from_list_standalone(out_glb_path, scene_list, **kwargs)
    if log:
        log.info("      Done: %s" % out_glb_path)
    return len(scene_list)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write scene.glb and scene_sam3d_only.glb containing only objects with resolved labels (same as overlay)."
    )
    parser.add_argument("--run-dir", type=Path, default=None, help="Run directory (default: required unless all paths given)")
    parser.add_argument("--scene-graph", type=Path, default=None, help="Input scene_graph.json (default: run_dir/scene_graph.json)")
    parser.add_argument("--scene-graph-sam3d", type=Path, default=None, help="Input scene_graph_sam3d_only.json (default: run_dir/scene_graph_sam3d_only.json)")
    parser.add_argument("--out-scene-glb", type=Path, default=None, help="Output scene GLB (default: run_dir/scene_labeled.glb)")
    parser.add_argument("--out-sam3d-glb", type=Path, default=None, help="Output SAM3D-only GLB (default: run_dir/scene_sam3d_only_labeled.glb)")
    parser.add_argument("--out-scene-json", type=Path, default=None, help="Output scene labeled JSON (default: run_dir/scene_graph_labeled.json)")
    parser.add_argument("--out-sam3d-json", type=Path, default=None, help="Output SAM3D-only labeled JSON (default: run_dir/scene_graph_sam3d_only_labeled.json)")
    parser.add_argument("--eval-config", default=None, help="Eval config JSON (default: EVAL_ROOT/eval_config.json)")
    parser.add_argument("--realsense-voxel-size", type=float, default=0.02, metavar="M",
                        help="Voxel size (m) for downsampling RealSense point cloud in GLB (default 0.02 = 2cm; much faster). Set 0 to disable.")
    parser.add_argument("--no-realsense-downsample", action="store_true",
                        help="Do not downsample RealSense point cloud (slower GLB export for large runs)")
    parser.add_argument("--realsense-use-all-files", action="store_true",
                        help="Merge all realsense_*.npy files (default: use single file with most points)")
    parser.add_argument("--realsense-z-offset", type=float, default=-0.2, metavar="M",
                        help="Z translation in meters for RealSense point cloud (default: -0.2 = move down 0.2m)")
    parser.add_argument("--no-scene-glb", action="store_true", help="Do not write scene_labeled.glb")
    parser.add_argument("--no-sam3d-glb", action="store_true", help="Do not write scene_sam3d_only_labeled.glb")
    args = parser.parse_args()

    eval_root = _EVAL_ROOT
    config_path = args.eval_config or str(eval_root / "eval_config.json")
    print("Loading eval config: %s" % config_path)
    config = _load_eval_config(config_path, str(eval_root), cwd=os.getcwd())
    label_source = config.get("prediction", {}).get("label_source", "prefer_retrieved_else_label")
    print("Label source: %s" % label_source)

    run_dir = args.run_dir
    if run_dir is None and (args.scene_graph is None or args.out_scene_glb is None):
        parser.error("Provide --run-dir or all of --scene-graph, --out-scene-glb (and optionally --scene-graph-sam3d, --out-sam3d-glb)")
    if run_dir is not None:
        run_dir = run_dir.resolve()
    scene_graph = args.scene_graph or (run_dir / "scene_graph.json" if run_dir else None)
    scene_graph_sam3d = args.scene_graph_sam3d or (run_dir / "scene_graph_sam3d_only.json" if run_dir else None)
    out_scene_glb = args.out_scene_glb or (run_dir / "scene_labeled.glb" if run_dir else None)
    out_sam3d_glb = args.out_sam3d_glb or (run_dir / "scene_sam3d_only_labeled.glb" if run_dir else None)
    out_scene_json = args.out_scene_json or (run_dir / "scene_graph_labeled.json" if run_dir else None)
    out_sam3d_json = args.out_sam3d_json or (run_dir / "scene_graph_sam3d_only_labeled.json" if run_dir else None)
    realsense_voxel_size = None if args.no_realsense_downsample else args.realsense_voxel_size
    if realsense_voxel_size is not None and realsense_voxel_size <= 0:
        realsense_voxel_size = None
    realsense_use_all_files = args.realsense_use_all_files
    realsense_z_offset = getattr(args, "realsense_z_offset", 0.0)

    class Log:
        def info(self, msg): print(msg)
        def warn(self, msg): print(msg, file=sys.stderr)
    log = Log()

    if not scene_graph or not scene_graph.exists():
        print("Error: scene_graph not found: %s" % scene_graph, file=sys.stderr)
        sys.exit(1)

    # Scene (registration): always run so we get scene_graph_labeled.json; skip GLB only when --no-scene-glb
    if scene_graph and (out_scene_glb is not None or out_scene_json is not None):
        print("\n--- Scene (registration) labeled output ---")
        n = write_labeled_scene_glb(
            scene_graph,
            out_scene_glb or (run_dir / "scene_labeled.glb" if run_dir else Path("scene_labeled.glb")),
            config,
            eval_root,
            label_source=label_source,
            normalize_yaw_rad=0.0,
            run_dir=run_dir,
            out_json_path=out_scene_json,
            realsense_voxel_size=realsense_voxel_size,
            realsense_use_all_files=realsense_use_all_files,
            realsense_z_offset=realsense_z_offset,
            log=log,
            skip_glb=args.no_scene_glb,
        )
        if not args.no_scene_glb and out_scene_glb is not None:
            print("Wrote %s (%d objects with resolved labels)" % (out_scene_glb, n))
        if out_scene_json is not None:
            print("Wrote %s (%d objects)" % (out_scene_json, n))
        if args.no_scene_glb:
            print("Skipped scene_labeled.glb (--no-scene-glb).")

    if not args.no_sam3d_glb and scene_graph_sam3d is not None and scene_graph_sam3d.exists() and out_sam3d_glb is not None:
        print("\n--- SAM3D-only labeled output ---")
        n2 = write_labeled_scene_glb(
            scene_graph_sam3d,
            out_sam3d_glb,
            config,
            eval_root,
            label_source=label_source,
            normalize_yaw_rad=0.0,
            run_dir=run_dir,
            out_json_path=out_sam3d_json,
            realsense_voxel_size=realsense_voxel_size,
            realsense_use_all_files=realsense_use_all_files,
            realsense_z_offset=realsense_z_offset,
            log=log,
        )
        print("Wrote %s (%d objects with resolved labels)" % (out_sam3d_glb, n2))
        if out_sam3d_json is not None:
            print("Wrote %s" % out_sam3d_json)
    elif args.no_sam3d_glb:
        pass
    elif scene_graph_sam3d is None or not scene_graph_sam3d.exists():
        print("Skipping SAM3D-only GLB (file not found: %s)" % scene_graph_sam3d, file=sys.stderr)

    print("\nDone.")


if __name__ == "__main__":
    main()
