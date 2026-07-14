"""Re-score / re-export a persisted run from its ``scene_graph.json`` -- no pipeline re-run.

Rebuilds predicted ``SceneObject``s from ``<result>/<scene>/scene_graph.json`` (poses + the
persisted SAM3D output GLBs), rebuilds GT from the source (optionally with real asset meshes
via ``--gt-mesh asset``), recomputes :func:`r2s3d_core.eval.metrics.evaluate`, and optionally
re-exports the pred/gt/overlay GLB. Use when the *metrics* evolve (Option A/B) or GT meshes
land (AI-8) but the *predictions* are unchanged -- turns an expensive campaign re-run into a
cheap re-eval. Predictions come straight off disk (round-trip: ``trimesh.load(<queue>/<mesh>)``
posed by ``T_world_mesh``), so no detector/SAM3D/registration is re-run.

Usage::

    python -m r2s3d_core.eval.rescore results/procthor_procthor_object_track_icp_s200_cached \\
        --source procthor --scene 200 --gt-mesh asset --export-glb
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np

from ..data.registry import make_source
from .metrics import SceneObject, evaluate, gt_to_scene_object


def load_scene_graph_preds(scene_graph_path, queue_root=None) -> List[SceneObject]:
    """Rebuild posed predicted ``SceneObject``s from a ``scene_graph.json``.

    Each object's mesh is ``trimesh.load(<queue>/<mesh>)`` transformed by ``T_world_mesh``
    (the round-trip the writer guarantees). ``queue_root`` overrides the queue dir; otherwise
    the file's ``sam3d_queue`` is tried both as-is (cwd-relative) and relative to the result
    dir. A mesh that fails to load leaves ``mesh=None`` (loudly) so geometry metrics skip it
    rather than crashing the whole re-score.
    """
    import trimesh

    path = Path(scene_graph_path)
    data = json.loads(path.read_text())
    queue_candidates = []
    if queue_root is not None:
        queue_candidates.append(Path(queue_root))
    q = data.get("sam3d_queue")
    if q:
        queue_candidates += [Path(q), path.parent.parent.parent / q, path.parent / q]

    def _resolve(rel: str) -> Optional[Path]:
        for base in queue_candidates:
            c = base / rel
            if c.exists():
                return c
        return None

    preds: List[SceneObject] = []
    n_missing_mesh = 0
    for o in data.get("objects", []):
        T = np.asarray(o["T_world_obj"], dtype=np.float64)
        ext = np.asarray(o["extents"], dtype=np.float64)
        mesh = None
        rel = o.get("mesh") or o.get("data_path")
        if rel:
            mpath = _resolve(rel)
            if mpath is None:
                n_missing_mesh += 1
                print(f"  WARNING: mesh not found for obj {o.get('id')}: {rel}")
            else:
                try:
                    m = trimesh.load(str(mpath), force="mesh", process=False)
                    m.apply_transform(np.asarray(o["T_world_mesh"], dtype=np.float64))
                    mesh = m
                except Exception as e:
                    n_missing_mesh += 1
                    print(f"  WARNING: failed to load/pose mesh for obj {o.get('id')}: {e}")
        preds.append(SceneObject(
            label=o.get("label", ""), T_world_obj=T, extents=ext, mesh=mesh,
            provenance={k: o.get(k) for k in ("id", "job_id", "registration") if k in o},
        ))
    if n_missing_mesh:
        print(f"  ({n_missing_mesh}/{len(preds)} predicted meshes unavailable -> geometry "
              f"metrics use the rest)")
    return preds


def rescore_scene(result_dir, source: str, scene: str, *, gt_mesh: str = "asset",
                  asset_root=None, iou_threshold: float = 0.25, compute_geometry: bool = True,
                  surface_points: int = 10000, export_glb: bool = False,
                  data_root=None, stride: int = 20) -> dict:
    """Re-evaluate one scene's persisted predictions against freshly-built GT."""
    result_dir = Path(result_dir)
    sg = result_dir / str(scene) / "scene_graph.json"
    if not sg.exists():
        raise FileNotFoundError(f"no scene_graph.json at {sg}")
    preds = load_scene_graph_preds(sg)

    src_kwargs = {"gt_mesh": gt_mesh}
    if asset_root is not None:
        src_kwargs["asset_root"] = asset_root
    src = make_source(source, scene, root=data_root, stride=stride, **src_kwargs)
    gt = src.gt() or []
    gts = [gt_to_scene_object(g) for g in gt]

    m = evaluate(preds, gts, iou_threshold=iou_threshold,
                 compute_geometry=compute_geometry, surface_points=surface_points)

    if export_glb:
        from ..recon.scene_glb import export_pred_vs_gt
        out = export_pred_vs_gt(preds, gts, result_dir / str(scene), lite=True)
        print(f"  re-exported GLB (GT mesh = {gt_mesh}) -> {out.get('compare_lite')}")

    if hasattr(src, "close"):
        try:
            src.close()
        except Exception:
            pass
    return m


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="re-score a persisted run from scene_graph.json")
    p.add_argument("result_dir", help="run dir containing <scene>/scene_graph.json")
    p.add_argument("--source", required=True, help="SequenceSource backend (e.g. procthor)")
    p.add_argument("--scene", nargs="+", required=True)
    p.add_argument("--gt-mesh", default="asset", choices=["box", "asset", "none"],
                   help="GT mesh policy for the rebuilt GT (default 'asset' = real meshes)")
    p.add_argument("--asset-root", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--stride", type=int, default=20)
    p.add_argument("--iou-threshold", type=float, default=0.25)
    p.add_argument("--no-geometry", action="store_true")
    p.add_argument("--surface-points", type=int, default=10000)
    p.add_argument("--export-glb", action="store_true",
                   help="re-export pred/gt/overlay GLB into <result>/<scene>/ (overwrites)")
    p.add_argument("--out", default=None, help="write the re-scored metrics to this json")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    per_scene = {}
    for scene in args.scene:
        print(f"[{scene}] re-scoring from scene_graph.json ...")
        m = rescore_scene(
            args.result_dir, args.source, scene, gt_mesh=args.gt_mesh,
            asset_root=args.asset_root, iou_threshold=args.iou_threshold,
            compute_geometry=not args.no_geometry, surface_points=args.surface_points,
            export_glb=args.export_glb, data_root=args.data_root, stride=args.stride)
        per_scene[scene] = m
        print(f"[{scene}] f1={m['f1']:.3f} micro_f1={m['micro_f1']:.3f} "
              f"macro_f1={m['macro_f1']:.3f} recall={m['recall']:.3f} "
              f"scene_chamfer={m.get('scene_chamfer_mean_m')} "
              f"geo_recall@5cm={m.get('geo_recall@0.05')}")
    if args.out:
        Path(args.out).write_text(json.dumps({"rescore": per_scene}, indent=2, default=float))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
