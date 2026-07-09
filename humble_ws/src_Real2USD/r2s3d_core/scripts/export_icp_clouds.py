"""Export the ICP source/target point clouds (world frame) for inspection, to check
whether the ICP target actually looks like the object geometry.

Current baseline uses a SINGLE-VIEW masked depth cloud as the ICP target (partial). This
script dumps, for the FULL-frame condition:
  - icp_target_singleview.ply : the target ICP actually uses (best view only)   [GREEN]
  - icp_target_accumulated.ply: same objects, depth fused over ALL views         [CYAN]
  - icp_source_posed.ply      : SAM3D posed-mesh surface pts, before ICP          [RED]
  - icp_source_afterICP.ply   : SAM3D posed-mesh surface pts, after ICP           [BLUE]
Load together with results/ablation_full_vs_crop/scene_gt.glb (GT meshes, grey) in a
viewer (Blender / MeshLab / CloudCompare) — targets should sit ON the GT surfaces.

  uv run python scripts/export_icp_clouds.py [scene]
"""
import sys, os
import numpy as np
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from r2s3d_core.data.registry import make_source
from r2s3d_core.eval import geometry as geo
from r2s3d_core.baselines.sam3d_layout import (
    select_best_view, render_instance_mask, run_sam3d, place_from_sam3d,
    refine_icp, _masked_depth_cloud, _default_queue,
)

SCENE = sys.argv[1] if len(sys.argv) > 1 else "room0"
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "ablation_full_vs_crop")


def save_ply(pts, color, path):
    if pts is None or len(pts) == 0:
        print("  (empty)", path); return
    pc = trimesh.PointCloud(np.asarray(pts), colors=np.tile(color, (len(pts), 1)))
    pc.export(path)
    print(f"  {len(pts):>7} pts -> {os.path.basename(path)}")


def main():
    src = make_source("replica", SCENE, stride=20)
    gt = src.gt()
    frames = list(src)
    queue = _default_queue()
    H, W = frames[0].depth.shape[:2]

    tgt_sv, tgt_acc, src_pre, src_post = [], [], [], []
    n_sv, n_acc = [], []
    for g in gt:
        vi = select_best_view(g, frames)
        if vi is None:
            continue
        frame = frames[vi]
        mask = render_instance_mask(g.mesh, frame)

        # FULL-frame SAM3D output (cached)
        res = run_sam3d(frame.rgb, mask, frame.depth, frame.K, (0, 0, W - 1, H - 1),
                        meta={"track_id": g.instance_id, "label": g.label,
                              "full_width": W, "full_height": H}, queue=queue)
        if res is None:
            continue
        mesh_raw, pose = res
        posed, T_wo, _ = place_from_sam3d(mesh_raw, pose["sam3d_scale"],
                                          pose["sam3d_rotation"], pose["sam3d_translation"],
                                          frame.T_world_cam)

        # single-view target (what ICP actually uses)
        t_sv = _masked_depth_cloud(frame, mask)
        # accumulated target: fuse masked depth over ALL frames the object is visible in
        acc = [t_sv]
        for fr in frames:
            if fr is frame:
                continue
            m = render_instance_mask(g.mesh, fr)
            if m is not None and (m > 0).sum() > 50:
                acc.append(_masked_depth_cloud(fr, m))
        t_acc = np.concatenate(acc, 0)

        # source before/after ICP
        s_pre = geo.sample_surface(posed, 2000, seed=g.instance_id) if len(posed.faces) else np.asarray(posed.vertices)
        delta, info = refine_icp(s_pre, t_sv, np.eye(4))
        s_post = (delta[:3, :3] @ s_pre.T).T + delta[:3, 3]

        tgt_sv.append(t_sv); tgt_acc.append(t_acc); src_pre.append(s_pre); src_post.append(s_post)
        n_sv.append(len(t_sv)); n_acc.append(len(t_acc))

    os.makedirs(OUT, exist_ok=True)
    print("exporting point clouds (world frame):")
    save_ply(np.concatenate(tgt_sv), (60, 220, 60, 255), os.path.join(OUT, "icp_target_singleview.ply"))
    save_ply(np.concatenate(tgt_acc), (60, 220, 220, 255), os.path.join(OUT, "icp_target_accumulated.ply"))
    save_ply(np.concatenate(src_pre), (230, 60, 60, 255), os.path.join(OUT, "icp_source_posed.ply"))
    save_ply(np.concatenate(src_post), (60, 120, 240, 255), os.path.join(OUT, "icp_source_afterICP.ply"))
    print(f"\nICP target is SINGLE-VIEW (best frame only), NOT accumulated.")
    print(f"  mean pts/object: single-view {np.mean(n_sv):.0f}  vs  accumulated {np.mean(n_acc):.0f}"
          f"  ({np.mean(n_acc)/max(np.mean(n_sv),1):.1f}x more if fused)")


if __name__ == "__main__":
    main()
