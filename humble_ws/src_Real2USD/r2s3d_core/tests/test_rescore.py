"""Round-trip test for loading predictions back from a scene_graph.json (no pipeline)."""

import json
from pathlib import Path

import numpy as np
import pytest

from r2s3d_core.eval import rescore


def test_load_scene_graph_preds_roundtrip(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    # a persisted SAM3D output GLB (unit box at the origin in mesh-local frame)
    queue = tmp_path / "sam3d_queue"
    job = queue / "output" / "job0"
    job.mkdir(parents=True)
    trimesh.creation.box(extents=[1, 1, 1]).export(job / "object.glb")

    # T_world_mesh translates the mesh to (5, 6, 7); T_world_obj/extents describe the OBB
    T_world_mesh = np.eye(4); T_world_mesh[:3, 3] = [5, 6, 7]
    T_world_obj = np.eye(4); T_world_obj[:3, 3] = [5, 6, 7]
    sg = {
        "sam3d_queue": str(queue),
        "objects": [{
            "id": 0, "label": "chair", "job_id": "job0",
            "T_world_obj": T_world_obj.tolist(), "extents": [1, 1, 1],
            "T_world_mesh": T_world_mesh.tolist(), "mesh": "output/job0/object.glb",
        }],
    }
    sg_path = tmp_path / "200" / "scene_graph.json"
    sg_path.parent.mkdir()
    sg_path.write_text(json.dumps(sg))

    preds = rescore.load_scene_graph_preds(sg_path)
    assert len(preds) == 1
    p = preds[0]
    assert p.label == "chair"
    assert p.mesh is not None
    # mesh posed by T_world_mesh -> centroid at (5,6,7)
    np.testing.assert_allclose(p.mesh.vertices.mean(axis=0), [5, 6, 7], atol=1e-6)


def test_load_scene_graph_preds_missing_mesh_is_none(tmp_path):
    sg = {
        "sam3d_queue": str(tmp_path / "nope"),
        "objects": [{
            "id": 0, "label": "table",
            "T_world_obj": np.eye(4).tolist(), "extents": [1, 1, 1],
            "T_world_mesh": np.eye(4).tolist(), "mesh": "output/missing/object.glb",
        }],
    }
    sg_path = tmp_path / "scene_graph.json"
    sg_path.write_text(json.dumps(sg))
    preds = rescore.load_scene_graph_preds(sg_path)
    assert len(preds) == 1 and preds[0].mesh is None  # missing mesh -> None, not a crash
