"""ProcTHOR render-cache replay contract (no ai2thor needed).

AI2-THOR RGB is non-deterministic across renders, which reshuffles appearance-based
re-ID and thus the tracker's sequential track_ids between the SAM3D queue and collect
passes. ProcThorSource caches the full render once and replays it deterministically.

These tests build a synthetic cache directory by hand (a couple of tiny frames + the
meta) and check that a cache-backed source replays frames, GT, and native masks WITHOUT
starting a controller — so the serialization contract can't silently drift.
"""

import json

import numpy as np
import trimesh

from r2s3d_core.data.procthor import ProcThorSource


def _write_cache(tmp_path, n=3, hw=(4, 5)):
    """Hand-build a minimal render cache and return (dir, rgb_list, seg color)."""
    H, W = hw
    base = tmp_path / "cache"
    (base / "frames").mkdir(parents=True)
    color = [10, 20, 30]  # instance 0's seg color
    rgbs = []
    for fid in range(n):
        rgb = np.full((H, W, 3), fid, np.uint8)
        rgbs.append(rgb)
        depth = np.full((H, W), float(fid) + 1.0, np.float32)
        seg = np.zeros((H, W, 3), np.uint8)
        seg[0, 0] = color  # instance 0 visible only at pixel (0,0)
        T = np.eye(4, dtype=np.float64)
        T[0, 3] = fid
        np.savez_compressed(base / "frames" / f"{fid}.npz", rgb=rgb, depth=depth, T=T, seg=seg)
    meta = {
        "version": 1, "scene": "999", "n_frames": n,
        "K": np.eye(3).tolist(),
        "inst_color": {"0": color},
        "objid": {"0": "Obj|0"},
        "gt": [{"instance_id": 0, "label": "chair",
                "T_world_obj": np.eye(4).tolist(), "extents": [1.0, 1.0, 1.0]}],
    }
    with open(base / "meta.json", "w") as f:
        json.dump(meta, f)
    return base, rgbs, color


def test_cache_replay_is_deterministic(tmp_path):
    base, rgbs, _ = _write_cache(tmp_path, n=3)
    s1 = ProcThorSource(scene="999", cache_dir=str(base))
    s2 = ProcThorSource(scene="999", cache_dir=str(base))
    f1, f2 = list(s1), list(s2)
    assert len(f1) == len(f2) == 3
    assert len(s1) == 3  # __len__ reads cache meta without a controller
    for a, b, expect in zip(f1, f2, rgbs):
        assert np.array_equal(a.rgb, b.rgb) and np.array_equal(a.rgb, expect)
        assert np.array_equal(a.depth, b.depth)
        assert np.array_equal(a.T_world_cam, b.T_world_cam)
    # controller must never be started on the replay path
    assert s1._controller is None and s2._controller is None


def test_cache_gt_and_native_mask_from_disk(tmp_path):
    base, _, color = _write_cache(tmp_path, n=2)
    s = ProcThorSource(scene="999", cache_dir=str(base))
    gt = s.gt()
    assert len(gt) == 1 and gt[0].label == "chair"
    assert isinstance(gt[0].mesh, trimesh.Trimesh)  # gt_mesh="box" rebuilt from extents
    frames = list(s)  # populates seg from cache
    m = s.native_mask(frames[0].frame_id, 0)
    assert m is not None and m[0, 0] == 255 and m.sum() == 255  # only pixel (0,0)
    assert s._controller is None


def test_max_frames_applied_on_replay(tmp_path):
    base, _, _ = _write_cache(tmp_path, n=5)
    s = ProcThorSource(scene="999", cache_dir=str(base), max_frames=2)
    assert len(list(s)) == 2 and len(s) == 2


def test_nocache_env_disables_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("R2S3D_PROCTHOR_NOCACHE", "1")
    s = ProcThorSource(scene="999", cache_dir=str(tmp_path / "cache"))
    assert s.cache_dir is None  # env override wins, would render live
