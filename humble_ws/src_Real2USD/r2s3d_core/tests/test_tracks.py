"""ObjectTrack tracker: association cascade, lifecycle, voxel fusion, re-ID
break-healing, and late merge — all on a tiny synthetic RGB-D + detection fixture
(no YOLOE, no torch)."""

import numpy as np

from r2s3d_core.data.base import Frame
from r2s3d_core.detect.cache import Detection, DetectionSet
from r2s3d_core.detect import corrupt
from r2s3d_core.tracks import TrackState, VoxelCloud, run_tracker

H, W = 120, 160
K = np.array([[100.0, 0, 80.0], [0, 100.0, 60.0], [0, 0, 1.0]])

# Two static objects: A (red) left @ depth 2 m, B (blue) right @ depth 3 m.
_A = dict(rows=(40, 71), cols=(30, 61), depth=2.0, color=(220, 20, 20))
_B = dict(rows=(40, 71), cols=(100, 131), depth=3.0, color=(20, 20, 220))


def _rect_mask(spec):
    m = np.zeros((H, W), bool)
    m[spec["rows"][0]:spec["rows"][1], spec["cols"][0]:spec["cols"][1]] = True
    return m


def _bbox(mask):
    ys, xs = np.where(mask)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], float)


def _frame(fid):
    rgb = np.zeros((H, W, 3), np.uint8)
    depth = np.zeros((H, W), np.float32)
    for spec in (_A, _B):
        m = _rect_mask(spec)
        rgb[m] = spec["color"]
        depth[m] = spec["depth"]
    return Frame(rgb=rgb, depth=depth, K=K.copy(), T_world_cam=np.eye(4),
                 stamp=float(fid), frame_id=fid)


def _det(fid, spec, track_id):
    m = _rect_mask(spec)
    return Detection(frame_id=fid, bbox=_bbox(m), mask=m,
                     label="chair" if spec is _A else "table", score=0.9,
                     track_id=track_id)


def _frames(n=8):
    return [_frame(i) for i in range(n)]


def _dets(n=8, a_ids=None, b_ids=None):
    """a_ids/b_ids: per-frame detector track id for A/B (default stable 1/2)."""
    a_ids = a_ids or [1] * n
    b_ids = b_ids or [2] * n
    ds = DetectionSet(scene="fix", height=H, width=W)
    for i in range(n):
        ds.detections.append(_det(i, _A, a_ids[i]))
        ds.detections.append(_det(i, _B, b_ids[i]))
    return ds


def _mature(tracks):
    return [t for t in tracks if t.state == TrackState.MATURE]


# --------------------------------------------------------------------- tests

def test_associate_by_id_two_objects():
    frames, ds = _frames(), _dets()
    tracks = run_tracker(frames, ds.by_frame(), {"reid": True, "late_merge": True})
    mature = _mature(tracks)
    assert len(mature) == 2
    centroids = np.array([t.centroid for t in mature])
    # the two objects are well separated in world
    assert np.linalg.norm(centroids[0] - centroids[1]) > 0.5
    # each track fused a real cloud and voted a label
    for t in mature:
        assert len(t.fused_cloud) > 0
        assert t.label() in ("chair", "table")


def test_lifecycle_tentative_rejected():
    # object A seen only twice (< MIN_ACTIVE_OBS) -> never ACTIVE -> REJECTED
    ds = DetectionSet(scene="fix", height=H, width=W)
    for i in range(8):
        ds.detections.append(_det(i, _B, 2))          # B every frame
        if i < 2:
            ds.detections.append(_det(i, _A, 1))       # A only twice
    tracks = run_tracker(_frames(), ds.by_frame(), {"reid": True, "late_merge": True})
    assert len(_mature(tracks)) == 1                   # only B matures
    assert any(t.state == TrackState.REJECTED for t in tracks)


def test_reid_heals_track_break():
    # A's detector id breaks (1 -> 99) at frame 4; B stable.
    a_ids = [1, 1, 1, 1, 99, 99, 99, 99]
    ds = _dets(a_ids=a_ids)
    on = run_tracker(_frames(), _dets(a_ids=a_ids).by_frame(),
                     {"reid": True, "late_merge": True})
    off = run_tracker(_frames(), ds.by_frame(),
                      {"reid": False, "late_merge": False})
    assert len(_mature(on)) == 2                        # break healed
    assert len(_mature(off)) == 3                        # A fragments into 2 + B
    # the healed A track carries both detector ids
    a_track = min(_mature(on), key=lambda t: t.centroid[0])
    assert {1, 99} <= a_track.det_track_ids


def test_late_merge_collapses_duplicate_object():
    # reid OFF so the id break spawns two A tracks at ingest; late_merge should fold
    # them back (identical overlapping clouds). (id 3, not 2 — 2 is object B's id.)
    a_ids = [1, 1, 1, 1, 3, 3, 3, 3]
    no_merge = run_tracker(_frames(), _dets(a_ids=a_ids).by_frame(),
                           {"reid": False, "late_merge": False})
    merged = run_tracker(_frames(), _dets(a_ids=a_ids).by_frame(),
                         {"reid": False, "late_merge": True})
    assert len(_mature(no_merge)) == 3
    assert len(_mature(merged)) == 2
    a_track = min(_mature(merged), key=lambda t: t.centroid[0])
    assert a_track.merged_from                          # records what it absorbed


def test_voxel_cloud_dedups_within_1cm():
    vc = VoxelCloud(0.01)
    vc.add(np.array([[0.0, 0.0, 0.0], [0.003, 0.0, 0.0], [0.006, 0.002, 0.0]]))
    assert vc.n == 1                                    # all in one 1 cm voxel
    vc.add(np.array([[0.05, 0.0, 0.0]]))
    assert vc.n == 2


def test_corrupt_split_fractures_detection():
    ds = _dets(n=2)
    n0 = len(ds)
    split = corrupt.apply(ds, split=1.0, seed=0)
    # every detection split into (up to) two -> more detections, fresh ids
    assert len(split) > n0
    assert split.meta["corruptions"]["split"] == 1.0


class _FakeSource:
    """Minimal SequenceSource over the fixture frames (for object_track wiring)."""

    def __init__(self, frames, scene="fix"):
        self._frames = frames
        self.scene = scene

    def __iter__(self):
        return iter(self._frames)

    def __len__(self):
        return len(self._frames)

    def gt(self):
        return None


def test_object_track_wiring_queues_sam3d(tmp_path):
    from r2s3d_core.baselines import get_method

    ds = _dets()
    ds.save(tmp_path / "det")
    queue = tmp_path / "queue"
    cfg = {"detections": str(tmp_path / "det"), "sam3d_queue": str(queue),
           "full_frame": True, "reid": None, "late_merge": None}
    method = get_method("object_track")
    # no SAM3D worker -> jobs get queued, 0 predictions returned, but stats populated
    preds = method(_FakeSource(_frames()), [1, 2], cfg)
    assert preds == []
    stats = cfg["_method_stats"]["fix"]
    assert stats["n_mature"] == 2                        # two objects -> two mature tracks
    assert stats["sam3d_invocations"] == 2               # one SAM3D job per mature track
    # jobs were actually written to the queue for the worker to pick up
    assert (queue / "input").is_dir() and any((queue / "input").iterdir())


def test_object_track_naive_more_invocations_than_full(tmp_path):
    """Detector id break -> naive fragments (more SAM3D calls) than full pipeline."""
    from r2s3d_core.baselines import get_method

    a_ids = [1, 1, 1, 1, 99, 99, 99, 99]
    ds = _dets(a_ids=a_ids)
    ds.save(tmp_path / "det")
    base = {"detections": str(tmp_path / "det"), "sam3d_queue": str(tmp_path / "q"),
            "full_frame": True}

    full_cfg = dict(base, reid=None, late_merge=None)
    get_method("object_track")(_FakeSource(_frames()), [1, 2], full_cfg)
    naive_cfg = dict(base, reid=None, late_merge=None)
    get_method("object_track_naive")(_FakeSource(_frames()), [1, 2], naive_cfg)

    assert naive_cfg["_method_stats"]["fix"]["sam3d_invocations"] > \
        full_cfg["_method_stats"]["fix"]["sam3d_invocations"]


def test_debug_html_renders(tmp_path):
    from r2s3d_core.tracks.debug_html import write_debug_html

    tracks = run_tracker(_frames(), _dets().by_frame(), {"reid": True, "late_merge": True})
    out = write_debug_html(tracks, tmp_path / "fix_tracks.html", scene="fix",
                           config={"reid": True, "late_merge": True})
    assert out.is_file()
    txt = out.read_text()
    assert "ObjectTrack debug" in txt and "track 0" in txt and "data:image/png;base64" in txt


def test_cache_roundtrip(tmp_path):
    ds = _dets(n=3)
    ds.save(tmp_path)
    back = DetectionSet.load(tmp_path, scene="fix")
    assert len(back) == len(ds)
    assert back.height == H and back.width == W
    d0, b0 = ds.detections[0], back.detections[0]
    assert np.array_equal(d0.mask, b0.mask)
    assert d0.label == b0.label and d0.track_id == b0.track_id
