"""On-disk detection cache (torch-free).

A :class:`DetectionSet` is the interface between the heavy detector step
(``detect/yoloe.py``, needs torch) and the ROS-free tracker/eval/tests. It stores,
per scene, every per-frame detection (bbox, binary instance mask, label, score,
detector track id) plus provenance (source/scene/prompt/git_sha/config).

Layout on disk (``<dir>/<scene>/``)::

    meta.json          # human-readable provenance + frame index
    detections.npz     # compressed arrays; masks bit-packed (np.packbits)

Masks are the bulk, so they are bit-packed and the whole npz is compressed — sparse
indoor masks shrink a lot. All frames in a Replica scene share one resolution, stored
once as (H, W).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

PathLike = Union[str, "os.PathLike[str]"]


@dataclass
class Detection:
    """One detector output in a single frame (full-image pixel coordinates)."""

    frame_id: int
    bbox: np.ndarray          # (4,) xyxy, float
    mask: np.ndarray          # (H, W) bool, full-image instance mask
    label: str
    score: float
    track_id: int             # detector/tracker id; -1 if untracked


@dataclass
class DetectionSet:
    """All detections for one scene, in frame order."""

    scene: str
    height: int
    width: int
    detections: List[Detection] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ access
    def frame_ids(self) -> List[int]:
        return sorted({int(d.frame_id) for d in self.detections})

    def by_frame(self) -> Dict[int, List[Detection]]:
        out: Dict[int, List[Detection]] = {}
        for d in self.detections:
            out.setdefault(int(d.frame_id), []).append(d)
        return out

    def __len__(self) -> int:
        return len(self.detections)

    # -------------------------------------------------------------------- I/O
    def save(self, out_dir: PathLike) -> Path:
        d = Path(out_dir) / self.scene
        d.mkdir(parents=True, exist_ok=True)
        n = len(self.detections)
        H, W = self.height, self.width
        boxes = np.zeros((n, 4), np.float32)
        scores = np.zeros((n,), np.float32)
        track_ids = np.zeros((n,), np.int32)
        frame_ids = np.zeros((n,), np.int32)
        labels = np.array([det.label for det in self.detections], dtype=object)
        # bit-pack masks: (n, H*W) bool -> (n, ceil(H*W/8)) uint8
        packed = np.zeros((n, (H * W + 7) // 8), np.uint8) if n else np.zeros((0, 0), np.uint8)
        for i, det in enumerate(self.detections):
            boxes[i] = det.bbox
            scores[i] = det.score
            track_ids[i] = det.track_id
            frame_ids[i] = det.frame_id
            packed[i] = np.packbits(np.asarray(det.mask, bool).reshape(-1))
        np.savez_compressed(
            d / "detections.npz",
            boxes=boxes, scores=scores, track_ids=track_ids, frame_ids=frame_ids,
            labels=labels, masks_packed=packed, hw=np.array([H, W], np.int32),
        )
        meta = dict(self.meta)
        meta.update({"scene": self.scene, "height": H, "width": W, "n_detections": n,
                     "frame_ids": self.frame_ids()})
        with open(d / "meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=_json_default)
        return d

    @classmethod
    def load(cls, in_dir: PathLike, scene: Optional[str] = None) -> "DetectionSet":
        d = Path(in_dir)
        if scene is not None and (d / scene).is_dir():
            d = d / scene
        if not (d / "detections.npz").is_file():
            raise FileNotFoundError(
                f"no detections.npz under {d}. Run the detector step first:\n"
                f"  uv run python -m r2s3d_core.detect.run --source <src> --scene <scene> "
                f"--prompt gt --out {Path(in_dir)}"
            )
        z = np.load(d / "detections.npz", allow_pickle=True)
        with open(d / "meta.json") as f:
            meta = json.load(f)
        H, W = (int(x) for x in z["hw"])
        boxes, scores = z["boxes"], z["scores"]
        track_ids, frame_ids = z["track_ids"], z["frame_ids"]
        labels, packed = z["labels"], z["masks_packed"]
        dets: List[Detection] = []
        for i in range(len(boxes)):
            mask = np.unpackbits(packed[i])[: H * W].astype(bool).reshape(H, W)
            dets.append(Detection(
                frame_id=int(frame_ids[i]), bbox=boxes[i].astype(np.float64),
                mask=mask, label=str(labels[i]), score=float(scores[i]),
                track_id=int(track_ids[i]),
            ))
        return cls(scene=meta.get("scene", scene or d.name), height=H, width=W,
                   detections=dets, meta=meta)


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serializable: {type(o)}")
