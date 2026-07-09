"""YOLOE detector runner (needs the ``detector`` optional extra: torch + ultralytics).

Runs Meta/Ultralytics YOLOE over a :class:`SequenceSource`'s RGB frames in trajectory
order (``model.track(persist=True)`` for BoT-SORT track ids that survive across frames)
and returns a :class:`DetectionSet`. This is the heavy step; everything downstream
consumes the cache and imports no torch (see ``detect/run.py``).

Prompt modes (the "prompting study"), mirroring v1's ``segment_cls.py``:
  * ``gt``      — text-prompt YOLOE with the scene's GT label vocabulary (oracle-vocab
                  upper bound; defensible open-vocab protocol).
  * ``generic`` — text-prompt with a fixed broad indoor noun-phrase list.
  * ``pf``      — prompt-free ``yoloe-11l-seg-pf.pt`` (native vocabulary).

Torch/ultralytics are imported lazily so the module can be imported (for constants)
without the extra installed; :func:`run_detector` raises a loud, actionable error if
it is missing.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import cv2
import numpy as np

from ..data.base import Frame
from .cache import Detection, DetectionSet

log = logging.getLogger(__name__)

# Broad indoor vocabulary for the `generic` prompt mode (no GT peeking).
GENERIC_VOCAB = [
    "chair", "table", "sofa", "couch", "bed", "desk", "cabinet", "shelf",
    "bookshelf", "lamp", "tv", "monitor", "potted plant", "vase", "stool",
    "bench", "pillow", "rug", "refrigerator", "oven", "sink", "toilet", "book",
    "clock", "picture frame", "basket", "bin", "bottle", "cup", "bowl", "laptop",
    "keyboard", "backpack", "box", "blanket", "towel", "cushion",
]

_WEIGHTS = {"gt": "yoloe-11l-seg.pt", "generic": "yoloe-11l-seg.pt",
            "pf": "yoloe-11l-seg-pf.pt"}


def _load_model(prompt: str, weights: Optional[str], vocab: Sequence[str], device):
    try:
        from ultralytics import YOLOE
    except Exception as e:  # pragma: no cover - needs the detector extra
        raise RuntimeError(
            "YOLOE needs the detector extra: `uv sync --extra detector` (torch + "
            "ultralytics). See docs/PHASE_SPECS.md §Phase 2 / DATASETS.md.") from e
    w = weights or _WEIGHTS[prompt]
    model = YOLOE(w)
    if device:
        model.to(device)
    if prompt != "pf":
        names = list(vocab)
        if not names:
            raise ValueError(f"prompt mode {prompt!r} needs a non-empty vocabulary")
        model.set_classes(names, model.get_text_pe(names))
        log.info("YOLOE %s prompted with %d classes", w, len(names))
    else:
        log.info("YOLOE %s prompt-free (native vocabulary)", w)
    return model


def run_detector(source, prompt: str = "gt", vocab: Optional[Sequence[str]] = None,
                 weights: Optional[str] = None, conf: float = 0.25, iou: float = 0.5,
                 device: Optional[str] = None, tracker: str = "botsort.yaml",
                 imgsz: int = 1024) -> DetectionSet:
    """Detect over every frame of ``source``; return a cached-ready DetectionSet."""
    if prompt not in _WEIGHTS:
        raise ValueError(f"prompt must be one of {sorted(_WEIGHTS)}; got {prompt!r}")
    vocab = list(vocab or (GENERIC_VOCAB if prompt == "generic" else []))
    model = _load_model(prompt, weights, vocab, device)

    dets: List[Detection] = []
    H = W = None
    n_frames = 0
    for frame in source:  # trajectory order -> tracking is meaningful
        n_frames += 1
        bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        H, W = frame.rgb.shape[:2]
        results = model.track(bgr, persist=True, conf=conf, iou=iou, imgsz=imgsz,
                              retina_masks=True, tracker=tracker, verbose=False)
        r = results[0]
        if r.masks is None or r.boxes is None:
            continue
        masks = r.masks.data.cpu().numpy()          # (N, h, w) in {0,1}
        boxes = r.boxes.xyxy.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        scores = r.boxes.conf.cpu().numpy()
        ids = (r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None
               else np.full(len(boxes), -1, int))
        names = r.names
        for i in range(len(boxes)):
            m = masks[i]
            if m.shape != (H, W):
                m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
            mask = m.astype(bool)
            if not mask.any():
                continue
            label = names.get(int(cls[i]), str(cls[i])) if isinstance(names, dict) else names[int(cls[i])]
            dets.append(Detection(
                frame_id=int(frame.frame_id), bbox=boxes[i].astype(np.float64),
                mask=mask, label=str(label).replace("_", " "),
                score=float(scores[i]), track_id=int(ids[i]),
            ))
    if H is None:
        raise RuntimeError("source yielded no frames")
    meta = {"detector": "yoloe", "weights": weights or _WEIGHTS[prompt], "prompt": prompt,
            "vocab": vocab, "conf": conf, "iou": iou, "imgsz": imgsz, "n_frames": n_frames}
    log.info("YOLOE produced %d detections over %d frames", len(dets), n_frames)
    return DetectionSet(scene=getattr(source, "scene", "scene"), height=H, width=W,
                        detections=dets, meta=meta)
