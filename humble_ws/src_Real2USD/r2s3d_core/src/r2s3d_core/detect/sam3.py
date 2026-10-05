"""SAM 3 open-vocabulary detector backend (HuggingFace ``transformers`` Sam3) —
a drop-in alternative to the YOLOE backend, producing the same ``DetectionSet``.

Why: the ObjectTrack front-end is detector-agnostic; SAM 3 gives SAM-quality
masks and (hopefully) higher recall than YOLOE. Because the reprojection
scale-fit reads the 2D mask boundary, cleaner SAM 3 masks may let scale-fit help
on the *deployed* path (not just the oracle GT-mask path). This backend is the
probe for that hypothesis (see docs/STATUS.md §NEXT).

SAM 3 is text-prompted per CONCEPT, each prompt returning every instance of that
concept (masks + boxes + scores). To avoid one forward per vocab word, we BATCH:
the frame is repeated against a chunk of vocab words in a single batched forward
(``images=[img]*k, text=[w1..wk]`` -> ``results[j]`` = instances of ``w_j``), so a
55-word vocab is a handful of batched calls, not 55 sequential ones. Cost is still
O(vocab) FLOPs (a real downside vs YOLOE's single pass, see [[sam3-detector-cost]]),
but GPU-parallel so wall-clock is far lower. We assign per-frame-unique track ids
and let ObjectTrack's 3D association link instances across frames — SAM 3's own
video tracking is not used in this per-image path (probe scope; future work).

Env:     ``uv run --extra sam3 --extra detector`` (transformers + cu128 torch).
Weights: ``facebook/sam3`` (GATED — the HF token must have accepted the license).
"""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import numpy as np

from .cache import Detection, DetectionSet

log = logging.getLogger(__name__)

_MODEL_ID = "facebook/sam3"
_BATCH = 12   # vocab words per batched forward (chunk to fit GPU memory)


def _load(device):
    import torch
    from transformers import Sam3Model, Sam3Processor
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = Sam3Model.from_pretrained(_MODEL_ID).to(dev).eval()
    proc = Sam3Processor.from_pretrained(_MODEL_ID)
    return model, proc, dev


def _to_np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def run_detector(source, prompt: str = "gt", vocab: Optional[Sequence[str]] = None,
                 conf: float = 0.5, mask_threshold: float = 0.5,
                 device: Optional[str] = None, **_ignored) -> DetectionSet:
    """Detect over every frame of ``source`` with SAM 3; return a DetectionSet.

    ``conf`` is SAM 3's instance score threshold. Extra kwargs (weights/iou/imgsz/
    tracker) are accepted and ignored so this is call-compatible with the YOLOE
    backend.
    """
    import torch
    from PIL import Image

    if prompt != "gt":
        log.warning("sam3 backend only wires the gt-vocab prompt; got %r -> using vocab", prompt)
    vocab = list(vocab or [])
    if not vocab:
        raise ValueError("sam3 backend needs a vocabulary (prompt=gt supplies the GT labels)")

    model, proc, dev = _load(device)
    log.info("SAM 3 loaded on %s; %d concept prompts/frame", dev, len(vocab))

    dets: List[Detection] = []
    H = W = None
    n_frames = 0
    for frame in source:
        n_frames += 1
        H, W = frame.rgb.shape[:2]
        img = Image.fromarray(frame.rgb)          # frame.rgb is RGB uint8
        tid = int(frame.frame_id) * 100000        # per-frame-unique base; 3D assoc links tracks
        for start in range(0, len(vocab), _BATCH):
            chunk = [str(w) for w in vocab[start:start + _BATCH]]
            inputs = proc(images=[img] * len(chunk), text=chunk,
                          return_tensors="pt").to(dev)
            with torch.no_grad():
                outputs = model(**inputs)
            results = proc.post_process_instance_segmentation(
                outputs, threshold=conf, mask_threshold=mask_threshold,
                target_sizes=inputs.get("original_sizes").tolist(),
            )  # one entry per (img, word) pair in the chunk
            for word, res in zip(chunk, results):
                masks, boxes, scores = res["masks"], res["boxes"], res["scores"]
                for i in range(len(masks)):
                    m = _to_np(masks[i]).astype(bool)
                    if m.shape != (H, W) or not m.any():
                        continue
                    dets.append(Detection(
                        frame_id=int(frame.frame_id),
                        bbox=_to_np(boxes[i]).astype(np.float64),
                        mask=m,
                        label=word.replace("_", " "),
                        score=float(_to_np(scores[i])),
                        track_id=tid,
                    ))
                    tid += 1
    if H is None:
        raise RuntimeError("source yielded no frames")
    meta = {"detector": "sam3", "model": _MODEL_ID, "prompt": "gt", "vocab": vocab,
            "conf": conf, "mask_threshold": mask_threshold, "n_frames": n_frames,
            "note": "per-image concept prompts; ObjectTrack 3D association links tracks"}
    log.info("SAM 3 produced %d detections over %d frames", len(dets), n_frames)
    return DetectionSet(scene=getattr(source, "scene", "scene"), height=H, width=W,
                        detections=dets, meta=meta)
