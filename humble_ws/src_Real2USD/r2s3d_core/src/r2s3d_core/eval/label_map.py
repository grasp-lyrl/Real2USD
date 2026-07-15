"""CLIP-nearest-in-set label mapping (coworker / SuperMap fairi-sgbench protocol).

The benchmark scores label-aware Object F1 after snapping each predicted label to the
CLOSEST word in the scene's GT label vocabulary via CLIP text-embedding cosine -- NOT exact
string match. This is how open-vocab methods (and our `generic`/`pf` detector prompts) are
made comparable: ``refrigerator``->``fridge``, ``tv``->``television``, ``couch``->``sofa``.
Confirmed with the coworker 2026-07-14 ("using CLIP to get the closest in-set word").

Kept OUT of ``eval/metrics.py`` on purpose: metrics is pure numpy/scipy (torch-free); CLIP
(torch) is lazily imported only when a caller opts into mapping via ``--label-map clip``.

Reconciliation still open (docs/ACTION_ITEMS.md AI-7): the coworker's exact CLIP model,
prompt template, and whether a cosine floor is applied. Defaults here (ViT-B/32,
"a photo of a {}", no threshold) are our provisional match; a low-similarity ``threshold``
maps to ``fallback`` ("unknown") instead of forcing a class -- the "arbitrarily clever"
option the coworker flagged, off by default so we bit-match the shipped bandaid.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

_MODEL_CACHE: dict = {}


def _load_clip(model_name: str, device: Optional[str] = None):
    import clip  # lazy: torch dependency, only when mapping is requested
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    key = (model_name, device)
    if key not in _MODEL_CACHE:
        model, _ = clip.load(model_name, device=device)
        model.eval()
        _MODEL_CACHE[key] = (model, device)
    return _MODEL_CACHE[key]


def _embed(words: Sequence[str], model_name: str, template: str, device=None) -> np.ndarray:
    import clip
    import torch

    model, device = _load_clip(model_name, device)
    with torch.no_grad():
        tok = clip.tokenize([template.format(w) for w in words]).to(device)
        e = model.encode_text(tok).float()
        e = e / e.norm(dim=-1, keepdim=True)
    return e.cpu().numpy()


def build_label_mapping(pred_labels: Sequence[str], vocab: Sequence[str], *,
                        model_name: str = "ViT-B/32", template: str = "a photo of a {}",
                        threshold: Optional[float] = None, fallback: str = "unknown",
                        ) -> Dict[str, str]:
    """Map each distinct predicted label to the closest ``vocab`` word by CLIP text cosine.

    ``threshold`` (cosine, e.g. 0.7): below it a label maps to ``fallback`` rather than being
    forced onto a class. Default None = force the argmax (matches the shipped benchmark).
    Returns ``{pred_label: in_set_word}``; empty inputs yield an empty mapping.
    """
    uniq = sorted({str(l) for l in pred_labels if str(l)})
    vocab = list(dict.fromkeys(v for v in vocab if v))
    if not uniq or not vocab:
        return {}
    sim = _embed(uniq, model_name, template) @ _embed(vocab, model_name, template).T
    out: Dict[str, str] = {}
    for i, u in enumerate(uniq):
        j = int(sim[i].argmax())
        out[u] = vocab[j] if (threshold is None or sim[i, j] >= threshold) else fallback
    return out


def remap_pred_labels(preds: List, vocab: Sequence[str], **kw) -> Dict[str, str]:
    """In place: rewrite each pred's ``.label`` to its CLIP-nearest ``vocab`` word.

    ``preds`` is a list of objects with a mutable ``.label`` (e.g. ``SceneObject``). Returns
    the mapping actually applied (for logging / provenance). No-op if ``preds`` is empty.
    """
    mapping = build_label_mapping([getattr(p, "label", "") for p in preds], vocab, **kw)
    for p in preds:
        lbl = getattr(p, "label", "")
        if lbl in mapping:
            p.label = mapping[lbl]
    return mapping
