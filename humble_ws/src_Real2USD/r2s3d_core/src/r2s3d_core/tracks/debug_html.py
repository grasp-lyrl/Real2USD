"""Self-contained per-scene debug page for ObjectTrack runs.

PHASE_SPECS §Phase 2 done-when: "association decisions inspectable in a per-scene
debug HTML." One card per track: state, label-vote histogram, kept-view thumbnails,
detector track ids it absorbed, and what it merged from — so over-fragmentation and
the association/late-merge cleanup are eyeballable. No external assets (thumbnails are
inlined as base64 PNGs).
"""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path
from typing import List

import numpy as np

from .types import ObjectTrack, TrackState

_STATE_COLOR = {
    TrackState.MATURE: "#4ade80", TrackState.ACTIVE: "#60a5fa",
    TrackState.RECONSTRUCTED: "#a78bfa", TrackState.REGISTERED: "#22d3ee",
    TrackState.TENTATIVE: "#9ca3af", TrackState.REJECTED: "#f87171",
    TrackState.MERGED: "#fbbf24",
}


def _thumb(rgb_crop: np.ndarray, max_px: int = 120) -> str:
    from PIL import Image

    im = Image.fromarray(np.ascontiguousarray(rgb_crop))
    im.thumbnail((max_px, max_px))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _votes_bar(track: ObjectTrack) -> str:
    if not track.label_votes:
        return ""
    total = sum(track.label_votes.values())
    parts = []
    for lbl, n in track.label_votes.most_common():
        pct = 100 * n / total
        parts.append(f'<span class="vote">{html.escape(lbl)} '
                     f'<b>{n}</b><span class="bar" style="width:{pct:.0f}px"></span></span>')
    return " ".join(parts)


def _card(track: ObjectTrack) -> str:
    color = _STATE_COLOR.get(track.state, "#ddd")
    thumbs = "".join(
        f'<img src="{_thumb(o.rgb_crop)}" title="frame {o.frame_id} score {o.view_score:.2f}">'
        for o in track.kept_views if o.rgb_crop is not None)
    cen = track.centroid
    cen_s = f"({cen[0]:.2f}, {cen[1]:.2f}, {cen[2]:.2f})" if cen is not None else "—"
    merged = f" ⇐ merged {track.merged_from}" if track.merged_from else ""
    return f"""
    <div class="card">
      <div class="hdr"><span class="pill" style="background:{color}">{track.state.value}</span>
        <b>track {track.track_id}</b> · {track.n_obs} obs · {len(track.kept_views)} kept views
        · det ids {sorted(track.det_track_ids)}{merged}</div>
      <div class="votes">{_votes_bar(track)}</div>
      <div class="meta">centroid {cen_s} · fused cloud {len(track.fused_cloud)} pts</div>
      <div class="thumbs">{thumbs or '<i>no retained views</i>'}</div>
    </div>"""


def write_debug_html(tracks: List[ObjectTrack], out_path, scene: str = "",
                     config: dict | None = None) -> Path:
    order = {s: i for i, s in enumerate(
        [TrackState.MATURE, TrackState.REGISTERED, TrackState.RECONSTRUCTED,
         TrackState.ACTIVE, TrackState.MERGED, TrackState.REJECTED, TrackState.TENTATIVE])}
    ts = sorted(tracks, key=lambda t: (order.get(t.state, 99), t.track_id))
    counts = {}
    for t in tracks:
        counts[t.state.value] = counts.get(t.state.value, 0) + 1
    summary = " · ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
    cfg = config or {}
    cfg_s = f"reid={cfg.get('reid')} late_merge={cfg.get('late_merge')} v1_dedup={cfg.get('v1_dedup')}"
    body = "".join(_card(t) for t in ts)
    doc = f"""<!doctype html><meta charset=utf-8>
<title>ObjectTrack debug — {html.escape(scene)}</title>
<style>
 body{{background:#0b0f17;color:#e5e7eb;font:14px/1.4 system-ui,sans-serif;margin:20px}}
 h1{{font-size:18px}} .sub{{color:#9ca3af;margin-bottom:16px}}
 .card{{background:#131a26;border:1px solid #223;border-radius:8px;padding:12px;margin:10px 0}}
 .hdr{{margin-bottom:6px}} .pill{{color:#0b0f17;border-radius:10px;padding:1px 8px;font-weight:700;font-size:12px}}
 .votes{{margin:4px 0}} .vote{{margin-right:12px;color:#cbd5e1}} .vote .bar{{display:inline-block;height:8px;background:#3b82f6;border-radius:4px;margin-left:4px;vertical-align:middle}}
 .meta{{color:#9ca3af;font-size:12px;margin:4px 0}}
 .thumbs img{{height:90px;border-radius:4px;margin:2px;border:1px solid #223}}
</style>
<h1>ObjectTrack debug — {html.escape(scene)}</h1>
<div class="sub">{summary} · {html.escape(cfg_s)}</div>
{body}
"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc)
    return out_path
