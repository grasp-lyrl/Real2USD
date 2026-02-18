"""
Pure helper for Ultralytics track_id handling. No heavy deps (no ultralytics).
Used by segment_cls and by tests so tests can run without ultralytics installed.
"""


def track_ids_from_boxes_id(boxes_id, num_boxes):
    """
    Convert Ultralytics result.boxes.id to a list of int (use -1 for no id).
    """
    if boxes_id is None:
        return [-1] * num_boxes
    try:
        raw = boxes_id.int().cpu().tolist()
    except (AttributeError, TypeError, Exception):
        return [-1] * num_boxes
    if not isinstance(raw, list):
        raw = [raw]
    out = [int(x) if x is not None else -1 for x in raw]
    if len(out) < num_boxes:
        out.extend([-1] * (num_boxes - len(out)))
    return out[:num_boxes]
