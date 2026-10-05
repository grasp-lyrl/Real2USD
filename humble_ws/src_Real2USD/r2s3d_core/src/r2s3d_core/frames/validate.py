"""Boundary input validation (Phase 1).

v1 failed silently: SAM3D scale was applied unchecked, depth was never validated,
malformed poses slipped through. Here every check is explicit and *loud* — it either
raises (hard violation) or returns a reason string to stamp into provenance. Never
pass a bad value through silently.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# SAM3D scale sanity window (docs/PHASE_SPECS.md Phase 1). Upstream documents ~3x
# scale errors; anything outside this is almost certainly garbage.
SAM3D_SCALE_MIN = 0.05
SAM3D_SCALE_MAX = 20.0


class ValidationError(ValueError):
    """Raised on a hard boundary violation."""


def check_depth(depth: np.ndarray, max_invalid_frac: float = 0.98,
                raise_on_fail: bool = False) -> Optional[str]:
    """Validate a depth map. Returns a reason string if suspect, else None.

    Flags: all-invalid frames, excessive hole fraction (0/NaN), negative depth.
    """
    d = np.asarray(depth, dtype=np.float64)
    invalid = ~np.isfinite(d) | (d <= 0)
    frac = float(invalid.mean()) if d.size else 1.0
    reason = None
    if d.size == 0:
        reason = "empty depth"
    elif frac >= max_invalid_frac:
        reason = f"depth {frac:.0%} invalid (holes/NaN/<=0)"
    elif np.any(d[np.isfinite(d)] < 0):
        reason = "negative depth values"
    if reason and raise_on_fail:
        raise ValidationError(reason)
    return reason


def check_sam3d_scale(scale, raise_on_fail: bool = True) -> Optional[str]:
    """SAM3D scale must be finite and within [SAM3D_SCALE_MIN, SAM3D_SCALE_MAX]."""
    s = np.asarray(scale, dtype=np.float64).reshape(-1)
    reason = None
    if not np.all(np.isfinite(s)):
        reason = f"non-finite scale {s.tolist()}"
    elif np.any(s < SAM3D_SCALE_MIN) or np.any(s > SAM3D_SCALE_MAX):
        reason = f"scale {s.tolist()} outside [{SAM3D_SCALE_MIN}, {SAM3D_SCALE_MAX}]"
    if reason and raise_on_fail:
        raise ValidationError(reason)
    return reason


def check_translation(t, raise_on_fail: bool = True) -> Optional[str]:
    """Translation must be finite."""
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(t)):
        reason = f"non-finite translation {t.tolist()}"
        if raise_on_fail:
            raise ValidationError(reason)
        return reason
    return None
