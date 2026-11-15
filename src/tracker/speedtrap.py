"""Two-line speed trap: speed = known distance between two lines / time between crossing them.

This is how average-speed cameras work. Unlike speed from a pixel scale, it does not depend
on the camera angle or perspective: it only needs the real distance between the two lines
(e.g. paced out, or measured on a map) and a still camera.

Crossing times are interpolated between frames (the moment the object's centre is exactly on
the line), so the timing is not limited to whole frames.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

MAX_GAP_S = 0.5      # do not interpolate a crossing across a longer gap in tracking


def _side(p, a, b) -> float:
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def crossing_times(center: np.ndarray, line: Sequence[float], fps: float) -> list:
    """Every crossing of segment ``(x1, y1, x2, y2)`` by the path, with sub-frame timing.

    Returns a list of ``{"time_s", "frame", "direction"}`` (``"A->B"`` side change as in
    :func:`tracker.analysis.line_crossings`), sorted by time.
    """
    a, b = (line[0], line[1]), (line[2], line[3])
    center = np.asarray(center, dtype=np.float64)
    valid = np.where(np.all(np.isfinite(center), axis=1))[0]
    events = []
    for i0, i1 in zip(valid[:-1], valid[1:]):
        if (i1 - i0) / fps > MAX_GAP_S:
            continue
        p, q = center[i0], center[i1]
        s0, s1 = _side(p, a, b), _side(q, a, b)
        if s0 == 0 or s0 * s1 > 0:
            continue
        f = s0 / (s0 - s1)                                   # where along p->q the line is crossed
        hit = p + f * (q - p)
        # the crossing point must lie on the drawn segment, not on its extension
        ab = np.array(b, float) - np.array(a, float)
        u = np.dot(hit - a, ab) / max(np.dot(ab, ab), 1e-12)
        if not (-0.02 <= u <= 1.02):
            continue
        t = (i0 + f * (i1 - i0)) / fps
        events.append({"time_s": float(t), "frame": int(round(t * fps)),
                       "direction": "A->B" if s0 < 0 else "B->A"})
    return events


def speed_trap(center: np.ndarray, line_a: Sequence[float], line_b: Sequence[float], distance_m: float,
               fps: float, limit_kmh: Optional[float] = None) -> list:
    """Passes through a two-line speed trap.

    A pass is a crossing of one line followed by a crossing of the other. An object can make
    several passes (e.g. a runner doing laps), each in either direction.

    Returns:
        list of ``{"from", "to", "enter_s", "exit_s", "seconds", "speed_kmh", "speed_ms",
        "over_limit"}``.
    """
    if distance_m <= 0:
        raise ValueError("Speed trap distance must be positive")
    events = [dict(e, line="A") for e in crossing_times(center, line_a, fps)]
    events += [dict(e, line="B") for e in crossing_times(center, line_b, fps)]
    events.sort(key=lambda e: e["time_s"])

    passes = []
    i = 0
    while i < len(events) - 1:
        e0, e1 = events[i], events[i + 1]
        if e0["line"] != e1["line"] and e1["time_s"] > e0["time_s"]:
            dt = e1["time_s"] - e0["time_s"]
            v = distance_m / dt
            passes.append({
                "from": e0["line"], "to": e1["line"],
                "enter_s": round(e0["time_s"], 4), "exit_s": round(e1["time_s"], 4),
                "exit_frame": e1["frame"], "seconds": round(dt, 4),
                "speed_ms": round(v, 3), "speed_kmh": round(v * 3.6, 2),
                "over_limit": bool(limit_kmh is not None and v * 3.6 > limit_kmh),
            })
            i += 2
        else:
            i += 1
    return passes
