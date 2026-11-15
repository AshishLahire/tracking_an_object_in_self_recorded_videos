"""Joint angles and repetition analysis (range-of-motion measurement for physiotherapy / rehab).

A joint is three tracked points: one end, the joint itself (the vertex) and the other end,
e.g. hip - knee - ankle. The angle at the vertex is measured in every frame.

The angle is measured in the image plane, so the camera should look at the joint side-on
(perpendicular to the plane the limb moves in); at an oblique view angles are distorted.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.signal import find_peaks, savgol_filter

from .smoothing import VIS_THRESHOLD, visible_runs


def angle_at(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle ABC in degrees (0-180) for (..., 2) arrays of points; NaN where undefined."""
    ba = np.asarray(a, dtype=np.float64) - b
    bc = np.asarray(c, dtype=np.float64) - b
    nb = np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.sum(ba * bc, axis=-1) / nb
    ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    return np.where(nb > 1e-9, ang, np.nan)


def joint_angle_series(tracks: np.ndarray, visibility: np.ndarray, idx: Sequence[int],
                       vis_threshold: float = VIS_THRESHOLD) -> np.ndarray:
    """(T,) angle at point ``idx[1]`` between ``idx[0]`` and ``idx[2]``; NaN when any is hidden."""
    i, j, k = idx
    ang = angle_at(tracks[:, i], tracks[:, j], tracks[:, k])
    vis = np.asarray(visibility) > vis_threshold
    ang[~(vis[:, i] & vis[:, j] & vis[:, k])] = np.nan
    return ang


def _fill_short_gaps(x: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly fill NaN gaps of at most ``max_gap`` frames (longer gaps stay NaN)."""
    x = x.copy()
    ok = np.isfinite(x)
    if ok.sum() < 2:
        return x
    for s, e in visible_runs(~ok):
        if e - s <= max_gap and s > 0 and e < len(x):
            x[s:e] = np.interp(np.arange(s, e), [s - 1, e], [x[s - 1], x[e]])
    return x


def repetition_stats(angle: np.ndarray, fps: float, min_rom_deg: float = 10.0) -> dict:
    """Range of motion and repetitions from a joint-angle series.

    * ``min_deg`` / ``max_deg``: 2nd / 98th percentile of the angle (robust to single-frame
      glitches); ``rom_deg`` is their difference (the range of motion).
    * ``reps``: number of movement cycles. A cycle is counted at each turning point (most bent
      or most straight, whichever kind occurs more often) whose prominence is at least
      ``max(min_rom_deg, 35 % of the range)``, so small wobbles are not counted.
    * ``rep_events``: frame, time and angle of each counted turning point.
    * ``mean_rep_s``: average time between consecutive counted turning points.
    """
    angle = np.asarray(angle, dtype=np.float64)
    filled = _fill_short_gaps(angle, max_gap=max(1, int(round(0.2 * fps))))
    ok = np.isfinite(filled)
    out = {"min_deg": float("nan"), "max_deg": float("nan"), "rom_deg": float("nan"),
           "reps": 0, "rep_events": [], "mean_rep_s": float("nan"),
           "measured_s": round(float(np.isfinite(angle).sum() / fps), 3)}
    if ok.sum() < 3:
        return out

    lo, hi = np.percentile(filled[ok], [2, 98])
    out.update(min_deg=round(float(lo), 2), max_deg=round(float(hi), 2), rom_deg=round(float(hi - lo), 2))

    # smooth each measured stretch a little so tracking jitter does not create turning points
    smooth = filled.copy()
    win = max(5, int(round(fps / 4)) | 1)
    for s, e in visible_runs(ok):
        if e - s > win:
            smooth[s:e] = savgol_filter(filled[s:e], win, 2)

    prominence = max(min_rom_deg, 0.35 * (hi - lo))
    sig = np.where(ok, smooth, np.nanmean(smooth[ok]))       # find_peaks cannot take NaN
    valleys, _ = find_peaks(-sig, prominence=prominence)
    peaks, _ = find_peaks(sig, prominence=prominence)
    valleys = [v for v in valleys if ok[v]]
    peaks = [p for p in peaks if ok[p]]
    events = valleys if len(valleys) >= len(peaks) else peaks
    kind = "most bent" if events is valleys else "most straight"

    out["reps"] = len(events)
    out["rep_events"] = [{"frame": int(f), "time_s": round(f / fps, 3), "angle_deg": round(float(smooth[f]), 1),
                          "kind": kind} for f in events]
    if len(events) >= 2:
        out["mean_rep_s"] = round(float(np.mean(np.diff(events)) / fps), 3)
    return out


def analyze_joints(tracks: np.ndarray, visibility: np.ndarray, joints: Sequence[Sequence[int]],
                   fps: float, names: Optional[Sequence[str]] = None) -> tuple:
    """Angle series and statistics for several joints.

    Returns:
        ``(angles, stats)``: angles is (T, J) degrees; stats is a list of dicts
        (``name``, ``points`` and the fields of :func:`repetition_stats`).
    """
    T = tracks.shape[0]
    angles = np.full((T, len(joints)), np.nan)
    stats = []
    for n, idx in enumerate(joints):
        angles[:, n] = joint_angle_series(tracks, visibility, idx)
        name = names[n] if names and n < len(names) and names[n] else f"joint {n}"
        stats.append({"name": name, "points": [int(v) for v in idx], **repetition_stats(angles[:, n], fps)})
    return angles, stats
