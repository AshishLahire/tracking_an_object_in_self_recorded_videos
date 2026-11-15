"""Turning a user selection (box) into query points, and tracked points back into a box."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def normalize_box(box: Sequence[float]) -> tuple:
    """Return ``(x1, y1, x2, y2)`` with x1 < x2 and y1 < y2."""
    x1, y1, x2, y2 = (float(v) for v in box)
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    if x2 - x1 < 2 or y2 - y1 < 2:
        raise ValueError(f"Box is too small: {box}")
    return x1, y1, x2, y2


def points_in_box(box: Sequence[float], grid: int = 4, margin: float = 0.15) -> list:
    """``grid x grid`` evenly spaced (x, y) points inside a box, kept away from its edges.

    The margin (fraction of the box size) keeps points off the background, which usually
    surrounds the object inside a hand-drawn box.
    """
    x1, y1, x2, y2 = normalize_box(box)
    grid = max(1, int(grid))
    if grid == 1:
        return [((x1 + x2) / 2, (y1 + y2) / 2)]
    w, h = x2 - x1, y2 - y1
    xs = np.linspace(x1 + margin * w, x2 - margin * w, grid)
    ys = np.linspace(y1 + margin * h, y2 - margin * h, grid)
    return [(float(x), float(y)) for y in ys for x in xs]


def boxes_from_tracks(tracks: np.ndarray, visibility: np.ndarray, box: Sequence[float],
                      query_frame: int, vis_threshold: float = 0.5,
                      min_points: int = 2, point_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Move the selected box along with the tracked points.

    For each frame a translation + uniform scale is estimated robustly (medians) from the
    points visible in that frame, relative to their positions on ``query_frame``, and applied
    to the original box. Frames with fewer than ``min_points`` visible points get NaN.

    Args:
        tracks: (T, N, 2) in the same pixel space as ``box``.
        box: (x1, y1, x2, y2) drawn on ``query_frame``.
        point_mask: (N,) bool, points to use (e.g. from :func:`object_point_mask`).

    Returns:
        (T, 4) boxes ``x1, y1, x2, y2`` (NaN where unknown).
    """
    x1, y1, x2, y2 = normalize_box(box)
    corners = np.array([[x1, y1], [x2, y2]], dtype=np.float64)
    ref = tracks[query_frame].astype(np.float64)                 # (N, 2)
    vis = np.asarray(visibility) > vis_threshold
    use = np.ones(ref.shape[0], bool) if point_mask is None else np.asarray(point_mask, dtype=bool)
    vis = vis & use[None, :]
    T = tracks.shape[0]
    out = np.full((T, 4), np.nan)

    def spread_center(p):
        # middle of the points' extent: stays on the object when its surface rotates and the
        # visible points bunch up on one side (a median would slide with them)
        if len(p) >= 8:
            lo, hi = np.percentile(p, 5, axis=0), np.percentile(p, 95, axis=0)
        else:
            lo, hi = p.min(axis=0), p.max(axis=0)
        return (lo + hi) / 2

    e0 = spread_center(ref[use]) if use.any() else corners.mean(axis=0)
    offset = corners - e0                                         # box corners relative to points' centre
    min_pair = 0.1 * float(np.hypot(x2 - x1, y2 - y1))            # ignore pairs too close to give a ratio

    shifts = np.full((T, 2), np.nan)
    scales = np.full(T, np.nan)
    all_idx = np.where(use)[0]
    for t in range(T):
        if vis[t].sum() < min_points:           # object not (sufficiently) visible in this frame
            continue
        # Translation uses ALL object points, including the model's estimates for points that
        # are momentarily hidden (e.g. rotated to the back of the object): with visible points
        # only, the box slides toward whichever side of the object stays in view.
        pt_all = tracks[t, all_idx].astype(np.float64)
        shifts[t] = spread_center(pt_all) - spread_center(ref[all_idx])

        # Scale uses VISIBLE points only: the ratio of pairwise distances now vs. on the query
        # frame (robust to single bad points; hidden-point estimates would inflate it).
        idx = np.where(vis[t])[0]
        if len(idx) >= 2:
            i, j = np.triu_indices(len(idx), k=1)
            d0 = np.linalg.norm(ref[idx[i]] - ref[idx[j]], axis=1)
            dt = np.linalg.norm(tracks[t, idx[i]] - tracks[t, idx[j]], axis=1)
            ok = d0 > min_pair
            if ok.any():
                scales[t] = float(np.clip(np.median(dt[ok] / d0[ok]), 0.2, 5.0))

    # Box size changes smoothly in real videos: fill gaps and median-filter the scale over time
    known = np.isfinite(scales)
    if known.any():
        frames = np.arange(T)
        scales = np.interp(frames, frames[known], scales[known])
        half = 4
        scales = np.array([np.median(scales[max(0, t - half): t + half + 1]) for t in range(T)])
    else:
        scales = np.ones(T)

    for t in range(T):
        if np.all(np.isfinite(shifts[t])):
            new = e0 + shifts[t] + scales[t] * offset
            out[t] = [new[0, 0], new[0, 1], new[1, 0], new[1, 1]]
    return out


def object_point_mask(tracks: np.ndarray, visibility: np.ndarray, box: Sequence[float], query_frame: int,
                      vis_threshold: float = 0.5, tolerance: float = 0.15,
                      core_fraction: float = 0.25) -> np.ndarray:
    """Which box points actually belong to the object (vs. background caught in the box).

    The points closest to the box centre (``core_fraction`` of them) are assumed to be on the
    object; their median motion is the object's motion. A point is kept when its own motion
    stays within ``tolerance`` x box diagonal of that motion (median over visible frames).

    Returns:
        (N,) bool mask; at least the core points are always kept.
    """
    x1, y1, x2, y2 = normalize_box(box)
    centre = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    diag = float(np.hypot(x2 - x1, y2 - y1))
    ref = tracks[query_frame].astype(np.float64)
    n = ref.shape[0]
    if n <= 3:
        return np.ones(n, bool)

    vis = np.asarray(visibility) > vis_threshold
    disp = tracks.astype(np.float64) - ref[None]                       # (T, N, 2)
    core = np.argsort(np.linalg.norm(ref - centre, axis=1))[:max(3, int(round(n * core_fraction)))]

    T = tracks.shape[0]
    core_disp = np.full((T, 2), np.nan)
    for t in range(T):
        idx = core[vis[t, core]]
        if len(idx):
            core_disp[t] = np.median(disp[t, idx], axis=0)

    dev = np.linalg.norm(disp - core_disp[:, None, :], axis=-1)       # (T, N), NaN where core unknown
    dev = np.where(vis & np.isfinite(dev), dev, np.nan)
    with np.errstate(all="ignore"):
        typical = np.nanmedian(dev, axis=0)                           # (N,)
    mask = np.isfinite(typical) & (typical <= tolerance * diag)
    mask[core] = True
    return mask


def object_center(tracks: np.ndarray, visibility: np.ndarray, vis_threshold: float = 0.5,
                  boxes: Optional[np.ndarray] = None) -> np.ndarray:
    """(T, 2) object position per frame: box centre if given, else median of visible points.

    Frames where the object is not visible are NaN.
    """
    T = tracks.shape[0]
    center = np.full((T, 2), np.nan)
    if boxes is not None:
        center[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2
        center[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2
        return center
    vis = np.asarray(visibility) > vis_threshold
    for t in range(T):
        if vis[t].any():
            center[t] = np.median(tracks[t, vis[t]], axis=0)
    return center
