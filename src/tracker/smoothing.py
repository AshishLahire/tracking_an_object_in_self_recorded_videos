"""Occlusion-aware trajectory smoothing."""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

VIS_THRESHOLD = 0.5


def visible_runs(mask) -> list:
    """Return (start, end) index pairs (end exclusive) of consecutive True values."""
    runs, start = [], None
    for t, v in enumerate(mask):
        if v and start is None:
            start = t
        elif not v and start is not None:
            runs.append((start, t))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def smooth_tracks(tracks: np.ndarray, visibility: np.ndarray, window: int = 9, polyorder: int = 3,
                  vis_threshold: float = VIS_THRESHOLD) -> np.ndarray:
    """Savitzky-Golay smoothing applied separately to each visible stretch of a track.

    Predictions made while a point is occluded are left untouched and never mixed
    into the visible path. Stretches shorter than ``polyorder + 2`` frames are not smoothed.

    Args:
        tracks: (T, N, 2) x/y positions.
        visibility: (T, N) visibility scores or booleans.
    """
    smoothed = np.array(tracks, dtype=np.float32, copy=True)
    _, n_points, _ = smoothed.shape
    for pid in range(n_points):
        for s, e in visible_runs(visibility[:, pid] > vis_threshold):
            length = e - s
            win = min(window, length if length % 2 == 1 else length - 1)   # odd, <= run length
            if win > polyorder:
                smoothed[s:e, pid, :] = savgol_filter(tracks[s:e, pid, :], win, polyorder, axis=0)
    return smoothed
