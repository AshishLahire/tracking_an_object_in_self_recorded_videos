"""Tracking evaluation metrics."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .smoothing import VIS_THRESHOLD


def evaluate_tracking(tracks: np.ndarray, visibility: np.ndarray,
                      gt_tracks: Optional[np.ndarray] = None,
                      gt_visibility: Optional[np.ndarray] = None,
                      pck_threshold: float = 5.0,
                      vis_threshold: float = VIS_THRESHOLD) -> dict:
    """Evaluate tracks of shape (T, N, 2) with visibility (T, N).

    Without ground truth only behaviour metrics are computed. They describe how the
    track behaves, not how accurate it is:

    * Smoothness: mean magnitude of the 2nd difference (acceleration) in px/frame^2,
      over visible frames only. Lower is smoother.
    * Continuity: fraction of (frame, point) pairs predicted visible.
    * Trajectory Length: mean path length in px per point, over visible steps only.

    With ``gt_tracks`` also MSE, EPE (mean end-point error, px) and PCK@threshold;
    with ``gt_visibility`` also visibility accuracy.
    """
    tracks = np.asarray(tracks, dtype=np.float64)
    vis = np.asarray(visibility) > vis_threshold
    results = {}

    if gt_tracks is not None:
        gt_tracks = np.asarray(gt_tracks, dtype=np.float64)
        if gt_tracks.shape != tracks.shape:
            raise ValueError(f"gt_tracks shape {gt_tracks.shape} != tracks shape {tracks.shape}")
        dist = np.linalg.norm(tracks - gt_tracks, axis=-1)
        results["MSE"] = float(np.mean((tracks - gt_tracks) ** 2))
        results["EPE"] = float(dist.mean())
        results[f"PCK@{pck_threshold:g}px"] = float(np.mean(dist < pck_threshold))
        if gt_visibility is not None:
            results["Visibility Accuracy"] = float(np.mean(vis == (np.asarray(gt_visibility) > 0.5)))

    # Acceleration only where three consecutive frames are visible
    if tracks.shape[0] >= 3:
        acc = np.linalg.norm(np.diff(tracks, n=2, axis=0), axis=-1)
        acc_mask = vis[:-2] & vis[1:-1] & vis[2:]
        results["Smoothness"] = float(acc[acc_mask].mean()) if acc_mask.any() else float("nan")
    else:
        results["Smoothness"] = float("nan")

    results["Continuity"] = float(vis.mean())

    # Path length only over steps where both ends are visible
    if tracks.shape[0] >= 2:
        steps = np.linalg.norm(np.diff(tracks, axis=0), axis=-1)
        step_mask = vis[:-1] & vis[1:]
        results["Trajectory Length"] = float(np.mean((steps * step_mask).sum(axis=0)))
    else:
        results["Trajectory Length"] = 0.0

    return results


def format_metrics(results: dict, title: str = "Evaluation Results") -> str:
    """Pretty, aligned text for printing metric dictionaries."""
    lines = [title]
    lines += [f"  {k:20s}: {v:.4f}" for k, v in results.items()]
    return "\n".join(lines)
