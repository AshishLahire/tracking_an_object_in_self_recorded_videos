"""Turning an object's path into measurements: speed, distance, line crossings, zones, heatmap."""

from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

from .smoothing import smooth_tracks


def meters_per_pixel(p1: Sequence[float], p2: Sequence[float], length_m: float) -> float:
    """Scale from a reference of known real length, e.g. a 1 m stick marked from p1 to p2."""
    px = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
    if px < 1 or length_m <= 0:
        raise ValueError("Reference must be at least 1 px long and have a positive real length")
    return float(length_m) / px


def motion_stats(center: np.ndarray, fps: float, m_per_px=None, smooth: bool = True) -> dict:
    """Speed and distance from an object path.

    Args:
        center: (T, 2) positions in pixels, NaN where the object is not visible.
        fps: frames per second.
        m_per_px: metres per pixel (from :func:`meters_per_pixel`), either one number or a
            (T,) array that changes per frame (e.g. from :func:`object_scale_m_per_px` when the
            object moves closer / farther). Without it the results are in pixels.

    Returns:
        dict with ``speed`` (T,) per frame (NaN where unknown), ``unit_distance``,
        ``unit_speed`` and summary values ``distance``, ``avg_speed``, ``max_speed``
        (95th percentile, robust to single-frame jumps) and ``moving_time_s``.
    """
    center = np.asarray(center, dtype=np.float64)
    valid = np.all(np.isfinite(center), axis=1)
    path = center.copy()
    if smooth and valid.sum() >= 5:
        filled = np.where(valid[:, None], center, 0.0)
        path = smooth_tracks(filled[:, None, :], valid[:, None].astype(float), window=7, polyorder=2)[:, 0]
        path[~valid] = np.nan

    step = np.linalg.norm(np.diff(path, axis=0), axis=1)          # px per frame, NaN across gaps
    calibrated = m_per_px is not None and bool(np.any(np.asarray(m_per_px, dtype=float) > 0))
    if not calibrated:
        scale = np.ones(len(step))
    elif np.ndim(m_per_px) == 0:
        scale = np.full(len(step), float(m_per_px))
    else:
        mpp = np.asarray(m_per_px, dtype=np.float64)
        scale = (mpp[:-1] + mpp[1:]) / 2                          # per step: mean of both frames
    step = step * scale                                           # metres (or px) per frame
    speed = np.full(len(center), np.nan)
    speed[1:] = step * fps
    if calibrated:
        speed_unit, dist_unit = "km/h", "m"
        speed = speed * 3.6
    else:
        speed_unit, dist_unit = "px/s", "px"

    ok = np.isfinite(speed)
    return {
        "speed": speed,
        "unit_speed": speed_unit,
        "unit_distance": dist_unit,
        "distance": float(np.nansum(step)),
        "avg_speed": float(np.mean(speed[ok])) if ok.any() else float("nan"),
        "max_speed": float(np.percentile(speed[ok], 95)) if ok.any() else float("nan"),
        "moving_time_s": float(ok.sum() / fps),
    }


def _side(p, a, b) -> float:
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def _segments_intersect(p, q, a, b) -> bool:
    d1, d2 = _side(p, a, b), _side(q, a, b)
    d3, d4 = _side(a, p, q), _side(b, p, q)
    return (d1 * d2 < 0) and (d3 * d4 <= 0)


def line_crossings(center: np.ndarray, line: Sequence[float], fps: float) -> list:
    """Times the object path crosses a line segment ``(x1, y1, x2, y2)``.

    Returns a list of ``{"frame", "time_s", "direction"}``; direction is ``"A->B"`` when moving
    from the left side of the line (looking from point 1 to point 2) to the right, else ``"B->A"``.
    """
    a, b = (line[0], line[1]), (line[2], line[3])
    events = []
    last_p = None            # last position that was clearly on one side (not exactly on the line)
    for t, p in enumerate(np.asarray(center, dtype=np.float64)):
        if not np.all(np.isfinite(p)) or _side(p, a, b) == 0:
            continue
        if last_p is not None and _segments_intersect(last_p, p, a, b):
            direction = "A->B" if _side(last_p, a, b) < 0 else "B->A"
            events.append({"frame": t, "time_s": round(t / fps, 3), "direction": direction})
        last_p = p
    return events


def zone_events(center: np.ndarray, polygon: Sequence[Sequence[float]], fps: float) -> dict:
    """Entries / exits of a polygon zone and total time inside it."""
    poly = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    if len(poly) < 3:
        raise ValueError("Zone polygon needs at least 3 points")
    inside_prev, entries, exits, frames_inside = None, [], [], 0
    for t, p in enumerate(np.asarray(center, dtype=np.float64)):
        if not np.all(np.isfinite(p)):
            continue
        inside = cv2.pointPolygonTest(poly, (float(p[0]), float(p[1])), False) >= 0
        frames_inside += int(inside)
        if inside_prev is not None and inside != inside_prev:
            (entries if inside else exits).append({"frame": t, "time_s": round(t / fps, 3)})
        inside_prev = inside
    return {"entries": entries, "exits": exits, "time_inside_s": round(frames_inside / fps, 3)}


def heatmap(points: np.ndarray, size: tuple, sigma: Optional[float] = None) -> np.ndarray:
    """Density map (H, W) in [0, 1] of where the object was, from (M, 2) or (T, 2) points."""
    w, h = size
    heat = np.zeros((h, w), np.float32)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    for x, y in pts:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < w and 0 <= yi < h:
            heat[yi, xi] += 1
    if sigma is None:
        sigma = max(w, h) / 60
    heat = cv2.GaussianBlur(heat, (0, 0), sigma)
    if heat.max() > 0:
        heat /= heat.max()
    return heat


def heatmap_overlay(frame_rgb: np.ndarray, heat: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Colour the heatmap and blend it over an RGB frame (cold areas stay transparent)."""
    colored = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    a = (alpha * np.clip(heat * 3, 0, 1))[..., None]
    return (frame_rgb * (1 - a) + colored * a).astype(np.uint8)


def object_scale_m_per_px(m_per_px_ref: float, boxes: np.ndarray, query_frame: int,
                          zoom: Optional[np.ndarray] = None, smooth_frames: int = 15) -> np.ndarray:
    """Per-frame metres-per-pixel when the reference was measured ON the tracked object.

    As the object comes closer (or the camera zooms in) it covers more pixels, so each pixel
    is fewer metres. The object's apparent size is taken from its tracked box width relative
    to ``query_frame`` (times ``zoom``, the camera-compensation scale, when positions were
    mapped to a reference frame), median-filtered over ``smooth_frames`` to reduce jitter.

    Returns:
        (T,) metres per pixel; frames without a box use the nearest known value.
    """
    widths = np.asarray(boxes[:, 2] - boxes[:, 0], dtype=np.float64)
    w0 = widths[query_frame]
    if not np.isfinite(w0) or w0 <= 0:
        w0 = np.nanmedian(widths)
    size = widths / w0
    if zoom is not None:
        size = size * np.asarray(zoom, dtype=np.float64)
    T = len(size)
    ok = np.isfinite(size) & (size > 0)
    if not ok.any():
        return np.full(T, float(m_per_px_ref))
    idx = np.arange(T)
    size = np.interp(idx, idx[ok], size[ok])                      # fill gaps from neighbours
    half = max(0, smooth_frames // 2)
    size = np.array([np.median(size[max(0, t - half): t + half + 1]) for t in range(T)])
    return float(m_per_px_ref) / np.clip(size, 0.05, 20)
