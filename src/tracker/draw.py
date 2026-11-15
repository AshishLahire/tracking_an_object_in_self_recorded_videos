"""Drawing tracked points, trajectories, boxes and measurement overlays on frames."""

from __future__ import annotations

from typing import Iterator, Optional, Sequence

import cv2
import numpy as np

from .smoothing import VIS_THRESHOLD

BOX_COLOR = (0, 255, 0)        # BGR
LINE_COLOR = (0, 215, 255)
ZONE_COLOR = (255, 160, 0)
TEXT_BG = (0, 0, 0)
CURVE_COLOR = (0, 255, 255)


def point_colors(n: int, seed: int = 0) -> list:
    """``n`` bright, well-separated BGR colours (reproducible for a given seed)."""
    if n <= 0:
        return []
    rng = np.random.default_rng(seed)
    offset = int(rng.integers(0, 180))
    hues = [(offset + i * 180 // n) % 180 for i in range(n)]
    hues = [hues[i] for i in rng.permutation(n)]
    return [tuple(int(c) for c in cv2.cvtColor(np.uint8([[[h, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])
            for h in hues]


def draw_label(img: np.ndarray, text: str, org: tuple, scale: float = 0.5,
               color=(255, 255, 255), thickness: int = 1) -> None:
    """Text with a dark background box, for readability on any video."""
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = int(org[0]), int(org[1])
    cv2.rectangle(img, (x - 2, y - th - 4), (x + tw + 2, y + base), TEXT_BG, -1)
    cv2.putText(img, text, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_static_overlays(img: np.ndarray, lines: Optional[Sequence] = None,
                         zones: Optional[Sequence] = None, line_thickness: int = 2) -> None:
    """Counting lines ``(x1, y1, x2, y2[, label])`` and zone polygons, in frame pixels."""
    for zone in zones or []:
        pts = np.asarray(zone, dtype=np.int32).reshape(-1, 1, 2)
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], ZONE_COLOR)
        cv2.addWeighted(overlay, 0.2, img, 0.8, 0, dst=img)
        cv2.polylines(img, [pts], True, ZONE_COLOR, line_thickness, cv2.LINE_AA)
    for line in lines or []:
        x1, y1, x2, y2 = (int(round(v)) for v in line[:4])
        cv2.line(img, (x1, y1), (x2, y2), LINE_COLOR, line_thickness, cv2.LINE_AA)
        cv2.circle(img, (x1, y1), 4, LINE_COLOR, -1)
        cv2.putText(img, "A", (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, LINE_COLOR, 1, cv2.LINE_AA)
        cv2.putText(img, "B", (x2 + 4, y2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, LINE_COLOR, 1, cv2.LINE_AA)
        if len(line) > 4 and line[4]:
            draw_label(img, str(line[4]), ((x1 + x2) / 2, (y1 + y2) / 2 - 6), 0.45, LINE_COLOR)


def render_frames(frames: np.ndarray, tracks: np.ndarray, visibility: np.ndarray,
                  point_radius: int = 4, line_thickness: int = 2,
                  vis_threshold: float = VIS_THRESHOLD, seed: int = 0,
                  boxes: Optional[np.ndarray] = None,
                  lines: Optional[Sequence] = None, zones: Optional[Sequence] = None,
                  texts: Optional[Sequence[Optional[str]]] = None,
                  draw_trails=True, point_radii: Optional[Sequence[int]] = None,
                  joints: Optional[Sequence[dict]] = None,
                  curves: Optional[Sequence[np.ndarray]] = None) -> Iterator[np.ndarray]:
    """Yield BGR frames with each point's current position and its trajectory so far.

    The trajectory is split into segments at occlusions, so no line is drawn
    across a gap where the point was not visible.

    Args:
        frames: (T, H, W, 3) RGB uint8.
        tracks: (T, N, 2) positions in the same pixel space as ``frames``.
        visibility: (T, N) visibility scores or booleans.
        boxes: optional (T, 4) object boxes (NaN rows are skipped).
        lines / zones: static overlays, see :func:`draw_static_overlays`.
        texts: optional per-frame label drawn top-left (e.g. current speed).
        draw_trails: draw each point's past path; a bool, or one bool per point.
        point_radii: optional radius per point (overrides ``point_radius``).
        joints: optional list of ``{"idx": (i, j, k), "angle": (T,) degrees, "name": str}``;
            draws the two limb segments and the current angle at the joint.
        curves: optional static (M, 2) polylines, e.g. a fitted trajectory.
    """
    num_frames = frames.shape[0]
    num_points = tracks.shape[1]
    colors = point_colors(num_points, seed)
    trails = [bool(draw_trails)] * num_points if np.isscalar(draw_trails) else [bool(v) for v in draw_trails]
    radii = list(point_radii) if point_radii is not None else [point_radius] * num_points

    segments = [[] for _ in range(num_points)]
    was_visible = [False] * num_points

    for t in range(num_frames):
        frame_bgr = cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR)
        draw_static_overlays(frame_bgr, lines, zones, line_thickness)
        for curve in curves or []:
            pts = np.asarray(curve, dtype=np.float64)
            pts = pts[np.all(np.isfinite(pts), axis=1)].round().astype(np.int32).reshape(-1, 1, 2)
            if len(pts) > 1:
                cv2.polylines(frame_bgr, [pts], False, CURVE_COLOR, 1, cv2.LINE_AA)

        for pid in range(num_points):
            visible = bool(visibility[t, pid] > vis_threshold)
            if visible:
                x, y = (int(round(float(v))) for v in tracks[t, pid])
                if not was_visible[pid]:
                    segments[pid].append([])            # (re)appeared -> new segment
                segments[pid][-1].append((x, y))
            was_visible[pid] = visible

            if trails[pid]:
                polylines = [np.array(seg, dtype=np.int32).reshape(-1, 1, 2)
                             for seg in segments[pid] if len(seg) > 1]
                if polylines:
                    cv2.polylines(frame_bgr, polylines, False, colors[pid], line_thickness, cv2.LINE_AA)
            if visible:
                if radii[pid] > 0:
                    cv2.circle(frame_bgr, (x, y), radii[pid], colors[pid], -1, cv2.LINE_AA)

        for jt in joints or []:
            draw_joint(frame_bgr, tracks[t], jt["idx"], jt["angle"][t], jt.get("name", ""), line_thickness)

        if boxes is not None and np.all(np.isfinite(boxes[t])):
            x1, y1, x2, y2 = (int(round(v)) for v in boxes[t])
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), BOX_COLOR, line_thickness, cv2.LINE_AA)

        if texts is not None and t < len(texts) and texts[t]:
            draw_label(frame_bgr, texts[t], (8, 22), 0.55)

        yield frame_bgr


JOINT_COLOR = (255, 255, 255)


def draw_joint(img: np.ndarray, pts: np.ndarray, idx: Sequence[int], angle: float, name: str = "",
               thickness: int = 2) -> None:
    """Two limb segments meeting at the joint, with the current angle written next to it."""
    if not np.isfinite(angle):
        return
    a, b, c = (tuple(int(round(float(v))) for v in pts[i]) for i in idx)
    for p, q in ((a, b), (b, c)):
        cv2.line(img, p, q, (0, 0, 0), thickness + 3, cv2.LINE_AA)
        cv2.line(img, p, q, JOINT_COLOR, thickness + 1, cv2.LINE_AA)
    cv2.circle(img, b, thickness + 4, (0, 0, 255), -1, cv2.LINE_AA)
    label = f"{name}: {angle:.0f} deg" if name else f"{angle:.0f} deg"
    draw_label(img, label, (b[0] + 10, b[1] - 8), 0.5)
