"""End-to-end pipeline: video + selection -> tracked video, CSVs, measurements and metrics."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import cv2
import numpy as np

from .analysis import (heatmap, heatmap_overlay, line_crossings, meters_per_pixel, motion_stats,
                       object_scale_m_per_px, zone_events)
from .draw import render_frames
from .metrics import evaluate_tracking
from .selection import boxes_from_tracks, normalize_box, object_center, object_point_mask, points_in_box
from .smoothing import smooth_tracks
from .stabilize import apply_transforms, estimate_camera_motion, transform_scale
from .video_io import VideoData, load_video, save_object_csv, save_tracks_csv, to_browser_mp4, write_video


def _clean(value):
    """Round floats and turn NaN/inf into None so the summary is valid JSON."""
    if isinstance(value, (float, np.floating)):
        return round(float(value), 4) if np.isfinite(value) else None
    return value


@dataclass
class TrackingResult:
    video: VideoData
    queries: np.ndarray                 # (N, 3) [frame, x, y] in loaded-frame pixels
    tracks: np.ndarray                  # (T, N, 2) raw model output, loaded-frame pixels
    visibility: np.ndarray              # (T, N)
    smoothed: np.ndarray                # (T, N, 2) tracks that were drawn
    query_frame: int = 0
    boxes: Optional[np.ndarray] = None  # (T, 4) object box, loaded-frame pixels (box mode)
    object_points: Optional[np.ndarray] = None  # (N,) bool, box mode: points judged to be on the object
    center: Optional[np.ndarray] = None # (T, 2) object position, ORIGINAL pixels (camera-compensated if stabilize)
    motion: dict = field(default_factory=dict)
    crossings: list = field(default_factory=list)
    zones: list = field(default_factory=list)
    heatmap_image: Optional[np.ndarray] = None   # RGB, loaded-frame size
    metrics: dict = field(default_factory=dict)
    raw_metrics: dict = field(default_factory=dict)
    stabilized: bool = False
    output_path: Optional[str] = None
    csv_path: Optional[str] = None
    object_csv_path: Optional[str] = None
    json_path: Optional[str] = None
    heatmap_path: Optional[str] = None

    def summary(self) -> dict:
        """JSON-serialisable summary of the run (no per-frame arrays)."""
        v = self.video
        motion = {k: _clean(val) for k, val in self.motion.items() if k != "speed"}
        return {
            "video": {"frames": v.num_frames, "fps": round(v.fps, 3), "original_size": list(v.original_size),
                      "processed_size": list(v.size), "duration_s": round(v.num_frames / v.fps, 3)},
            "query_frame": self.query_frame,
            "num_points": int(self.queries.shape[0]),
            "box_mode": self.boxes is not None,
            "camera_compensated": self.stabilized,
            "motion": motion,
            "line_crossings": self.crossings,
            "zones": self.zones,
            "metrics": {k: _clean(val) for k, val in self.metrics.items()},
        }


def make_queries(points: Sequence[Sequence[float]], query_frame: int = 0, scale: float = 1.0,
                 num_frames: Optional[int] = None) -> np.ndarray:
    """Build an (N, 3) ``[frame, x, y]`` query array.

    Each point is ``(x, y)`` (uses ``query_frame``) or ``(frame, x, y)``, given in
    ORIGINAL video pixels; x/y are multiplied by ``scale`` to match the loaded frames.
    """
    rows = []
    for p in points:
        if len(p) == 2:
            f, x, y = query_frame, p[0], p[1]
        elif len(p) == 3:
            f, x, y = p
        else:
            raise ValueError(f"Point must be (x, y) or (frame, x, y), got {p!r}")
        if num_frames is not None and not (0 <= int(f) < num_frames):
            raise ValueError(f"Query frame {f} is outside the video (0..{num_frames - 1})")
        rows.append([float(int(f)), float(x) * scale, float(y) * scale])
    if not rows:
        raise ValueError("At least one point (or a box) is required")
    return np.array(rows, dtype=np.float32)


def _speed_texts(speed: np.ndarray, unit: str, crossings: list, fps: float) -> list:
    counts = np.zeros(len(speed), int)
    for c in crossings:
        counts[c["frame"]:] += 1
    texts = []
    for t, s in enumerate(speed):
        parts = []
        if np.isfinite(s):
            parts.append(f"{s:.1f} {unit}")
        if crossings:
            parts.append(f"crossings: {counts[t]}")
        texts.append("  |  ".join(parts) if parts else None)
    return texts


def track_video(video_path: str, points: Sequence[Sequence[float]] = (), query_frame: int = 0,
                box: Optional[Sequence[float]] = None, box_grid: int = 4,
                output_path: Optional[str] = None, csv_path: Optional[str] = None,
                json_path: Optional[str] = None, heatmap_path: Optional[str] = None,
                max_side: Optional[int] = 640, max_frames: Optional[int] = None,
                smooth: bool = True, device: Optional[str] = None, mode: str = "auto",
                reference: Optional[Sequence[float]] = None, reference_is_object: bool = False,
                count_lines: Optional[Sequence[Sequence[float]]] = None,
                zones: Optional[Sequence[Sequence[Sequence[float]]]] = None,
                stabilize: bool = False, browser_compatible: bool = False,
                show_speed: bool = True,
                tracker_fn: Optional[Callable] = None,
                progress: Optional[Callable[[float, str], None]] = None,
                **draw_kwargs) -> TrackingResult:
    """Track an object through a video and measure its motion.

    All coordinates are in ORIGINAL video pixels, selected on ``query_frame``.

    Args:
        points: ``(x, y)`` or ``(frame, x, y)`` points on the object.
        box: ``(x1, y1, x2, y2)`` around the object; a ``box_grid`` x ``box_grid`` grid of
            points inside it is tracked and the box is moved with them.
        output_path / csv_path / json_path / heatmap_path: outputs to write (each optional).
            ``csv_path`` writes per-point rows plus ``<name>_object.csv`` with one row per frame.
        max_side / max_frames: see :func:`tracker.video_io.load_video`.
        smooth: occlusion-aware smoothing of the tracks before drawing / measuring.
        device / mode: see :func:`tracker.model.run_cotracker`.
        reference: ``(x1, y1, x2, y2, length_m)``: a segment of known real length, so speed
            and distance are reported in km/h and metres instead of pixels.
        reference_is_object: the reference was measured on the tracked object itself (e.g. the
            car's length). Needs ``box``; the scale then follows the object's size, so speed stays
            right when it moves toward or away from the camera.
        count_lines: ``(x1, y1, x2, y2)`` lines; every crossing of the object is reported.
        zones: polygons ``[(x, y), ...]``; entries, exits and time inside are reported.
        stabilize: compensate camera shake / panning before measuring (handheld videos).
        browser_compatible: re-encode the output video to H.264 (needs imageio-ffmpeg / ffmpeg).
        show_speed: write the current speed (and crossing count) on the video.
        tracker_fn: ``fn(frames, queries) -> (tracks, visibility)``; defaults to CoTracker3.
        progress: optional ``fn(fraction, message)`` callback.
        **draw_kwargs: forwarded to :func:`tracker.draw.render_frames`.
    """
    def report(frac, msg):
        if progress:
            progress(frac, msg)

    report(0.0, "Loading video")
    video = load_video(video_path, max_side=max_side, max_frames=max_frames)
    s = video.scale
    if not (0 <= query_frame < video.num_frames):
        raise ValueError(f"Query frame {query_frame} is outside the video (0..{video.num_frames - 1})")

    all_points = list(points)
    if box is not None:
        box = normalize_box(box)
        all_points += points_in_box(box, box_grid)
    queries = make_queries(all_points, query_frame, s, video.num_frames)

    report(0.1, f"Tracking {len(queries)} point(s) over {video.num_frames} frames")
    if tracker_fn is None:
        from .model import run_cotracker          # imported lazily: needs torch + download
        msg = f"Tracking {len(queries)} point(s) over {video.num_frames} frames"
        tracks, visibility = run_cotracker(video.frames, queries, device=device, mode=mode,
                                           progress=lambda f: report(0.1 + 0.65 * f, msg))
    else:
        tracks, visibility = tracker_fn(video.frames, queries)
    tracks = np.asarray(tracks, dtype=np.float32)
    visibility = np.asarray(visibility, dtype=np.float32)
    if tracks.shape[:2] != (video.num_frames, len(queries)):
        raise RuntimeError(f"tracker returned shape {tracks.shape}, expected ({video.num_frames}, {len(queries)}, 2)")

    report(0.75, "Measuring motion")
    smoothed = smooth_tracks(tracks, visibility) if smooth else tracks.copy()
    result = TrackingResult(
        video=video, queries=queries, tracks=tracks, visibility=visibility, smoothed=smoothed,
        query_frame=query_frame,
        metrics=evaluate_tracking(smoothed, visibility),
        raw_metrics=evaluate_tracking(tracks, visibility),
    )

    # ---- object position (loaded pixels) ------------------------------------------------
    boxes = None
    if box is not None:
        box_px = [v * s for v in box]
        result.object_points = object_point_mask(smoothed, visibility, box_px, query_frame)
        boxes = boxes_from_tracks(smoothed, visibility, box_px, query_frame, point_mask=result.object_points)
        result.boxes = boxes
    center_px = object_center(smoothed, visibility, boxes=boxes)

    measure_px = center_px
    zoom = None
    if stabilize:
        transforms = estimate_camera_motion(video.frames, query_frame, boxes=boxes, center=center_px)
        measure_px = apply_transforms(center_px, transforms)
        zoom = transform_scale(transforms)
        result.stabilized = True
    result.center = measure_px / s                       # original pixels

    # ---- measurements (original pixels) -------------------------------------------------
    m_per_px = None
    if reference is not None:
        x1, y1, x2, y2, length_m = reference
        m_per_px = meters_per_pixel((x1, y1), (x2, y2), length_m)
        if reference_is_object:
            if boxes is None:
                raise ValueError("reference_is_object needs a box around the object")
            m_per_px = object_scale_m_per_px(m_per_px, boxes, query_frame, zoom)
    result.motion = motion_stats(result.center, video.fps, m_per_px)
    for i, line in enumerate(count_lines or []):
        for ev in line_crossings(result.center, line, video.fps):
            result.crossings.append({"line": i, **ev})
    result.crossings.sort(key=lambda e: e["frame"])
    for i, zone in enumerate(zones or []):
        result.zones.append({"zone": i, **zone_events(result.center, zone, video.fps)})

    heat = heatmap(measure_px, video.size)
    result.heatmap_image = heatmap_overlay(video.frames[query_frame], heat)
    if heatmap_path:
        os.makedirs(os.path.dirname(os.path.abspath(heatmap_path)), exist_ok=True)
        cv2.imwrite(heatmap_path, cv2.cvtColor(result.heatmap_image, cv2.COLOR_RGB2BGR))
        result.heatmap_path = heatmap_path

    # ---- outputs --------------------------------------------------------------------------
    if output_path:
        report(0.85, "Rendering video")
        n = len(queries)
        draw_tracks, draw_vis = smoothed, visibility
        trails, radii = True, None
        if boxes is not None:
            # box mode: small dots for the grid points, one trail for the object centre
            c_vis = np.all(np.isfinite(center_px), axis=1).astype(np.float32)[:, None]
            draw_tracks = np.concatenate([smoothed, np.nan_to_num(center_px)[:, None, :]], axis=1)
            draw_vis = np.concatenate([visibility, c_vis], axis=1)
            trails = [False] * n + [True]
            radii = [2 if on else 0 for on in result.object_points] + [5]   # hide background points
        texts = None
        if show_speed and (reference is not None or count_lines):
            texts = _speed_texts(result.motion["speed"], result.motion["unit_speed"], result.crossings, video.fps)
        lines = [[v * s for v in ln] for ln in (count_lines or [])]
        polys = [[(x * s, y * s) for x, y in z] for z in (zones or [])]
        frames_iter = render_frames(video.frames, draw_tracks, draw_vis, boxes=boxes, lines=lines,
                                    zones=polys, texts=texts, draw_trails=trails, point_radii=radii,
                                    **draw_kwargs)
        write_video(output_path, frames_iter, video.fps)
        if browser_compatible:
            to_browser_mp4(output_path)
        result.output_path = output_path

    if csv_path:
        save_tracks_csv(csv_path, smoothed, visibility, video.fps, s)
        result.csv_path = csv_path
        stem, _ = os.path.splitext(csv_path)
        result.object_csv_path = f"{stem}_object.csv"
        save_object_csv(result.object_csv_path, result.center, video.fps, result.motion["speed"],
                        result.motion["unit_speed"], None if boxes is None else boxes / s)

    if json_path:
        os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result.summary(), f, indent=2, default=float)
        result.json_path = json_path

    report(1.0, "Done")
    return result
