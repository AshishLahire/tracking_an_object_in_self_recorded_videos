"""Track several objects in one video, each with its own box, path, speed, crossings and
speed-trap passes. All objects are tracked in a single model run."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from .analysis import line_crossings, meters_per_pixel, motion_stats, zone_events
from .draw import point_colors, render_objects
from .selection import boxes_from_tracks, normalize_box, object_center, object_point_mask, points_in_box
from .smoothing import smooth_tracks
from .speedtrap import speed_trap as run_speed_trap
from .video_io import VideoData, load_video, to_browser_mp4, write_video


@dataclass
class ObjectTrack:
    name: str
    color: tuple                        # BGR
    point_idx: list                     # indices into the tracked points
    boxes: Optional[np.ndarray] = None  # (T, 4) loaded-frame pixels, NaN when hidden
    center: Optional[np.ndarray] = None # (T, 2) ORIGINAL pixels, NaN when hidden
    visible_fraction: float = 0.0
    motion: dict = field(default_factory=dict)
    crossings: list = field(default_factory=list)
    zones: list = field(default_factory=list)
    passes: list = field(default_factory=list)     # speed-trap passes

    def summary(self) -> dict:
        m = {k: (round(v, 3) if isinstance(v, float) and np.isfinite(v) else (None if isinstance(v, float) else v))
             for k, v in self.motion.items() if k != "speed"}
        return {"name": self.name, "visible_fraction": round(self.visible_fraction, 3), "motion": m,
                "line_crossings": self.crossings, "zones": self.zones, "speed_trap_passes": self.passes}


@dataclass
class MultiResult:
    video: VideoData
    objects: list
    tracks: np.ndarray
    visibility: np.ndarray
    speed_trap: Optional[dict] = None
    output_path: Optional[str] = None
    csv_path: Optional[str] = None
    passes_csv_path: Optional[str] = None
    json_path: Optional[str] = None

    @property
    def passes(self) -> list:
        """All speed-trap passes, of all objects, in time order."""
        out = [dict(p, object=o.name) for o in self.objects for p in o.passes]
        return sorted(out, key=lambda p: p["enter_s"])

    def summary(self) -> dict:
        v = self.video
        passes = self.passes
        trap = None
        if self.speed_trap:
            speeds = [p["speed_kmh"] for p in passes]
            trap = {**{k: self.speed_trap[k] for k in ("distance_m", "limit_kmh")},
                    "passes": len(passes), "over_limit": sum(p["over_limit"] for p in passes),
                    "mean_speed_kmh": round(float(np.mean(speeds)), 2) if speeds else None,
                    "max_speed_kmh": round(float(np.max(speeds)), 2) if speeds else None}
        return {
            "video": {"frames": v.num_frames, "fps": round(v.fps, 3), "original_size": list(v.original_size),
                      "processed_size": list(v.size), "duration_s": round(v.num_frames / v.fps, 3)},
            "objects": [o.summary() for o in self.objects],
            "speed_trap": trap,
        }


def _parse_object(obj, i: int) -> dict:
    if isinstance(obj, dict):
        d = dict(obj)
    elif len(obj) == 4 and all(np.isscalar(v) for v in obj):
        d = {"box": tuple(obj)}
    else:
        d = {"points": list(obj)}
    if "box" not in d and not d.get("points"):
        raise ValueError(f"Object {i + 1} needs a 'box' or 'points'")
    d.setdefault("name", f"object {i + 1}")
    return d


def track_objects(video_path: str, objects: Sequence, query_frame: int = 0, box_grid: int = 4,
                  speed_trap: Optional[dict] = None,
                  count_lines: Optional[Sequence[Sequence[float]]] = None,
                  zones: Optional[Sequence[Sequence[Sequence[float]]]] = None,
                  reference: Optional[Sequence[float]] = None,
                  output_path: Optional[str] = None, csv_path: Optional[str] = None,
                  json_path: Optional[str] = None,
                  max_side: Optional[int] = 640, max_frames: Optional[int] = None,
                  smooth: bool = True, device: Optional[str] = None, mode: str = "auto",
                  model_resolution=None, browser_compatible: bool = False,
                  tracker_fn: Optional[Callable] = None,
                  progress: Optional[Callable[[float, str], None]] = None) -> MultiResult:
    """Track several objects through a video.

    All coordinates are in ORIGINAL video pixels, selected on ``query_frame``.

    Args:
        objects: one entry per object: a box ``(x1, y1, x2, y2)``, a list of ``(x, y)`` points,
            or a dict ``{"box": ..., "points": ..., "name": ...}``.
        speed_trap: ``{"line_a": (x1, y1, x2, y2), "line_b": (...), "distance_m": float,
            "limit_kmh": optional float}``. Each object's speed is measured from the time it takes
            to get from one line to the other. Needs a still camera.
        count_lines / zones / reference: as in :func:`tracker.pipeline.track_video`, per object.
        csv_path: one row per frame per object; also writes ``<name>_speed_trap.csv`` when a
            speed trap is set.
    """
    def report(frac, msg):
        if progress:
            progress(frac, msg)

    objs = [_parse_object(o, i) for i, o in enumerate(objects)]
    if not objs:
        raise ValueError("Give at least one object")
    if speed_trap:
        for k in ("line_a", "line_b", "distance_m"):
            if speed_trap.get(k) is None:
                raise ValueError(f"speed_trap needs '{k}'")

    report(0.0, "Loading video")
    video = load_video(video_path, max_side=max_side, max_frames=max_frames)
    s = video.scale
    if not (0 <= query_frame < video.num_frames):
        raise ValueError(f"Query frame {query_frame} is outside the video (0..{video.num_frames - 1})")

    # one query list for all objects: object i owns points point_idx[i]
    all_points, owners = [], []
    for o in objs:
        pts = list(o.get("points") or [])
        if "box" in o:
            o["box"] = normalize_box(o["box"])
            pts += points_in_box(o["box"], box_grid)
        owners.append(list(range(len(all_points), len(all_points) + len(pts))))
        all_points += pts
    q = np.array([[query_frame, x * s, y * s] for x, y in all_points], np.float32)

    msg = f"Tracking {len(objs)} object(s), {len(q)} points, {video.num_frames} frames"
    report(0.1, msg)
    if tracker_fn is None:
        from .model import run_cotracker
        tracks, visibility = run_cotracker(video.frames, q, device=device, mode=mode, model_resolution=model_resolution,
                                           progress=lambda f: report(0.1 + 0.7 * f, msg))
    else:
        tracks, visibility = tracker_fn(video.frames, q)
    tracks = np.asarray(tracks, np.float32)
    visibility = np.asarray(visibility, np.float32)
    if tracks.shape[:2] != (video.num_frames, len(q)):
        raise RuntimeError(f"tracker returned shape {tracks.shape}, expected ({video.num_frames}, {len(q)}, 2)")
    smoothed = smooth_tracks(tracks, visibility) if smooth else tracks.copy()

    report(0.8, "Measuring")
    m_per_px = None
    if reference is not None:
        x1, y1, x2, y2, length_m = reference
        m_per_px = meters_per_pixel((x1, y1), (x2, y2), length_m)

    colors = point_colors(len(objs), seed=1)
    results = []
    for o, idx, color in zip(objs, owners, colors):
        tr, vi = smoothed[:, idx], visibility[:, idx]
        ot = ObjectTrack(name=str(o["name"]), color=color, point_idx=idx)
        boxes = None
        if "box" in o and not o.get("points"):
            box_px = [v * s for v in o["box"]]
            mask = object_point_mask(tr, vi, box_px, query_frame)
            boxes = boxes_from_tracks(tr, vi, box_px, query_frame, point_mask=mask)
        ot.boxes = boxes
        center_px = object_center(tr, vi, boxes=boxes)
        ot.center = center_px / s
        ot.visible_fraction = float(np.mean(np.all(np.isfinite(center_px), axis=1)))
        ot.motion = motion_stats(ot.center, video.fps, m_per_px)
        for i, line in enumerate(count_lines or []):
            ot.crossings += [{"line": i, **ev} for ev in line_crossings(ot.center, line, video.fps)]
        for i, zone in enumerate(zones or []):
            ot.zones.append({"zone": i, **zone_events(ot.center, zone, video.fps)})
        if speed_trap:
            ot.passes = run_speed_trap(ot.center, speed_trap["line_a"], speed_trap["line_b"],
                                       float(speed_trap["distance_m"]), video.fps, speed_trap.get("limit_kmh"))
        results.append(ot)

    result = MultiResult(video=video, objects=results, tracks=smoothed, visibility=visibility,
                         speed_trap=dict(speed_trap, limit_kmh=speed_trap.get("limit_kmh")) if speed_trap else None)

    if output_path:
        report(0.85, "Rendering video")
        trap_lines = None
        if speed_trap:
            trap_lines = [[v * s for v in speed_trap["line_a"]] + ["A"], [v * s for v in speed_trap["line_b"]] + ["B"]]
        frames = render_objects(
            video.frames,
            [{"name": o.name, "color": o.color, "boxes": o.boxes, "center": o.center * s, "passes": o.passes}
             for o in results],
            fps=video.fps,
            lines=[[v * s for v in ln] for ln in (count_lines or [])],
            zones=[[(x * s, y * s) for x, y in z] for z in (zones or [])],
            trap_lines=trap_lines, limit_kmh=(speed_trap or {}).get("limit_kmh"))
        write_video(output_path, frames, video.fps)
        if browser_compatible:
            to_browser_mp4(output_path)
        result.output_path = output_path

    if csv_path:
        _save_objects_csv(csv_path, results, video.fps, s)
        result.csv_path = csv_path
        if speed_trap:
            stem, _ = os.path.splitext(csv_path)
            result.passes_csv_path = f"{stem}_speed_trap.csv"
            _save_passes_csv(result.passes_csv_path, result.passes)

    if json_path:
        os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result.summary(), f, indent=2, default=float)
        result.json_path = json_path

    report(1.0, "Done")
    return result


def _fmt(v):
    return "" if v is None or not np.isfinite(v) else f"{v:.2f}"


def _save_objects_csv(path: str, objects: list, fps: float, scale: float = 1.0) -> None:
    """One row per frame per object; positions and boxes in original-video pixels."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    unit = objects[0].motion.get("unit_speed", "px/s").replace("/", "_per_")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "object", "center_x", "center_y", "visible", f"speed_{unit}",
                    "box_x1", "box_y1", "box_x2", "box_y2"])
        T = len(objects[0].center)
        for t in range(T):
            for o in objects:
                cx, cy = o.center[t]
                box = o.boxes[t] / scale if o.boxes is not None else [np.nan] * 4
                w.writerow([t, f"{t / fps:.4f}", o.name, _fmt(cx), _fmt(cy), int(np.isfinite(cx)),
                            _fmt(o.motion["speed"][t])] + [_fmt(v) for v in box])


def _save_passes_csv(path: str, passes: list) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cols = ["object", "from", "to", "enter_s", "exit_s", "seconds", "speed_kmh", "speed_ms", "over_limit"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for p in passes:
            w.writerow([p[c] for c in cols])
