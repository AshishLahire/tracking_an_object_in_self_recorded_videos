"""Object tracking in self-recorded videos with CoTracker3.

Typical use::

    from tracker import track_video
    result = track_video("examples/bicycle.mp4", points=[(125, 50)], query_frame=70,
                         output_path="results/bicycle_tracked.mp4")
    print(result.summary())
"""

from .video_io import (VideoData, load_video, write_video, save_tracks_csv, save_object_csv,
                       to_browser_mp4, make_gif)
from .smoothing import visible_runs, smooth_tracks
from .draw import point_colors, render_frames
from .metrics import evaluate_tracking, format_metrics
from .selection import points_in_box, boxes_from_tracks, object_center, object_point_mask
from .analysis import meters_per_pixel, motion_stats, object_scale_m_per_px, line_crossings, zone_events, heatmap, heatmap_overlay
from .joints import angle_at, joint_angle_series, repetition_stats, analyze_joints
from .stabilize import estimate_camera_motion, apply_transforms, transform_scale
from .pipeline import TrackingResult, make_queries, track_video

__all__ = [
    "VideoData", "load_video", "write_video", "save_tracks_csv", "save_object_csv",
    "to_browser_mp4", "make_gif",
    "visible_runs", "smooth_tracks",
    "point_colors", "render_frames",
    "evaluate_tracking", "format_metrics",
    "points_in_box", "boxes_from_tracks", "object_center", "object_point_mask",
    "meters_per_pixel", "motion_stats", "object_scale_m_per_px", "line_crossings", "zone_events", "heatmap", "heatmap_overlay",
    "estimate_camera_motion", "apply_transforms", "transform_scale",
    "angle_at", "joint_angle_series", "repetition_stats", "analyze_joints",
    "TrackingResult", "make_queries", "track_video",
]

__version__ = "0.3.0"
