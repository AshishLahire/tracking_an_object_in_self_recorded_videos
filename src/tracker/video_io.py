"""Reading and writing videos, and exporting track data."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Iterable, Optional

import cv2
import numpy as np

DEFAULT_FPS = 30.0


@dataclass
class VideoData:
    """Frames of a video plus the information needed to map back to the original."""

    frames: np.ndarray            # (T, H, W, 3) uint8, RGB, possibly downscaled
    fps: float                    # frame rate of the source video
    original_size: tuple          # (W, H) of the source video
    scale: float = 1.0            # loaded size = original size * scale

    @property
    def num_frames(self) -> int:
        return int(self.frames.shape[0])

    @property
    def size(self) -> tuple:
        """(W, H) of the loaded frames."""
        return int(self.frames.shape[2]), int(self.frames.shape[1])


def _even(value: float) -> int:
    """Round to the nearest even integer (>= 2); most video encoders need even sizes."""
    return max(2, 2 * int(round(value / 2)))


def load_video(path: str, max_side: Optional[int] = 640, max_frames: Optional[int] = None) -> VideoData:
    """Load a video as RGB frames.

    Args:
        path: video file path.
        max_side: frames whose longer side exceeds this are downscaled (keeps memory
            bounded for long / high-resolution videos). ``None`` keeps the original size.
        max_frames: stop after this many frames (``None`` reads the whole video).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Video not found: {path}")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or np.isnan(fps):
        fps = DEFAULT_FPS

    # Frames are resized as they are read and written into one preallocated array, so a long
    # 1080p/4K video never sits in memory at full resolution (or twice, as list + stacked copy).
    expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if max_frames is not None:
        expected = min(expected, max_frames) if expected > 0 else max_frames
    frames = None
    n = 0
    h0 = w0 = 0
    scale, new_size = 1.0, None
    try:
        while max_frames is None or n < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if frames is None:
                h0, w0 = frame.shape[:2]
                if max_side is not None and max(h0, w0) > max_side:
                    scale = max_side / max(h0, w0)
                    new_size = (_even(w0 * scale), _even(h0 * scale))
                h, w = (new_size[1], new_size[0]) if new_size else (h0, w0)
                frames = np.empty((max(expected, 1), h, w, 3), np.uint8)
            if new_size:
                frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
            if n >= len(frames):                               # container under-reported its length
                frames = np.concatenate([frames, np.empty((max(16, len(frames) // 2),) + frames.shape[1:], np.uint8)])
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, dst=frames[n])
            n += 1
    finally:
        cap.release()

    if n == 0:
        raise ValueError(f"No frames could be read from: {path}")

    return VideoData(frames[:n], float(fps), (w0, h0), scale)


def write_video(path: str, frames_bgr: Iterable[np.ndarray], fps: float) -> int:
    """Write BGR frames to an .mp4 file. Returns the number of frames written."""
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)

    writer = None
    count = 0
    try:
        for frame in frames_bgr:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                if not writer.isOpened():
                    raise IOError(f"Could not open video writer for: {path}")
            writer.write(frame)
            count += 1
    finally:
        if writer is not None:
            writer.release()
    return count


def save_tracks_csv(path: str, tracks: np.ndarray, visibility: np.ndarray, fps: float,
                    scale: float = 1.0, vis_threshold: float = 0.5) -> None:
    """Save tracks as CSV with one row per (frame, point).

    Coordinates are written in ORIGINAL video pixels (divided by ``scale``).
    Columns: frame, time_s, point_id, x, y, visible
    """
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    T, N, _ = tracks.shape
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "point_id", "x", "y", "visible"])
        for t in range(T):
            for p in range(N):
                x, y = tracks[t, p] / scale
                w.writerow([t, f"{t / fps:.4f}", p, f"{x:.2f}", f"{y:.2f}",
                            int(visibility[t, p] > vis_threshold)])


def save_object_csv(path: str, center: np.ndarray, fps: float, speed: Optional[np.ndarray] = None,
                    speed_unit: str = "", boxes: Optional[np.ndarray] = None) -> None:
    """One row per frame for the whole object: centre position, speed and box (original pixels)."""
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    header = ["frame", "time_s", "center_x", "center_y", "visible"]
    if speed is not None:
        header.append(f"speed_{speed_unit.replace('/', '_per_')}" if speed_unit else "speed")
    if boxes is not None:
        header += ["box_x1", "box_y1", "box_x2", "box_y2"]

    def fmt(v):
        return "" if not np.isfinite(v) else f"{v:.2f}"

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for t in range(len(center)):
            cx, cy = center[t]
            row = [t, f"{t / fps:.4f}", fmt(cx), fmt(cy), int(np.isfinite(cx))]
            if speed is not None:
                row.append(fmt(speed[t]))
            if boxes is not None:
                row += [fmt(v) for v in boxes[t]]
            w.writerow(row)


def ffmpeg_exe() -> Optional[str]:
    """Path to an ffmpeg binary (bundled by imageio-ffmpeg, else from PATH), or None."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        import shutil
        return shutil.which("ffmpeg")


def to_browser_mp4(path: str) -> bool:
    """Re-encode an .mp4 in place to H.264 so it plays in web browsers.

    OpenCV's default 'mp4v' codec does not play in most browsers. Returns False (and
    leaves the file unchanged) when no ffmpeg is available or encoding fails.
    """
    import subprocess
    import tempfile

    exe = ffmpeg_exe()
    if not exe:
        return False
    fd, tmp = tempfile.mkstemp(suffix=".mp4", dir=os.path.dirname(os.path.abspath(path)))
    os.close(fd)
    cmd = [exe, "-y", "-loglevel", "error", "-i", path, "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-preset", "veryfast", "-crf", "23", "-movflags", "+faststart", tmp]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        os.replace(tmp, path)
        return True
    except (subprocess.CalledProcessError, OSError):
        if os.path.exists(tmp):
            os.remove(tmp)
        return False


def make_gif(video_path: str, gif_path: str, width: int = 360, fps: int = 10,
             start_s: float = 0.0, duration_s: Optional[float] = None) -> bool:
    """Make a small looping GIF preview of a video (for README / docs). Needs ffmpeg."""
    import subprocess

    exe = ffmpeg_exe()
    if not exe:
        return False
    os.makedirs(os.path.dirname(os.path.abspath(gif_path)), exist_ok=True)
    filt = (f"fps={fps},scale={width}:-2:flags=lanczos,split[a][b];"
            f"[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=4")
    cmd = [exe, "-y", "-loglevel", "error", "-ss", str(start_s)]
    if duration_s:
        cmd += ["-t", str(duration_s)]
    cmd += ["-i", video_path, "-vf", filt, "-loop", "0", gif_path]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, OSError):
        return False
