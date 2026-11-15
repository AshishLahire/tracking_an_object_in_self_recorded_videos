import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


@pytest.fixture
def make_video(tmp_path):
    """Create a small synthetic .mp4 with a white square moving left -> right."""

    def _make(width=320, height=180, frames=40, fps=25.0, name="clip.mp4"):
        path = str(tmp_path / name)
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        for t in range(frames):
            img = np.zeros((height, width, 3), np.uint8)
            x = int(10 + (width - 40) * t / max(frames - 1, 1))
            cv2.rectangle(img, (x, height // 2 - 10), (x + 20, height // 2 + 10), (255, 255, 255), -1)
            writer.write(img)
        writer.release()
        return path

    return _make


def fake_tracker(frames, queries):
    """Stand-in for CoTracker: points move right 2 px/frame, hidden in frames 10-14."""
    T, N = frames.shape[0], queries.shape[0]
    t = np.arange(T, dtype=np.float32)[:, None]
    tracks = np.stack([queries[None, :, 1] + 2 * t, np.repeat(queries[None, :, 2], T, 0)], axis=-1)
    vis = np.ones((T, N), np.float32)
    vis[10:15] = 0
    return tracks.astype(np.float32), vis
