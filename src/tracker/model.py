"""Loading and running the CoTracker3 model (downloaded via torch.hub on first use)."""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, Optional

import numpy as np
import torch

HUB_REPO = "facebookresearch/co-tracker"
VARIANTS = {"offline": "cotracker3_offline", "online": "cotracker3_online"}


def get_device(device: Optional[str] = None) -> str:
    """``device`` if given, otherwise ``"cuda"`` when available, else ``"cpu"``."""
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


MODES = ("auto", "offline", "online")


def resolve_mode(mode: str, device: str) -> str:
    """``auto`` -> ``offline`` on GPU (best quality), ``online`` on CPU (bounded memory)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if mode == "auto":
        return "offline" if device.startswith("cuda") else "online"
    return mode


@lru_cache(maxsize=2)
def load_cotracker(device: Optional[str] = None, variant: str = "offline"):
    """Load (and cache) a CoTracker3 model: ``"offline"`` or ``"online"``."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {list(VARIANTS)}")
    model = torch.hub.load(HUB_REPO, VARIANTS[variant])
    model = model.to(get_device(device)).eval()
    return model


def _to_video_tensor(frames: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2)[None].float().to(device)


@torch.inference_mode()
def run_cotracker(frames: np.ndarray, queries: np.ndarray, model=None,
                  device: Optional[str] = None, mode: str = "auto",
                  progress: Optional[Callable[[float], None]] = None, **kwargs):
    """Track query points through a video.

    Args:
        frames: (T, H, W, 3) uint8 RGB frames.
        queries: (N, 3) array of ``[frame_index, x, y]`` in the pixel space of ``frames``.
        model: a loaded CoTracker model matching ``mode`` (loaded automatically if ``None``).
        mode: ``"offline"`` sees the whole video at once (best quality, tracks before the
            query frame too, but memory grows with video length); ``"online"`` processes
            short sliding windows (memory stays flat, only tracks forward from the query
            frame); ``"auto"`` picks offline on GPU and online on CPU.
        progress: optional ``fn(fraction)`` called as the online mode works through the video.
        **kwargs: passed to the offline model call (e.g. ``backward_tracking=True``).

    Returns:
        tracks (T, N, 2) float32 and visibility (T, N) float32 in {0, 1}.
    """
    device = get_device(device)
    mode = resolve_mode(mode, device)
    if model is None:
        model = load_cotracker(device, mode)
    q = torch.as_tensor(np.asarray(queries), dtype=torch.float32, device=device)[None]  # (1, N, 3)

    if mode == "offline":
        tracks, visibility = model(_to_video_tensor(frames, device), queries=q, **kwargs)
    else:
        tracks, visibility = _run_online(model, frames, q, device, progress)

    tracks = tracks[0, : len(frames)].float().cpu().numpy()
    visibility = visibility[0, : len(frames)].float().cpu().numpy()
    return tracks, visibility


def _run_online(model, frames: np.ndarray, q: torch.Tensor, device: str, progress=None):
    """Sliding-window tracking with the online model; only one window is on the device at a time."""
    T = len(frames)
    step = model.step
    # Pad with copies of the last frame so the final windows cover every real frame
    n_windows = max(1, -(-max(T - step, 1) // step))
    padded_len = n_windows * step + step
    if padded_len > T:
        frames = np.concatenate([frames, np.repeat(frames[-1:], padded_len - T, axis=0)])

    model(video_chunk=_to_video_tensor(frames[:1], device), is_first_step=True, queries=q)
    tracks = visibility = None
    for i, start in enumerate(range(0, n_windows * step, step)):
        tracks, visibility = model(video_chunk=_to_video_tensor(frames[start:start + 2 * step], device))
        if progress:
            progress((i + 1) / n_windows)
    return tracks, visibility
