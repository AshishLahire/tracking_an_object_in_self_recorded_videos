"""Camera-motion compensation for handheld / panning videos.

The background is tracked with sparse optical flow between consecutive frames; a
similarity transform (rotation + scale + translation) per frame maps every frame into the
coordinates of one reference frame. Applying it to the object's positions removes camera
shake and panning, so paths and speeds reflect the object's own motion (assuming a
mostly distant / planar background).
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def _object_mask(shape, boxes_t: Optional[np.ndarray], center_t: Optional[np.ndarray], radius: int) -> np.ndarray:
    """255 = background (usable for camera-motion estimation), 0 = object area."""
    h, w = shape
    mask = np.full((h, w), 255, np.uint8)
    if boxes_t is not None and np.all(np.isfinite(boxes_t)):
        x1, y1, x2, y2 = boxes_t
        pad_x, pad_y = 0.15 * (x2 - x1), 0.15 * (y2 - y1)
        cv2.rectangle(mask, (int(x1 - pad_x), int(y1 - pad_y)), (int(x2 + pad_x), int(y2 + pad_y)), 0, -1)
    elif center_t is not None and np.all(np.isfinite(center_t)):
        cv2.circle(mask, (int(center_t[0]), int(center_t[1])), radius, 0, -1)
    return mask


def estimate_camera_motion(frames: np.ndarray, reference_frame: int = 0,
                           boxes: Optional[np.ndarray] = None, center: Optional[np.ndarray] = None,
                           max_corners: int = 300) -> np.ndarray:
    """Per-frame 3x3 transforms mapping frame-t pixels into ``reference_frame`` pixels.

    Args:
        frames: (T, H, W, 3) RGB.
        boxes / center: object location per frame (same pixel space as ``frames``), excluded
            from the background so the object's own motion is not mistaken for camera motion.

    Returns:
        (T, 3, 3) float64 array; identity for the reference frame. If motion cannot be
        estimated between two frames, the previous transform is reused.
    """
    T, H, W, _ = frames.shape
    radius = max(H, W) // 8
    step = np.repeat(np.eye(3)[None], T, axis=0)          # step[t]: frame t -> frame t-1

    prev = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    for t in range(1, T):
        cur = cv2.cvtColor(frames[t], cv2.COLOR_RGB2GRAY)
        mask = _object_mask((H, W), None if boxes is None else boxes[t - 1],
                            None if center is None else center[t - 1], radius)
        p_prev = cv2.goodFeaturesToTrack(prev, max_corners, 0.01, 8, mask=mask)
        if p_prev is not None and len(p_prev) >= 6:
            p_cur, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p_prev, None)
            ok = st.ravel() == 1
            if ok.sum() >= 6:
                A, _ = cv2.estimateAffinePartial2D(p_cur[ok], p_prev[ok], method=cv2.RANSAC,
                                                   ransacReprojThreshold=2.0)
                if A is not None:
                    step[t, :2] = A
        prev = cur

    # cumulative: frame t -> frame 0
    to_first = np.empty_like(step)
    to_first[0] = np.eye(3)
    for t in range(1, T):
        to_first[t] = to_first[t - 1] @ step[t]
    # re-anchor on the reference frame: frame t -> reference
    ref_inv = np.linalg.inv(to_first[reference_frame])
    return np.einsum("ij,tjk->tik", ref_inv, to_first)


def apply_transforms(points: np.ndarray, transforms: np.ndarray) -> np.ndarray:
    """Apply per-frame transforms to (T, 2) or (T, N, 2) points (NaNs stay NaN)."""
    pts = np.asarray(points, dtype=np.float64)
    single = pts.ndim == 2
    if single:
        pts = pts[:, None, :]
    homo = np.concatenate([pts, np.ones(pts.shape[:2] + (1,))], axis=-1)     # (T, N, 3)
    out = np.einsum("tij,tnj->tni", transforms, homo)[..., :2]
    return out[:, 0] if single else out


def transform_scale(transforms: np.ndarray) -> np.ndarray:
    """(T,) uniform scale of each transform (how much frame t is shrunk/enlarged into the reference)."""
    return np.sqrt(np.abs(np.linalg.det(transforms[:, :2, :2])))
